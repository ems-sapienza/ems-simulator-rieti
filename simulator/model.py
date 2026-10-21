"""
Discrete-event simulation model (SimPy) of the ASI emergency medical service of
a province of Lazio.

The simulator reads the parameters already estimated in `data/parameters/`
(produced by `preprocessing/estimate_parameters.py`): it never touches the
mission-level microdata. The calibration uses the anonymised instances in
`data/anonymized-instances/`.

The time samplers (triage, assignment, departure, on-scene times, cancelled
missions, block at the emergency department, interarrivals) are built once: the
ECDF interpolator is not rebuilt at every event. Values are drawn one at a
time, at the exact point where they are needed: changing the order of the draws
changes the sequence of random numbers, and with it the results of a given seed
(simulator/report_*).

Stations are identified internally by the code of the grid cell (e.g. "Q17_12"),
because that is the key the travel-time matrices are indexed by; `STATION_NAME`
maps every code to the name of the station, used in the logs and in the reports.
Capacities and scenarios can be defined either by name or by code (see
`station_code`).

File layout
-----------
 1. Configuration and constants
 2. General utilities (seed, log, simulation time, time bands)
 3. Empirical distributions and pre-computed times
 4. Loading of the parameters and of the travel-time matrices
 5. Statistics collected by the simulation
 6. Simulation model (AmbulanceSystem, CallGenerator)
 7. Running the replications
 8. KPIs and Excel reports
 9. Calibration (entry point)
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import simpy
from scipy import stats
from scipy.interpolate import interp1d

# =============================================================================
# 1. CONFIGURATION AND CONSTANTS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EMS_DATA_DIR", ROOT_DIR / "data"))
PARAMS_DIR = DATA_DIR / "parameters"
SPATIAL_DIR = DATA_DIR / "spatial"
INSTANCES_DIR = DATA_DIR / "anonymized-instances"

PROVINCE = "RI"
VEHICLE_TYPE = "ASI"

BASE_SEED = 1234

# Replications run in parallel. Every replication has its own seed and is
# deterministic, so the number of processes does not change the results.
# Beyond a handful of processes memory grows more than the time gained, because
# each of them loads its own copy of the travel-time matrices.
WORKERS_MAX = 6

DEBUG = False            # True to write the event log to LOG_FILE
LOG_FILE = "log.txt"
LOG_LIMIT = 1000         # maximum number of log lines per run

# ---- Time horizon (minutes) -------------------------------------------------
MINUTES_PER_DAY = 1440
WARMUP_DAYS = 15
SIM_DAYS = 365
WARMUP_MINUTES = WARMUP_DAYS * MINUTES_PER_DAY
SIM_END_MINUTES = WARMUP_MINUTES + SIM_DAYS * MINUTES_PER_DAY
SIM_START_DATETIME = datetime(2023, 1, 1)

# ---- Operational parameters -------------------------------------------------
DAY_START_HOUR, DAY_END_HOUR = 8, 20      # daytime shift of the stations
TRAVEL_JITTER_DELTA = 5                   # ± minutes of variability on travel times
MAX_WAIT_TRAVEL_TIME = 120                # beyond this time the station does not queue
NON_EMERGENCY_LIMIT = 60                  # assignment limit for codes V/B
UNDER_THRESHOLD_LIMIT = 20                # threshold used by the dynamic limit
MIN_STATIONS_CONSIDERED = 3               # min. number of stations considered
PRIORITY_BASE = 10 ** 6                   # separates the priority blocks by colour code

THRESHOLDS = (10, 15, 20, 30, 40, 50, 60)  # coverage thresholds (minutes)
CVAR_ALPHAS = (0.10, 0.05, 0.01)           # worst tails for the CVaR
GROUPS = ("ALL", "EMER", "NONEMER")

# ---- Encodings --------------------------------------------------------------
EMERGENCY_CODES = (1, 2)                   # R, G
COLOUR_BY_CODE = {1: "R", 2: "G", 3: "V", 4: "B"}
CODE_BY_COLOUR = {"R": 1, "G": 2, "V": 3, "B": 4, "N": 5}
# 5 = "N" (deceased on scene): on-scene times use the urgent distributions
CRIT_GROUP_BY_CODE = {1: "RG", 2: "RG", 3: "VB", 4: "VB", 5: "RG"}
URGENCY_ORDER = ["R", "G", "V", "B"]
DECEASED_ON_SCENE_CODE = 5                 # urgency "N": no transport

NETWORK_ORDER = [
    "Gen", "Rete Trauma", "Rete Ictus", "Sindrome Coronarica Acuta",
    "Aritmie Gravi e Scompenso Cardiaco", "Rete Emergenza Cardiochirurgica",
    "Rete Perinatale", "SPDC", "Non classificata",
]
NETWORK_BY_CODE = dict(enumerate(NETWORK_ORDER, start=1))

AMBULANCES_ORDER = [1, 2, 3, 4]            # ambulances per emergency

# ---- Call generation slots: (name, offset, duration) in minutes -------------
GENERATION_SLOTS = (
    ("00_07", 0, 420),
    ("07_12", 420, 300),
    ("12_18", 720, 360),
    ("18_24", 1080, 360),
)

# ---- Time bands of the travel-time matrices ---------------------------------
SLOT_PREFIXES = [
    "weekday_peak",
    "weekday_normal",
    "weekday_night",
    "weekend_august_normal",
    "weekend_august_night",
]

# ---- Sanitation time: random.triangular(min, max, mode) in minutes ----------
SANITATION_TRIANGULAR = (0.05, 60.0, 38.57)

# ---- Default station capacities (as-is scenario, province RI) ---------------
# Keys may be station names or grid-cell codes (see station_code).
CAPACITIES_AS_IS = {
    "AMATRICE": 1,            # Q30_19
    "BORGO S.PIETRO": 1,      # Q23_7
    "LEONESSA": 1,            # Q18_18
    "MAGLIANO": 1,            # Q3_10
    "OSTERIA NUOVA": 1,       # Q14_5
    "PAGANICO": 1,            # Q20_4
    "PASSO CORESE": 1,        # Q8_3
    "POGGIO MIRTETO": 1,      # Q9_7
    "POSTA": 1,               # Q25_17
    "RIETI": 2,               # Q17_12
    "STIMIGLIANO SCALO": 1,   # Q5_8
    "TORRI IN SABINA": 1,     # Q8_10
}



# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Fixes the seed of the generators used by the model."""
    np.random.seed(seed)
    random.seed(seed)


_call_counter = 0
_log_counter = 0


def reset_call_counter() -> None:
    global _call_counter
    _call_counter = 0


def next_priority(colour_code: int) -> int:
    """
    Combined priority for the ambulance queue:
    the colour code first (1 = most urgent), then an increasing counter that
    makes the order deterministic within the same code.
    """
    global _call_counter
    _call_counter += 1
    return colour_code * PRIORITY_BASE + _call_counter


def get_sim_datetime(env_now: float, start_datetime: datetime = SIM_START_DATETIME) -> datetime:
    """Converts the simulation time (minutes) into a date and time."""
    return start_datetime + timedelta(minutes=env_now)


def log(msg, env_time=None, logfile: str = LOG_FILE) -> None:
    """
    Writes a log line (only if DEBUG is on and below LOG_LIMIT lines).
    On hot paths the call has to be guarded by `if DEBUG:` to avoid the cost of
    formatting the string.
    """
    global _log_counter
    if not DEBUG or _log_counter >= LOG_LIMIT:
        return

    if env_time is not None:
        instant = get_sim_datetime(env_time).strftime("%Y-%m-%d %H:%M:%S")
        line = f"{instant} - {msg}"
    else:
        line = str(msg)

    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    _log_counter += 1


def should_count_call(start_time: float) -> bool:
    """The statistics only consider the calls started after the warm-up."""
    return WARMUP_MINUTES <= start_time < SIM_END_MINUTES


def is_emergency(colour_code: int) -> bool:
    """True for codes R/G (1, 2)."""
    return colour_code in EMERGENCY_CODES


def classify_time_slot(when: datetime):
    """Time band (weekday/weekend, peak/normal/night) of an instant."""
    weekday = when.weekday()     # 0 = Monday ... 6 = Sunday
    hour = when.hour
    is_weekday = weekday < 5 and when.month != 8

    if is_weekday:
        if (6 <= hour < 7) or (9 <= hour < 17) or (19 <= hour < 23):
            return "weekday_normal"
        if hour >= 23 or hour < 6:
            return "weekday_night"
        return "weekday_peak"

    if 6 <= hour < 23:
        return "weekend_august_normal"
    return "weekend_august_night"


def confidence_interval(mean, std, n, level: float = 0.95):
    """Student t confidence interval on the mean over the replications."""
    if n is None or n < 2 or not np.isfinite(std):
        return np.nan, np.nan
    t_value = stats.t.ppf(0.5 + level / 2, df=n - 1)
    half_width = t_value * std / np.sqrt(n)
    return mean - half_width, mean + half_width


def risk_averse_response_time(values, alpha: float = 0.05):
    """CVaR: mean of the worst `alpha` of the response times."""
    values = np.array([v for v in values if pd.notna(v)], dtype=float)
    if len(values) == 0:
        return np.nan
    k = max(1, int(np.ceil(alpha * len(values))))
    return float(np.mean(np.sort(values)[-k:]))


# =============================================================================
# 3. EMPIRICAL DISTRIBUTIONS AND PRE-COMPUTED TIMES
# =============================================================================

class EcdfSampler:
    """
    Inverse-transform sampling of an ECDF. The interpolator is built once:
    `draw(size)` returns a block of values.
    """

    __slots__ = ("values", "probabilities", "_inverse_cdf")

    def __init__(self, values, probabilities):
        self.values = np.asarray(values, dtype=float)
        self.probabilities = np.asarray(probabilities, dtype=float)

        cdf = np.cumsum(self.probabilities)
        self._build_inverse(cdf / cdf[-1])

    @classmethod
    def from_cdf(cls, values, cdf):
        """Builds the sampler from an already cumulated CDF (ECDF file)."""
        sampler = cls.__new__(cls)
        sampler.values = np.asarray(values, dtype=float)
        cdf = np.asarray(cdf, dtype=float)
        sampler.probabilities = np.diff(np.concatenate([[0.0], cdf]))
        sampler._build_inverse(cdf)
        return sampler

    def _build_inverse(self, cdf):
        # Out of range the values are clipped to the observed extremes.
        self._inverse_cdf = interp1d(
            cdf, self.values, kind="linear", bounds_error=False,
            fill_value=(self.values[0], self.values[-1]),
        )

    def draw_one(self):
        """One value sampled by inversion (a single random number)."""
        return float(self._inverse_cdf(np.random.uniform(0, 1)))


class ConstantSampler:
    """Degenerate sampler: used when the data are not enough to fit the ECDF."""

    __slots__ = ("value",)

    def __init__(self, value=0.0):
        self.value = float(value)

    def draw_one(self):
        return self.value          # no random number consumed


def make_sampler(values, probabilities):
    """EcdfSampler if the data are enough, otherwise a constant sampler at 0."""
    if values is None or len(values) < 2 or len(probabilities) != len(values):
        return ConstantSampler(0.0)
    return EcdfSampler(values, probabilities)


def load_ecdf_sampler(csv_path):
    """Interarrival sampler from an ECDF file (columns x, cdf)."""
    df_ecdf = pd.read_csv(DATA_DIR / csv_path)
    return EcdfSampler.from_cdf(df_ecdf["x"].values, df_ecdf["cdf"].values)


def load_interarrival_samplers(ecdf_index_path="interarrivals/index.csv"):
    """Zone → time band → interarrival sampler."""
    samplers = {}
    ecdf_index = pd.read_csv(DATA_DIR / ecdf_index_path)
    for _, row in ecdf_index.iterrows():
        samplers.setdefault(row["zone"], {})[row["time_slot"]] = load_ecdf_sampler(row["ecdf_file"])
    return samplers


# Phases of the mission present in data/parameters/duration_distributions.csv
PHASES_BY_URGENCY = (
    "triage",             # call → end of triage
    "assignment",         # end of triage → dispatch
    "departure",          # dispatch → departure
    "operating_to_hosp",  # arrival on scene → loading
    "operating",          # mission closed on scene
    "delivery_to_ED",     # handover at the emergency department
)
FLAT_PHASES = ("cancelled_after_dispatch", "cancelled_after_departure",
               "cancelled_after_arrival")
BLOCK_PHASE = "ed_block"
POOLED_BLOCK_GROUP = "ALL_HOSPITALS"     # same name as in preprocessing/estimation.py


@dataclass(frozen=True)
class DelaySamplers:
    """Samplers (built once) of every time of the model."""

    by_urgency: dict             # name → {'RG': sampler, 'VB': sampler}
    flat: dict                   # name → sampler
    block: dict                  # hospital → sampler


def load_duration_distributions(path=None):
    """
    Empirical time distributions read from duration_distributions.csv
    (columns phase, group, minutes, probability).
    """
    durations = pd.read_csv(path if path is not None
                            else PARAMS_DIR / "duration_distributions.csv")
    durations["group"] = durations["group"].fillna("")
    return durations


def build_delay_samplers(durations):
    """Builds the time samplers from the empirical distributions."""
    by_urgency, flat, block = {}, {}, {}

    for (phase, group), rows in durations.groupby(["phase", "group"], sort=False):
        sampler = make_sampler(rows["minutes"].to_numpy(), rows["probability"].to_numpy())
        if phase == BLOCK_PHASE:
            block[group] = sampler           # group = hospital
        elif phase in PHASES_BY_URGENCY:
            by_urgency.setdefault(phase, {})[group] = sampler
        else:
            flat[phase] = sampler

    return DelaySamplers(by_urgency=by_urgency, flat=flat, block=block)


class ReplicationTimes:
    """
    Stochastic times of the model. The samplers, and with them the ECDF
    interpolators, are built once; every value is drawn at the moment it is
    needed.
    """

    def __init__(self, samplers: DelaySamplers, interarrival_samplers):
        self._by_urgency = samplers.by_urgency
        self._flat = samplers.flat
        self._block = samplers.block
        self._interarrival = interarrival_samplers

    # ---- times conditioned on the urgency -----------------------------------
    # triage, assignment, departure: urgency at triage; operating,
    # operating_to_hosp, delivery_to_ED: urgency assigned on scene
    def _by_urgency_draw(self, name, colour_code):
        return self._by_urgency[name][CRIT_GROUP_BY_CODE[colour_code]].draw_one()

    def triage(self, colour_code):
        return self._by_urgency_draw("triage", colour_code)

    def assignment(self, colour_code):
        return self._by_urgency_draw("assignment", colour_code)

    def departure(self, colour_code):
        return self._by_urgency_draw("departure", colour_code)

    def operating(self, colour_code):
        return self._by_urgency_draw("operating", colour_code)

    def operating_to_hosp(self, colour_code):
        return self._by_urgency_draw("operating_to_hosp", colour_code)

    def delivery_to_ED(self, colour_code):
        return self._by_urgency_draw("delivery_to_ED", colour_code)

    # ---- unconditioned times ------------------------------------------------
    def cancelled_after_dispatch(self):
        return self._flat["cancelled_after_dispatch"].draw_one()

    def cancelled_after_departure(self):
        return self._flat["cancelled_after_departure"].draw_one()

    def cancelled_after_arrival(self):
        return self._flat["cancelled_after_arrival"].draw_one()

    def sanitation(self):
        """Triangular drawn from the `random` module, not from numpy (both are
        seeded by set_global_seed)."""
        return random.triangular(*SANITATION_TRIANGULAR)

    def block(self, hospital):
        """Hospitals with too few blocks use the distribution pooled over every hospital."""
        sampler = self._block.get(hospital, self._block.get(POOLED_BLOCK_GROUP))
        return sampler.draw_one() if sampler is not None else 0.0

    def interarrival(self, zone, slot):
        return self._interarrival[zone][slot].draw_one()


# =============================================================================
# 4. LOADING OF THE PARAMETERS AND OF THE TRAVEL-TIME MATRICES
# =============================================================================

def extract_slot_and_sheet(filename: str):
    """
    Extracts (time band, matrix type) from the name of the parquet file.
    Example: "weekday_peak_base_target_NON_EMER.parquet".
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    for prefix in SLOT_PREFIXES:
        if base.startswith(prefix):
            return prefix, base.replace(prefix + "_", "")
    return base, "unknown"


def df_to_lookup(DF: pl.DataFrame) -> dict:
    """Turns a matrix (origin, destination, time) into a dictionary."""
    col1, col2, col3 = DF.columns[:3]
    return {
        (row[0], row[1]): float(row[2])
        for row in DF.select([col1, col2, col3]).iter_rows()
    }


def load_travel_time_lookups(parquet_dir="travel-times", slots=None):
    """Time band → matrix type → lookup (origin, destination) → minutes."""
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR / parquet_dir, "*.parquet")))

    per_slot = {}
    for file in parquet_files:
        slot, sheet_name = extract_slot_and_sheet(file)
        if slots is not None and slot not in slots:
            continue
        per_slot.setdefault(slot, {})[sheet_name] = df_to_lookup(pl.read_parquet(file))
    return per_slot


def merge_lookups_by_urgency(travel_lookups):
    """
    A single dictionary per urgency instead of the five matrices looked up in
    sequence by `get_travel_time`. The matrices are inserted in reverse order of
    priority, so that overwriting reproduces "the first one that answers wins":
    the result is identical, with one lookup instead of five.
    """
    merged_by_urgency = {}
    for emer in (True, False):
        suffix = "EMER" if emer else "NON_EMER"
        merged = {}
        for name in ("target_base", f"hospital_target_{suffix}", f"target_target_{suffix}",
                     f"target_hospital_{suffix}", f"base_target_{suffix}"):
            merged.update(travel_lookups[name])
        merged_by_urgency[emer] = merged
    return merged_by_urgency


_MERGED_LOOKUPS = {}


def merged_lookups(travel_lookups):
    """Merged version of the matrices, built once per session."""
    key = id(travel_lookups)
    if key not in _MERGED_LOOKUPS:
        # the original reference is kept too, so that the id stays valid
        _MERGED_LOOKUPS[key] = (travel_lookups, merge_lookups_by_urgency(travel_lookups))
    return _MERGED_LOOKUPS[key][1]


# ---- Estimated parameters (preprocessing/estimate_parameters.py) ------------
with open(PARAMS_DIR / "model_parameters.json", encoding="utf-8") as _f:
    PARAMETERS = json.load(_f)

assert PARAMETERS["network_order"] == NETWORK_ORDER, "inconsistent network order"
assert PARAMETERS["criticality_order"] == URGENCY_ORDER, "inconsistent colour code order"
assert PARAMETERS["ambulances_order"] == AMBULANCES_ORDER, "inconsistent ambulance order"

PROVINCE = PARAMETERS["province"]
VEHICLE_TYPE = PARAMETERS["vehicle_type"]
ZONES = PARAMETERS["zones"]
HOSPITALS = PARAMETERS["hospitals"]

colour_code_probabilities = PARAMETERS["colour_code_probabilities"]
# on-scene colour code (1..5) -> network distribution / % of transport to hospital
network_probabilities = {CODE_BY_COLOUR[c]: p
                         for c, p in PARAMETERS["network_probabilities_by_on_scene"].items()}
transport_pct_by_on_scene = {CODE_BY_COLOUR[c]: pct
                             for c, pct in PARAMETERS["transport_to_hospital_pct_by_on_scene"].items()}
hospitals_by_network = PARAMETERS["hospitals_by_network"]
ambulances_per_emergency = PARAMETERS["ambulances_per_emergency"]
ed_block_pct_by_hospital = PARAMETERS["ed_block_pct_by_hospital"]
sanitation_pct = PARAMETERS["sanitation_pct"]
cancel_after_dispatch_pct = PARAMETERS["mission_cancelled_pct"]["after_dispatch"]
cancel_after_departure_pct = PARAMETERS["mission_cancelled_pct"]["after_departure"]
cancel_after_arrival_pct = PARAMETERS["mission_cancelled_pct"]["after_arrival"]
triage_transition_matrix = PARAMETERS["triage_transition_matrix"]
cell_probabilities_by_zone = PARAMETERS["cell_probabilities_by_zone"]

# ---- Stations: grid-cell code, name, coordinates ---------------------------
unique_stations = pd.read_csv(SPATIAL_DIR / "stations.csv")
STATIONS = unique_stations["code"].tolist()

# The internal identifier of a station is the grid cell ("Q17_12"), because that
# is the key the travel-time matrices are indexed by. Names are labels: they are
# used in the logs, in the reports and to define the scenarios. Candidate cells
# that do not host an actual station have no name and stay identified by their
# code.
STATION_NAME = {
    code: name
    for code, name in zip(unique_stations["code"], unique_stations["name"].fillna(""))
    if name
}
_CODE_BY_NAME = {name.upper(): code for code, name in STATION_NAME.items()}


def station_name(code):
    """Name of the station; the grid-cell code if it has no name."""
    return STATION_NAME.get(code, code)


def station_code(station):
    """Internal code of a station, accepting its name just as well."""
    if station in STATIONS:
        return station

    code = _CODE_BY_NAME.get(str(station).strip().upper())
    if code is None:
        raise ValueError(
            f"Station '{station}' not recognised. "
            f"Valid names: {sorted(STATION_NAME.values())}; "
            f"valid codes: {sorted(STATIONS)}"
        )
    return code


def normalise_capacities(capacities):
    """Translates the keys of a capacity dictionary into the internal codes."""
    if capacities is None:
        return None
    return {station_code(station): value
            for station, value in capacities.items()}


# ---- Time distributions and travel-time matrices ---------------------------
duration_distributions = load_duration_distributions()
delay_samplers = build_delay_samplers(duration_distributions)
interarrival_distributions = load_interarrival_samplers()
# Matrices of every time band: each trip uses the band of the instant it starts.
travel_time_lookups = load_travel_time_lookups()


def default_capacities():
    """As-is capacities: 0 for every station but the active ones."""
    capacities = {name: 0 for name in STATIONS}
    capacities.update(normalise_capacities(CAPACITIES_AS_IS))
    return capacities


# =============================================================================
# 5. STATISTICS COLLECTED BY THE SIMULATION
# =============================================================================

@dataclass
class SimStats:
    """
    Statistics of one replication. They are updated only for the calls started
    after the warm-up (see `should_count_call`).
    """

    response_times: list = field(default_factory=list)
    response_times_EMER: list = field(default_factory=list)
    response_times_NONEMER: list = field(default_factory=list)

    # threshold (minutes) → response times under the threshold
    under_target_EMER: dict = field(default_factory=lambda: {t: [] for t in THRESHOLDS})
    under_target_NONEMER: dict = field(default_factory=lambda: {t: [] for t in THRESHOLDS})

    ambulances_used: dict = field(default_factory=lambda: {p: 0 for p in STATIONS})
    ambulance_service_times: dict = field(default_factory=lambda: {p: [] for p in STATIONS})
    calls_by_zone: dict = field(default_factory=lambda: {z: 0 for z in ZONES})
    hospitals_simulated: dict = field(default_factory=lambda: {h: 0 for h in HOSPITALS})

    counts_colour_code: dict = field(default_factory=lambda: {c: 0 for c in (1, 2, 3, 4)})
    counts_colour_code_on_scene: dict = field(default_factory=lambda: {c: 0 for c in (1, 2, 3, 4, 5)})
    arrival_counts: defaultdict = field(default_factory=lambda: defaultdict(int))

    calls_responded: int = 0
    calls_cancelled_before: int = 0
    calls_responded_or_cancelled: int = 0

    def reset(self):
        """Clears in place (the references held by the model stay valid)."""
        self.__init__()

    def response_times_by_group(self):
        return {
            "ALL": self.response_times,
            "EMER": self.response_times_EMER,
            "NONEMER": self.response_times_NONEMER,
        }

    def under_target_by_group(self):
        return {"EMER": self.under_target_EMER, "NONEMER": self.under_target_NONEMER}

    def record_response(self, response_time, colour_code, station):
        """Records a response time and the coverages under the thresholds."""
        self.response_times.append(response_time)
        self.ambulances_used[station] = self.ambulances_used.get(station, 0) + 1

        if is_emergency(colour_code):
            self.response_times_EMER.append(response_time)
            under_target = self.under_target_EMER
        else:
            self.response_times_NONEMER.append(response_time)
            under_target = self.under_target_NONEMER

        for threshold in THRESHOLDS:
            if response_time <= threshold:
                under_target[threshold].append(response_time)

    def record_service_time(self, station, service_time):
        self.ambulance_service_times.setdefault(station, []).append(service_time)

    def record_cancelled(self):
        self.calls_cancelled_before += 1
        self.calls_responded_or_cancelled += 1

    def record_responded(self):
        self.calls_responded += 1
        self.calls_responded_or_cancelled += 1


sim_stats = SimStats()


# =============================================================================
# 6. SIMULATION MODEL
# =============================================================================

class Emergency(NamedTuple):
    """A request for help (one row = one ambulance requested)."""

    zone: str
    start_time: float
    colour_code: int
    num_ambulances: int
    cell: str
    network: int
    goes_to_hospital: bool
    colour_code_on_scene: int
    id: str = ""

    @property
    def is_emergency(self) -> bool:
        return is_emergency(self.colour_code)


class AmbulanceSystem:
    """Ambulance fleet, assignment to the calls and mission cycle."""

    def __init__(self, env, travel_lookups, possible_hospitals, stats, times,
                 capacities_day=None, capacities_night=None):
        self.env = env
        self.stats = stats
        self.times = times
        self.possible_hospitals = possible_hospitals
        self.hospitals = {
            hospital for hospitals in possible_hospitals.values() for hospital in hospitals
        }

        # Travel-time matrices of every time band (band → matrices); the band in
        # use is the one of the current instant, see `travel()`.
        self._bands = {slot: self._prepare_band(lookups)
                       for slot, lookups in travel_lookups.items()}
        self._band_hour = None
        self._band = None

        # Capacities may come indexed by name: bring them back to the codes
        capacities_day = normalise_capacities(capacities_day)
        capacities_night = normalise_capacities(capacities_night)
        self.capacities_day = capacities_day if capacities_day is not None else default_capacities()
        self.capacities_night = capacities_night if capacities_night is not None else default_capacities()

        # The capacity of the resource is the maximum between the day and the
        # night shift; `can_assign` then limits the assignments to the capacity
        # of the current hour.
        max_capacities = {}
        for name in STATIONS:
            capacity = max(self.capacities_day.get(name, 0), self.capacities_night.get(name, 0))
            if capacity > 0:
                max_capacities[name] = capacity

        self.ambulances = {
            name: simpy.PriorityResource(env, capacity=capacity)
            for name, capacity in max_capacities.items()
        }
        # One state per ambulance: the ambulances of the same station move
        # independently. `busy` covers the whole mission, return trip included,
        # so the busy units of a station are always as many as the resource count.
        self.ambulance_states = {
            name: [{"location": name, "busy": False} for _ in range(resource.capacity)]
            for name, resource in self.ambulances.items()
        }
        self.waiting_events = {name: [] for name in self.ambulances}

        log(f"[INIT] {len(self.ambulances)} active stations, "
            f"{sum(max_capacities.values())} ambulances.")

    # ---- capacities and shifts ----------------------------------------------
    def is_daytime(self):
        hour = int(self.env.now // 60) % 24
        return DAY_START_HOUR <= hour < DAY_END_HOUR

    def get_current_capacity(self, name):
        capacities = self.capacities_day if self.is_daytime() else self.capacities_night
        return capacities.get(name, 0)

    def can_assign(self, name, capacities=None):
        """
        True if the station still has ambulances available in the current shift.
        `capacities` avoids recomputing the shift for every station.
        """
        if capacities is None:
            capacities = self.capacities_day if self.is_daytime() else self.capacities_night
        return self.ambulances[name].count < capacities.get(name, 0)

    # ---- travel times -------------------------------------------------------
    @staticmethod
    def _prepare_band(lookups):
        """Matrices of one time band, indexed by trip type and urgency
        (True = codes R/G, False = codes V/B)."""
        return {
            "base_target": {True: lookups["base_target_EMER"],
                            False: lookups["base_target_NON_EMER"]},
            "hospital_target": {True: lookups["hospital_target_EMER"],
                                False: lookups["hospital_target_NON_EMER"]},
            "target_target": {True: lookups["target_target_EMER"],
                              False: lookups["target_target_NON_EMER"]},
            "target_hospital": {True: lookups["target_hospital_EMER"],
                                False: lookups["target_hospital_NON_EMER"]},
            "target_base": lookups["target_base"],
            "hospital_base": lookups["hospital_base"],
            # merged version of the five matrices with variability, by urgency
            "merged": merged_lookups(lookups),
            # dynamic limit per cell: it depends on the band and on the active stations
            "limit_by_cell": {},
        }

    def travel(self):
        """Matrices of the time band of the current instant (bands change on the hour)."""
        hour = int(self.env.now // 60)
        if hour != self._band_hour:
            self._band_hour = hour
            self._band = self._bands[classify_time_slot(get_sim_datetime(hour * 60))]
        return self._band

    def travel_time_with_jitter(self, value):
        """Travel time with triangular variability ±TRAVEL_JITTER_DELTA, clipped at 0."""
        return max(0.0, np.random.triangular(value - TRAVEL_JITTER_DELTA, value,
                                             value + TRAVEL_JITTER_DELTA))

    def lookup_towards_target(self, from_location, emer):
        """Matrix (and origin type) to use for a trip towards a target."""
        band = self.travel()
        if from_location in self.ambulances:
            return band["base_target"][emer], "station"
        if from_location in self.hospitals:
            return band["hospital_target"][emer], "hospital"
        return band["target_target"][emer], "target"

    def get_travel_time(self, from_location, to_cell, call):
        """
        Travel time towards a target cell, with the urgency of the call: the
        pair is looked up in the merged matrices. None if it does not appear.
        """
        value = self.travel()["merged"][call.is_emergency].get((from_location, to_cell))
        return self.travel_time_with_jitter(value) if value is not None else None

    def return_travel_time(self, from_location, station):
        """
        Return trip to the station: no lights and sirens, so it does not depend
        on the urgency of the call. These legs have no calibration data and use
        the routing times (factor 1). From a hospital both endpoints are exact
        points, so there is no variability.
        """
        band = self.travel()
        if from_location in self.hospitals:
            return band["hospital_base"][(from_location, station)]
        return self.travel_time_with_jitter(band["target_base"][(from_location, station)])

    # ---- waiting queues -----------------------------------------------------
    def check_waiting_events_for_station(self, station):
        """True if there is at least one waiting call (at any station)."""
        # a call waits on several stations: its event stays in the other lists
        # after being woken up, so only the events not yet triggered count
        return any(not event.triggered
                   for waiting in self.waiting_events.values() for _, event in waiting)

    def notify_waiting(self, station, env):
        """
        Releases the calls waiting for the station that has just freed up, most
        urgent first (then in order of arrival): the processes resume in the
        order the events are triggered, so the first to retry the assignment is
        the most urgent call.
        """
        notified = 0
        for _, event in sorted(self.waiting_events[station], key=lambda item: item[0]):
            if not event.triggered:
                event.succeed()
                notified += 1
        self.waiting_events[station].clear()

        if notified and DEBUG:
            log(f"[{env.now}] 🔔 Notified {notified} waiting events "
                f"for {station_name(station)}")

    # ---- releasing the ambulance -------------------------------------------
    def release_and_notify(self, station, unit, req, call, current_location,
                           service_start_time=None, sanitation_needed=False):
        """Return to the station (possibly after sanitation) and release."""
        if sanitation_needed:
            yield self.env.process(self.sanitation(self.env, call))
        # drawn when the trip starts, so it uses the band of that instant
        yield self.env.timeout(self.return_travel_time(current_location, station))

        self.ambulances[station].release(req)
        unit["location"] = station
        unit["busy"] = False
        if service_start_time is not None and should_count_call(service_start_time):
            self.stats.record_service_time(station, self.env.now - service_start_time)
        self.notify_waiting(station, self.env)

    def release_and_notify_no_return(self, call, station, unit, req, current_location,
                                     service_start_time=None):
        """
        Immediate release on the spot: the ambulance stays available for the
        queued calls and goes back to the station only if it is not reassigned.
        """
        self.ambulances[station].release(req)
        unit["busy"] = False
        if service_start_time is not None and should_count_call(service_start_time):
            self.stats.record_service_time(station, self.env.now - service_start_time)
        self.notify_waiting(station, self.env)

        def delayed_return():
            yield self.env.timeout(0.0000000001)

            # reassigned in the meantime: it is now busy, or it has already moved
            if unit["busy"] or unit["location"] != current_location:
                return

            travel_time = self.return_travel_time(current_location, station)
            unit["busy"] = True
            with self.ambulances[station].request() as back_req:
                yield back_req
                yield self.env.timeout(travel_time)
                unit["location"] = station
                unit["busy"] = False
                self.ambulances[station].release(back_req)
                if DEBUG:
                    log(f"[{self.env.now}] 🚑 Ambulance of {station_name(station)} "
                        f"back at the station after {call}")
                self.notify_waiting(station, self.env)

        self.env.process(delayed_return())

    def _release(self, call, station, unit, req, current_location, service_start_time,
                 check_sanitation):
        """
        Closes the mission: sanitation (if drawn) with return to the station,
        otherwise return only if there are no waiting calls.
        """
        # sanitation_pct is the probability that sanitation is NOT needed
        if check_sanitation and np.random.uniform(0, 100) < (100 - sanitation_pct):
            if DEBUG:
                log(f"[{self.env.now}] 🧼 Sanitation for {call}")
            yield from self.release_and_notify(
                station, unit, req, call, current_location, service_start_time,
                sanitation_needed=True,
            )
            return

        if not self.check_waiting_events_for_station(station):
            yield from self.release_and_notify(
                station, unit, req, call, current_location, service_start_time
            )
        else:
            self.release_and_notify_no_return(
                call, station, unit, req, current_location, service_start_time
            )

    def _release_cancelled(self, call, station, unit, req, location, service_start_time):
        """
        Mission cancelled before the arrival on scene: the ambulance is back
        where it started from and is free again there: at the station, or on
        the spot (from where it goes back to the station if it is not reassigned).
        """
        if location != station:
            self.release_and_notify_no_return(call, station, unit, req, location,
                                              service_start_time)
            return

        self.ambulances[station].release(req)
        unit["busy"] = False
        if should_count_call(service_start_time):
            self.stats.record_service_time(station, self.env.now - service_start_time)
        self.notify_waiting(station, self.env)

    # ---- assignment ---------------------------------------------------------
    def compute_dynamic_limit(self, cell, call):
        """
        Maximum travel time accepted for the assignment: fixed for codes V/B,
        equal to the worst of the "nearby" stations for codes R/G.
        """
        if not call.is_emergency:
            return NON_EMERGENCY_LIMIT

        band = self.travel()
        limit = band["limit_by_cell"].get(cell)
        if limit is not None:
            return limit

        lookup = band["base_target"][True]
        distances = []
        for station in self.ambulances:
            travel_time = lookup.get((station, cell))
            if travel_time is not None:
                distances.append(travel_time)
        distances.sort()

        under_threshold = [d for d in distances if d <= UNDER_THRESHOLD_LIMIT]
        if len(under_threshold) >= MIN_STATIONS_CONSIDERED:
            limit = max(under_threshold)
        else:
            limit = max(distances[:MIN_STATIONS_CONSIDERED], default=0.0)

        limit = max(limit, 0.0)
        band["limit_by_cell"][cell] = limit
        return limit

    def assign_ambulance(self, cell, call):
        """
        Closest free ambulance within the dynamic limit.
        Returns (travel_time, station, unit, current location) or None.
        """
        limit = self.compute_dynamic_limit(cell, call)
        emer = call.is_emergency
        capacities = self.capacities_day if self.is_daytime() else self.capacities_night

        candidates = []
        for name, units in self.ambulance_states.items():
            if not self.can_assign(name, capacities):
                continue

            # free ambulances in the same place are equivalent: one candidate each
            seen = set()
            for unit in units:
                location = unit["location"]
                if unit["busy"] or location in seen:
                    continue
                seen.add(location)

                lookup, origin_type = self.lookup_towards_target(location, emer)
                travel_time = lookup.get((location, cell), math.inf)
                if travel_time <= limit:
                    candidates.append(
                        (self.travel_time_with_jitter(travel_time), name, unit, location,
                         origin_type)
                    )

        if not candidates:
            return None

        travel_time, name, unit, location, origin_type = min(candidates, key=lambda c: c[0])
        if DEBUG:
            log(f"Ambulance '{station_name(name)}' leaves from {origin_type} "
                f"'{station_name(location)}' with travel time {travel_time}")
        return travel_time, name, unit, location

    def _wait_for_ambulance(self, cell, call, priority):
        """
        Retries the assignment every time a nearby ambulance frees up; among the
        waiting calls the most urgent retries first (`priority`, see next_priority).
        """
        assigned = self.assign_ambulance(cell, call)
        attempts = 0

        while assigned is None:
            if DEBUG:
                log(f"[{self.env.now}] 🚨 No ambulance for {cell} "
                    f"({call}), waiting...")

            wait_event = self.env.event()
            for station, units in self.ambulance_states.items():
                for unit in units:
                    travel_time = self.get_travel_time(unit["location"], cell, call)
                    if travel_time is not None and travel_time <= MAX_WAIT_TRAVEL_TIME:
                        self.waiting_events[station].append((priority, wait_event))
                        break
            yield wait_event

            assigned = self.assign_ambulance(cell, call)
            attempts += 1

        if attempts and DEBUG:
            log(f"[{self.env.now}] Assigned {call} after {attempts} attempts")
        return assigned

    # ---- phases of the mission ----------------------------------------------
    # Every phase is a process of its own: the time is drawn when the process
    # starts, not when it is created.
    def triage(self, env, call, colour_code):
        yield env.timeout(self.times.triage(colour_code))

    def assignment(self, env, call, colour_code):
        yield env.timeout(self.times.assignment(colour_code))

    def departure(self, env, call, colour_code):
        yield env.timeout(self.times.departure(colour_code))

    def operating_time(self, env, call, colour_code):
        yield env.timeout(self.times.operating(colour_code))

    def operating_time_to_hosp(self, env, call, colour_code):
        yield env.timeout(self.times.operating_to_hosp(colour_code))

    def delivery_of_the_patient_to_ED(self, env, call, colour_code):
        yield env.timeout(self.times.delivery_to_ED(colour_code))

    def block(self, env, call, hospital):
        yield env.timeout(self.times.block(hospital))

    def cancelled_after_dispatch(self, env, call):
        yield env.timeout(self.times.cancelled_after_dispatch())

    def cancelled_after_arrival(self, env, call):
        yield env.timeout(self.times.cancelled_after_arrival())

    def sanitation(self, env, call):
        yield env.timeout(self.times.sanitation())

    def request_ambulance(self, env, call, start_time, priority, cell):
        """SimPy process: from the triage to the end of the mission."""
        countable = should_count_call(start_time)
        colour_code = call.colour_code

        yield env.process(self.triage(env, call, colour_code))

        # The ambulance is chosen and taken right after the triage: it stays
        # busy during the assignment and the preparation of the crew.
        _, chosen_station, unit, location = yield from self._wait_for_ambulance(
            cell, call, priority
        )

        unit["busy"] = True
        req = self.ambulances[chosen_station].request(priority=priority)
        yield req
        service_start_time = env.now

        yield env.process(self.assignment(env, call, colour_code))

        if np.random.uniform(0, 100) < cancel_after_dispatch_pct:
            if DEBUG:
                log(f"[{env.now}] ❌ Mission cancelled after dispatch ({call})")
            yield env.process(self.cancelled_after_dispatch(env, call))
            self._release_cancelled(call, chosen_station, unit, req, location,
                                    service_start_time)
            if countable:
                self.stats.record_cancelled()
            return

        yield env.process(self.departure(env, call, colour_code))

        # Travel time drawn at departure, with the matrices of the current band
        lookup, _ = self.lookup_towards_target(location, call.is_emergency)
        travel_time = self.travel_time_with_jitter(lookup[(location, cell)])

        if np.random.uniform(0, 100) < cancel_after_departure_pct:
            if DEBUG:
                log(f"[{env.now}] ❌ Mission cancelled after departure ({call})")
            # The closure marks the cancellation, while the ambulance is still on
            # its way (median 6.7 min against 11.9 of the trip to the scene):
            # it then drives back the stretch already covered.
            cancel_time = self.times.cancelled_after_departure()
            yield env.timeout(cancel_time)
            yield env.timeout(min(cancel_time, travel_time))
            self._release_cancelled(call, chosen_station, unit, req, location,
                                    service_start_time)
            if countable:
                self.stats.record_cancelled()
            return

        if countable:
            self.stats.record_responded()

        yield env.timeout(travel_time)
        unit["location"] = cell
        if countable:
            response_time = env.now - start_time
            self.stats.record_response(response_time, colour_code, chosen_station)
            if DEBUG:
                log(f"[{env.now}] 🚑 {station_name(chosen_station)} responded in "
                    f"{response_time:.2f} min (from {station_name(location)}) "
                    f"for {call}")

        # --- cancellation on arrival at the target ---
        if np.random.uniform(0, 100) < cancel_after_arrival_pct:
            if DEBUG:
                log(f"[{env.now}] ❌ Mission cancelled on arrival ({call})")
            yield env.process(self.cancelled_after_arrival(env, call))
            yield from self._release(
                call, chosen_station, unit, req, cell, service_start_time,
                check_sanitation=False,
            )
            return

        # --- mission closed on the target ---
        if not call.goes_to_hospital:
            yield env.process(self.operating_time(env, call, call.colour_code_on_scene))
            yield from self._release(
                call, chosen_station, unit, req, cell, service_start_time,
                check_sanitation=True,
            )
            return

        # --- transport to hospital ---
        yield env.process(self.operating_time_to_hosp(env, call, call.colour_code_on_scene))

        if call.colour_code_on_scene == DECEASED_ON_SCENE_CODE:
            yield from self._release(
                call, chosen_station, unit, req, cell, service_start_time,
                check_sanitation=False,
            )
            return

        network_hospitals = self.possible_hospitals[NETWORK_BY_CODE[call.network]]
        lookup = self.travel()["target_hospital"][is_emergency(call.colour_code_on_scene)]
        hospital, travel_time_hosp = min(
            ((hosp, lookup.get((cell, hosp), math.inf)) for hosp in network_hospitals),
            key=lambda pair: pair[1],
        )

        yield env.timeout(self.travel_time_with_jitter(travel_time_hosp))
        unit["location"] = hospital
        if countable:
            self.stats.hospitals_simulated[hospital] = (
                self.stats.hospitals_simulated.get(hospital, 0) + 1
            )

        # --- block at the emergency department ---
        if np.random.uniform(0, 100) < ed_block_pct_by_hospital.get(hospital, 0.0):
            if DEBUG:
                log(f"[{env.now}] ⏳ Ambulance blocked at {hospital} ({call})")
            yield env.process(self.block(env, call, hospital))
        else:
            yield env.process(self.delivery_of_the_patient_to_ED(env, call, call.colour_code_on_scene))

        yield from self._release(
            call, chosen_station, unit, req, hospital, service_start_time,
            check_sanitation=True,
        )


class CallGenerator:
    """Generator of the calls of one zone."""

    def __init__(self, name, ambulance_system, stats, times, colour_probabilities,
                 num_ambulances_probabilities, cell_values, cell_probabilities,
                 network_probabilities):
        self.name = name
        self.ambulance_system = ambulance_system
        self.stats = stats
        self.times = times
        self.colour_probabilities = colour_probabilities
        self.num_ambulances_probabilities = num_ambulances_probabilities
        self.cell_values = cell_values
        self.cell_probabilities = cell_probabilities
        self.network_probabilities = network_probabilities

    def generate_calls(self, env):
        """
        SimPy process: for every day and every time band it draws the
        interarrivals of the band. If an interarrival overruns the band the next
        one is started without generating the call. Calls are generated during
        the warm-up too, so that the system reaches a loaded state; their
        statistics are discarded (see `should_count_call` and `run_replication`).
        """
        for day in range(WARMUP_DAYS + SIM_DAYS):
            day_start = day * MINUTES_PER_DAY

            for slot, offset, duration in GENERATION_SLOTS:
                slot_start = day_start + offset
                slot_end = slot_start + duration

                if env.now < slot_start:
                    yield env.timeout(slot_start - env.now)

                while env.now < slot_end:
                    interarrival = self.times.interarrival(self.name, slot)
                    if env.now + interarrival >= slot_end:
                        break

                    yield env.timeout(interarrival)
                    self.stats.arrival_counts[(self.name, slot)] += 1
                    self._emit_call(env)

    def _emit_call(self, env):
        """Draws the attributes of the emergency and starts a mission per ambulance."""
        colour_code = int(np.random.choice([1, 2, 3, 4], p=self.colour_probabilities))
        num_ambulances = int(np.random.choice(AMBULANCES_ORDER, p=self.num_ambulances_probabilities))
        cell = np.random.choice(self.cell_values, p=self.cell_probabilities)
        start_time = env.now
        colour_code_on_scene = self._draw_on_scene_urgency(colour_code)
        # transport and clinical network depend on the severity assessed on scene
        network = int(np.random.choice(np.arange(1, len(NETWORK_ORDER) + 1),
                                       p=self.network_probabilities[colour_code_on_scene]))
        goes_to_hospital = bool(
            np.random.uniform(0, 100) < transport_pct_by_on_scene[colour_code_on_scene]
        )

        self.stats.counts_colour_code[colour_code] += num_ambulances
        self.stats.counts_colour_code_on_scene[colour_code_on_scene] += num_ambulances
        self.stats.calls_by_zone[self.name] += num_ambulances

        base = Emergency(
            zone=self.name,
            start_time=start_time,
            colour_code=colour_code,
            num_ambulances=num_ambulances,
            cell=cell,
            network=network,
            goes_to_hospital=goes_to_hospital,
            colour_code_on_scene=colour_code_on_scene,
        )
        if DEBUG:
            log(f"[{env.now}] 📞 New call: {base}")

        for i_amb in range(num_ambulances):
            call = base._replace(
                num_ambulances=1,
                id=f"{self.name}_{env.now:.2f}_AMB_{i_amb + 1}",
            )
            env.process(
                self.ambulance_system.request_ambulance(
                    env, call, start_time,
                    priority=next_priority(colour_code),
                    cell=cell,
                )
            )

    def _draw_on_scene_urgency(self, colour_code):
        """On-scene urgency, conditioned on the one of the telephone triage."""
        probabilities = triage_transition_matrix[COLOUR_BY_CODE[colour_code]]
        colour_on_scene = np.random.choice(
            list(probabilities.keys()), p=list(probabilities.values())
        )
        return CODE_BY_COLOUR[colour_on_scene]


# =============================================================================
# 7. RUNNING THE REPLICATIONS
# =============================================================================

def run_replication(capacities_day, capacities_night, interarrival_samplers=None,
                    seed=None):
    """
    Runs one replication (warm-up + simulated year) and leaves the statistics in
    `sim_stats`. Returns the capacities actually used (day, night).
    """
    if seed is not None:
        set_global_seed(seed)

    reset_call_counter()
    sim_stats.reset()

    # Every time of the replication is drawn here, right after the seed.
    times = ReplicationTimes(
        delay_samplers,
        interarrival_samplers if interarrival_samplers is not None else interarrival_distributions,
    )

    env = simpy.Environment()

    amb_system = AmbulanceSystem(
        env, travel_time_lookups, hospitals_by_network, sim_stats, times, capacities_day, capacities_night
    )

    for zone in ZONES:
        generator = CallGenerator(
            name=zone,
            ambulance_system=amb_system,
            stats=sim_stats,
            times=times,
            colour_probabilities=colour_code_probabilities,
            num_ambulances_probabilities=ambulances_per_emergency,
            cell_values=cell_probabilities_by_zone[zone]["values"],
            cell_probabilities=cell_probabilities_by_zone[zone]["probabilities"],
            network_probabilities=network_probabilities,
        )
        env.process(generator.generate_calls(env))

    env.run(until=WARMUP_MINUTES)
    sim_stats.reset()                  # discard the transient
    env.run(until=SIM_END_MINUTES)

    return amb_system.capacities_day, amb_system.capacities_night


# =============================================================================
# 8. KPIs AND REPORTS
# =============================================================================

WORKING_MINUTES_PER_SHIFT = (DAY_END_HOUR - DAY_START_HOUR) * 60 * SIM_DAYS


def compute_vehicle_utilization(ambulance_service_times, capacities_day, capacities_night):
    """Station → utilisation (%) = service time / opening time."""
    utilization = {}
    for station, service_times in ambulance_service_times.items():
        working_time = (
            capacities_day.get(station, 0) * WORKING_MINUTES_PER_SHIFT
            + capacities_night.get(station, 0) * WORKING_MINUTES_PER_SHIFT
        )
        utilization[station] = (
            sum(service_times) / working_time * 100 if working_time > 0 else 0
        )
    return utilization


class ReplicationKPIs:
    """Collects the KPIs of every replication, reading them from `SimStats`."""

    def __init__(self):
        self._num_repl = 0
        self.mean_response_time = {g: [] for g in GROUPS}
        self.std_response_time = []
        self.max_response_time = []
        self.cvar = {(alpha, g): [] for alpha in CVAR_ALPHAS for g in GROUPS}
        self.coverage = {(g, t): [] for g in ("EMER", "NONEMER") for t in THRESHOLDS}

        self.ambulances_used = []
        self.hospitals_simulated = []
        self.calls_by_zone = []
        self.counts_colour_code = []
        self.counts_colour_code_on_scene = []

        self.calls_generated = []
        self.calls_responded = []
        self.calls_cancelled = []
        self.calls_resp_or_cancel = []
        self.calls_diff = []

        self.vehicle_utilization_rows = []
        self.first_replication_response_times = []

    @property
    def num_repl(self):
        return self._num_repl

    def add(self, replication_index, stats, capacities_day, capacities_night):
        """Extracts the KPIs of the replication that has just finished."""
        self._num_repl += 1
        per_group = stats.response_times_by_group()
        under_target = stats.under_target_by_group()

        for group, values in per_group.items():
            self.mean_response_time[group].append(np.mean(values) if len(values) else np.nan)
            for alpha in CVAR_ALPHAS:
                self.cvar[(alpha, group)].append(risk_averse_response_time(values, alpha=alpha))

        self.std_response_time.append(np.std(per_group["ALL"]) if per_group["ALL"] else np.nan)
        self.max_response_time.append(max(per_group["ALL"]) if per_group["ALL"] else np.nan)

        for group in ("EMER", "NONEMER"):
            total = len(per_group[group])
            for threshold in THRESHOLDS:
                self.coverage[(group, threshold)].append(
                    len(under_target[group][threshold]) / total * 100 if total else 0
                )

        self.ambulances_used.append(dict(stats.ambulances_used))
        self.hospitals_simulated.append({
            k: v for k, v in stats.hospitals_simulated.items()
            if k is not None and str(k).strip() != ""
        })
        self.calls_by_zone.append(dict(stats.calls_by_zone))
        self.counts_colour_code.append(dict(stats.counts_colour_code))
        self.counts_colour_code_on_scene.append(dict(stats.counts_colour_code_on_scene))

        calls_generated = sum(stats.calls_by_zone.values())
        self.calls_generated.append(calls_generated)
        self.calls_responded.append(stats.calls_responded)
        self.calls_cancelled.append(stats.calls_cancelled_before)
        self.calls_resp_or_cancel.append(stats.calls_responded_or_cancelled)
        self.calls_diff.append(stats.calls_responded_or_cancelled - calls_generated)

        utilization = compute_vehicle_utilization(
            stats.ambulance_service_times, capacities_day, capacities_night
        )
        for station, util_rate in utilization.items():
            cap_day = capacities_day.get(station, 0)
            cap_night = capacities_night.get(station, 0)
            service_times = stats.ambulance_service_times.get(station, [])
            self.vehicle_utilization_rows.append({
                "Replication": replication_index + 1,
                "Station": station_name(station),
                "Code": station,
                "Cap_day": cap_day,
                "Cap_night": cap_night,
                "Total_service_time_min": sum(service_times),
                "Working_time_min": (cap_day + cap_night) * WORKING_MINUTES_PER_SHIFT,
                "Num_services": len(service_times),
                "Vehicle_utilization_rate_pct": util_rate,
            })

        if replication_index == 0:
            self.first_replication_response_times = [
                rt for rt in per_group["ALL"] if pd.notna(rt)
            ]

    # ---- aggregations -------------------------------------------------------
    def mean_std(self, values):
        """Mean and sample standard deviation over the replications."""
        series = pd.Series(values, dtype=float).dropna()
        if series.empty:
            return np.nan, np.nan
        return float(series.mean()), float(series.std(ddof=1)) if len(series) > 1 else 0.0

    def summary_row(self, values, metric_name):
        """Summary row (mean, std, min, max, CI) of a per-replication KPI."""
        series = pd.Series(values, dtype=float).dropna()
        if series.empty:
            return {"Metric": metric_name, "Mean": np.nan, "Std": np.nan,
                    "Min": np.nan, "Max": np.nan, "CI_low": np.nan, "CI_high": np.nan}

        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        ci_low, ci_high = confidence_interval(mean, std, self.num_repl)
        return {"Metric": metric_name, "Mean": mean, "Std": std,
                "Min": float(series.min()), "Max": float(series.max()),
                "CI_low": ci_low, "CI_high": ci_high}

    def mean_by_key(self, list_of_dicts, keys):
        """Mean and std over the replications of a key → count dictionary."""
        mean = {k: float(np.mean([d.get(k, 0) for d in list_of_dicts])) for k in keys}
        std = {
            k: float(np.std([d.get(k, 0) for d in list_of_dicts], ddof=1))
            if self.num_repl > 1 else 0.0
            for k in keys
        }
        return mean, std

    def cvar_label(self, alpha, group):
        return f"Worst {int(round(alpha * 100))}% response time {group}"

    def extra_kpi_frame(self):
        """Table of the response KPIs per replication."""
        columns = {"Replication": np.arange(1, self.num_repl + 1)}
        for group in GROUPS:
            columns[f"Mean_response_time_{group}"] = self.mean_response_time[group]
        for alpha in CVAR_ALPHAS:
            for group in GROUPS:
                columns[f"Worst_{int(round(alpha * 100))}pct_response_time_{group}"] = (
                    self.cvar[(alpha, group)]
                )
        columns["Max_response_time"] = self.max_response_time
        return pd.DataFrame(columns)

    def extra_kpi_summary_frame(self):
        """Summary (mean, CI) of the response KPIs over the replications."""
        rows = [
            self.summary_row(self.mean_response_time[g], f"Mean response time {g}")
            for g in GROUPS
        ]
        for alpha in CVAR_ALPHAS:
            for group in GROUPS:
                rows.append(
                    self.summary_row(self.cvar[(alpha, group)], self.cvar_label(alpha, group))
                )
        rows.append(self.summary_row(self.max_response_time, "Max response time"))
        return pd.DataFrame(rows)

    def coverage_frame(self):
        """Coverage per replication, by group and threshold."""
        columns = {"Replication": np.arange(1, self.num_repl + 1)}
        for group, label in (("NONEMER", "NON_EMER"), ("EMER", "EMER")):
            for threshold in THRESHOLDS:
                columns[f"Coverage_{label}_lt{threshold}"] = self.coverage[(group, threshold)]
        return pd.DataFrame(columns)

    def calls_frame(self):
        return pd.DataFrame({
            "Replication": np.arange(1, self.num_repl + 1),
            "Generated": self.calls_generated,
            "Responded": self.calls_responded,
            "Cancelled_before": self.calls_cancelled,
            "Responded+Cancelled_before": self.calls_resp_or_cancel,
            "Diff": self.calls_diff,
        })

    def vehicle_utilization_frames(self):
        df_long = pd.DataFrame(self.vehicle_utilization_rows)
        df_summary = (
            df_long.groupby("Station")
            .agg(
                Mean_utilization_pct=("Vehicle_utilization_rate_pct", "mean"),
                Std_utilization_pct=("Vehicle_utilization_rate_pct", "std"),
                Min_utilization_pct=("Vehicle_utilization_rate_pct", "min"),
                Max_utilization_pct=("Vehicle_utilization_rate_pct", "max"),
                Mean_num_services=("Num_services", "mean"),
                Mean_total_service_time_min=("Total_service_time_min", "mean"),
                Mean_working_time_min=("Working_time_min", "mean"),
            )
            .reset_index()
        )
        return df_long, df_summary


def format_duration(seconds):
    """Duration in readable form: '45.2 s' or '2m 13s'."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m {rest:02d}s"


def default_workers():
    """
    Number of processes to use when it is not given explicitly: one core is left
    free and WORKERS_MAX is never exceeded. The environment variable EMS_WORKERS
    takes precedence (EMS_WORKERS=1 turns the parallel run off).
    """
    from_environment = os.environ.get("EMS_WORKERS")
    if from_environment:
        return max(1, int(from_environment))
    return max(1, min(WORKERS_MAX, (os.cpu_count() or 1) - 1))


def parallelism_available():
    """
    Spawning new processes requires a `__main__` the child can recreate: with
    `python -c "..."` or in an interactive session there is none, so the run
    stays serial instead of risking an error.
    """
    main = sys.modules.get("__main__")
    return bool(getattr(main, "__file__", None) or getattr(main, "__spec__", None))


def _replication_in_process(arguments):
    """Runs one replication in a separate process and returns its statistics."""
    capacities_day, capacities_night, seed = arguments
    cap_day_eff, cap_night_eff = run_replication(
        capacities_day, capacities_night, None, seed
    )
    return sim_stats, cap_day_eff, cap_night_eff


def run_replications(num_repl, capacities_day, capacities_night, seed=BASE_SEED,
                     label=None, on_replication=None, workers=None):
    """
    Runs `num_repl` replications and returns the KPIs collected.

    With `workers` greater than 1 the replications are spread over several
    processes: each one has its own seed and is deterministic, so the results do
    not depend on the number of processes.
    """
    kpi = ReplicationKPIs()
    workers = default_workers() if workers is None else max(1, workers)
    prefix = f"[{label}] " if label else ""

    if workers > 1 and num_repl > 1 and not parallelism_available():
        print(f"{prefix}Serial run: the parallel one requires launching a script "
              f"(python model.py ...), not 'python -c'.")
        workers = 1

    def record(i, stats, cap_day_eff, cap_night_eff):
        kpi.add(i, stats, cap_day_eff, cap_night_eff)
        if on_replication is not None:
            on_replication(i, stats)
        print(f"{prefix}Replication {i + 1}/{num_repl} - "
              f"Mean response time: {kpi.mean_response_time['ALL'][-1]:.2f} min")

    start = time.perf_counter()

    if workers > 1 and num_repl > 1:
        print(f"{prefix}{num_repl} replications on {min(workers, num_repl)} processes")
        arguments = [(capacities_day, capacities_night, seed + i) for i in range(num_repl)]
        with ProcessPoolExecutor(max_workers=min(workers, num_repl)) as pool:
            for i, result in enumerate(pool.map(_replication_in_process, arguments)):
                record(i, *result)
    else:
        for i in range(num_repl):
            cap_day_eff, cap_night_eff = run_replication(
                capacities_day, capacities_night, interarrival_distributions, seed + i
            )
            record(i, sim_stats, cap_day_eff, cap_night_eff)

    duration = time.perf_counter() - start
    print(f"{prefix}{num_repl} replications completed in {format_duration(duration)} "
          f"({duration / num_repl:.1f} s per replication)")
    return kpi


def run_simulation(num_repl=30, capacities_day=None, capacities_night=None,
                   seed=BASE_SEED, workers=None):
    """Quick run: returns the summary KPIs used in optimisation."""
    kpi = run_replications(num_repl, capacities_day, capacities_night, seed=seed,
                           workers=workers)

    results = {
        "coverage_emer_20": float(np.nanmean(kpi.coverage[("EMER", 20)])),
        "coverage_nonemer_20": float(np.nanmean(kpi.coverage[("NONEMER", 20)])),
        "mean_rt_emer": float(np.nanmean(kpi.mean_response_time["EMER"])),
        "cvar_emer_5": float(np.nanmean(kpi.cvar[(0.05, "EMER")])),
        "cvar_all_5": float(np.nanmean(kpi.cvar[(0.05, "ALL")])),
    }

    print("\nFinal results over the replications:")
    print(f"Coverage EMER <20 min:     {results['coverage_emer_20']:.2f}%")
    print(f"Mean response time EMER:   {results['mean_rt_emer']:.2f} min")
    print(f"CVaR EMER, worst 5%:       {results['cvar_emer_5']:.2f} min")
    return results


def run_and_export_kpi(num_repl=30, capacities_day=None, capacities_night=None,
                       output_dir="report_scenario", label="scenario", workers=None):
    """
    Runs `num_repl` replications and saves an Excel file with every simulated
    KPI (without any comparison with the real data).
    """
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f"report_{label}.xlsx")

    kpi = run_replications(num_repl, capacities_day, capacities_night,
                           seed=BASE_SEED, label=label, workers=workers)
    n = kpi.num_repl

    # --- Summary: mean response time and coverage ---
    metrics = [("Mean response time ALL", kpi.mean_response_time["ALL"])]
    metrics += [(f"Coverage NON EMER <{t}", kpi.coverage[("NONEMER", t)]) for t in THRESHOLDS]
    metrics += [(f"Coverage EMER <{t}", kpi.coverage[("EMER", t)]) for t in THRESHOLDS]

    summary_rows = []
    for name, values in metrics:
        mean, std = kpi.mean_std(values)
        ci_low, ci_high = confidence_interval(mean, std, n)
        summary_rows.append({"Metric": name, "Mean": mean,
                             "CI_low": ci_low, "CI_high": ci_high})
    df_summary = pd.DataFrame(summary_rows)

    # --- Hospitals, zones, stations ---
    mean_hospitals, std_hospitals = kpi.mean_by_key(
        kpi.hospitals_simulated,
        sorted({h for d in kpi.hospitals_simulated for h in d}, key=str),
    )
    total_hospitals = sum(mean_hospitals.values())
    hospital_rows = []
    for hospital, mean in mean_hospitals.items():
        ci_low, ci_high = confidence_interval(mean, std_hospitals[hospital], n)
        hospital_rows.append({
            "Hospital": hospital,
            "Sim_mean_abs": mean,
            "Sim_CI_abs_low": ci_low,
            "Sim_CI_abs_high": ci_high,
            "Sim_mean_pct": mean / total_hospitals * 100 if total_hospitals > 0 else 0,
        })
    df_hospitals = pd.DataFrame(hospital_rows).sort_values("Sim_mean_abs", ascending=False)

    mean_zones, std_zones = kpi.mean_by_key(kpi.calls_by_zone, sorted(ZONES))
    zone_rows = []
    for zone, mean in mean_zones.items():
        ci_low, ci_high = confidence_interval(mean, std_zones[zone], n)
        zone_rows.append({"Zone": zone, "Sim_mean_abs": mean,
                          "Sim_CI_abs_low": ci_low, "Sim_CI_abs_high": ci_high})
    df_zones = pd.DataFrame(zone_rows).sort_values("Sim_mean_abs", ascending=False)

    mean_stations, std_stations = kpi.mean_by_key(kpi.ambulances_used, sorted(STATIONS))
    df_stations = pd.DataFrame([
        {"Station": station_name(p), "Code": p,
         "Mean_ambulances": mean_stations[p], "Std_ambulances": std_stations[p]}
        for p in mean_stations
    ]).sort_values("Mean_ambulances", ascending=False)

    df_vehicle_utilization, df_vehicle_utilization_summary = kpi.vehicle_utilization_frames()

    df_max_target_summary = pd.DataFrame({
        "Mean": [np.nanmean(kpi.max_response_time)],
        "Standard deviation": [np.nanstd(kpi.max_response_time, ddof=1) if n > 1 else np.nan],
    })

    sheets = {
        "Summary": df_summary,
        "Coverage_per_replication": kpi.coverage_frame(),
        "Extra_KPI": kpi.extra_kpi_frame(),
        "Extra_KPI_summary": kpi.extra_kpi_summary_frame(),
        "Calls_per_replication": kpi.calls_frame(),
        "MaxResponseTime_raw": pd.DataFrame({
            "Replication": np.arange(1, n + 1),
            "Max_response_time": kpi.max_response_time,
        }),
        "MaxResponseTime_stats": df_max_target_summary,
        "Hospitals": df_hospitals,
        "Zones": df_zones,
        "Stations": df_stations,
        "Vehicle_Utilization": df_vehicle_utilization,
        "Vehicle_Util_summary": df_vehicle_utilization_summary,
    }

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[{label}] Excel report saved in: {excel_path}")
    return excel_path


def arrival_diagnostics(arrival_counts, df_unique, sim_days=SIM_DAYS):
    """
    Compares the simulated arrivals with the real ones by (zone, time slot).
    `arrival_counts` is `sim_stats.arrival_counts` after a replication.
    """
    real_counts = (
        df_unique.groupby(["zone", "time_slot"]).size().reset_index(name="real_count")
    )
    # The real data cover 365 days: they are scaled to sim_days if different
    real_counts["real_count_scaled"] = real_counts["real_count"] * (sim_days / 365)

    rows = []
    for (zone, slot), sim_count in arrival_counts.items():
        real_row = real_counts[
            (real_counts["zone"] == zone) & (real_counts["time_slot"] == slot)
        ]
        real = float(real_row["real_count_scaled"].values[0]) if len(real_row) else np.nan
        error = (sim_count - real) / real * 100 if real > 0 else np.nan
        rows.append({
            "Zone": zone, "TimeSlot": slot, "Sim": sim_count,
            "Real": round(real, 1), "Error_pct": round(error, 2),
        })

    df_diag = pd.DataFrame(rows).sort_values("Error_pct")

    print("\n" + "═" * 65)
    print("ARRIVAL DIAGNOSTICS: simulated vs real")
    print("═" * 65)
    print(df_diag.to_string(index=False))
    print(f"\nMean error:   {df_diag['Error_pct'].mean():.2f}%")
    print(f"Median error: {df_diag['Error_pct'].median():.2f}%")
    print("Error by time slot:\n"
          f"{df_diag.groupby('TimeSlot')['Error_pct'].mean().round(2).to_string()}")
    print("Error by zone:\n"
          f"{df_diag.groupby('Zone')['Error_pct'].mean().round(2).to_string()}")

    return df_diag


# =============================================================================
# 9. CALIBRATION
# =============================================================================

RI_STATIONS = [
    "AMATRICE", "BORGO S.PIETRO", "LEONESSA", "MAGLIANO", "OSTERIA NUOVA",
    "PAGANICO", "PASSO CORESE", "POGGIO MIRTETO", "POSTA", "RIETI",
    "STIMIGLIANO", "STIMIGLIANO SCALO", "TORRI IN SABINA",
]

COLOUR_CODE_GROUPS = {"1_2": [1, 2], "3_4": [3, 4]}


def slot_from_hour(hour):
    """Call-generation slot corresponding to an hour of the day."""
    for slot, offset, duration in GENERATION_SLOTS:
        if offset // 60 <= hour < (offset + duration) // 60:
            return slot
    return GENERATION_SLOTS[-1][0]


def load_real_data(year=2023, path=None):
    """
    Anonymised emergency instances of the given year
    (data/anonymized-instances/emergencies.csv).
    """
    df_real = pd.read_csv(path if path is not None
                          else INSTANCES_DIR / "emergencies.csv")
    df_real["date"] = pd.to_datetime(df_real["date"], errors="coerce")

    df_real = df_real[
        (df_real["date"].dt.year == year) & (df_real["vehicle_type"] == VEHICLE_TYPE)
    ].copy()
    return df_real


def prepare_real_arrivals(df_real):
    """Unique emergencies with the time band of the call (for the diagnostics)."""
    df_unique = df_real.drop_duplicates(subset="emergency_id").copy()
    return df_unique[df_unique["date"].notna()]


def real_coverage(response_times):
    """Threshold → % of response times under the threshold."""
    times = pd.Series(response_times, dtype=float).dropna()
    if times.empty:
        return {t: np.nan for t in THRESHOLDS}
    return {t: float((times <= t).sum()) / len(times) * 100 for t in THRESHOLDS}


# ---- Tables and figures of the paper ---------------------------------------
# arXiv:2604.24394v2 — "A Realistic Discrete Event Simulation model for Ambulance
# Location and Deployment within a regional Emergency Medical Service".
# The calibration produces Tables 1-7 (one Excel sheet each) and Figures 6-8
# (one image file each): nothing else.

COL_MU = "mu"                    # historical value
COL_AVG = "AVGsim"               # mean over the replications
COL_LB, COL_UB = "LBsim", "UBsim"
COL_DLB, COL_DUB = "|LBsim-mu|", "|UBsim-mu|"

COLOUR_SIM = "#c0392b"           # simulated: red, as in the paper
COLOUR_REAL = "#2471a3"          # historical: blue

CORRECTION_FACTOR_FILES = (
    "travel_time_correction_factors.xlsx",
    "travel_time_correction_factors.csv",
    "scaling_factors.xlsx",
)
CORRECTION_SEGMENTS = {
    "base_target_EMER": ("Urgent", "Base-scene"),
    "base_target_NON_EMER": ("Non-urgent", "Base-scene"),
    "target_hospital_EMER": ("Urgent", "Scene-hospital"),
    "target_hospital_NON_EMER": ("Non-urgent", "Scene-hospital"),
}


# Decimal digits with which the paper presents each table.
# "*" is the value used for every numeric column of the sheet, the other entries
# are the exceptions: counts and thresholds are integers.
TABLE_FORMATS = {
    "T1 - Potential EDs": {"*": 0},
    "T2 - Calls by zone and slot": {"*": 0},
    "T3 - Correction factors": {"*": 3},
    "T4 - Calls validation": {"*": 2, COL_MU: 0},
    "T5 - Coverage urgent": {"*": 2, "Threshold (min)": 0},
    "T6 - Coverage non-urgent": {"*": 2, "Threshold (min)": 0},
    "T7 - Calls per base": {"*": 2},
}


def round_table(table, formats):
    """Rounds every numeric column to the digits used in the paper."""
    default = formats.get("*", 2)
    rounded = table.copy()

    for column in rounded.columns:
        if not pd.api.types.is_numeric_dtype(rounded[column]):
            continue
        digits = formats.get(column, default)
        rounded[column] = rounded[column].round(digits)
        if digits == 0 and rounded[column].notna().all():
            rounded[column] = rounded[column].astype("int64")

    return rounded


def write_excel_tables(tables, excel_path):
    """
    Writes one table per sheet, with the decimal digits of the paper: the value
    is rounded and the cell format shows the trailing zeros, so that 62.10 does
    not become 62.1.
    """
    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            cell_formats = {}

            for sheet, table in tables.items():
                formats = TABLE_FORMATS.get(sheet, {})
                default = formats.get("*", 2)
                table = round_table(table, formats)
                table.to_excel(writer, sheet_name=sheet, index=False)

                worksheet = writer.sheets[sheet]
                worksheet.freeze_panes(1, 1)

                for i, column in enumerate(table.columns):
                    width = max(12, min(45, len(str(column)) + 4))
                    if not pd.api.types.is_numeric_dtype(table[column]):
                        worksheet.set_column(i, i, width)
                        continue

                    digits = formats.get(column, default)
                    if digits not in cell_formats:
                        num_format = "0" if digits == 0 else "0." + "0" * digits
                        cell_formats[digits] = workbook.add_format(
                            {"num_format": num_format}
                        )
                    worksheet.set_column(i, i, width, cell_formats[digits])

    except PermissionError as error:
        raise PermissionError(
            f"Cannot write {excel_path}: the file is open in Excel. "
            "Close it and run again (the figures have already been saved)."
        ) from error


def table_1_emergency_departments():
    """Table 1: destination emergency departments reachable from the territory."""
    hospitals = sorted({h for hosps in hospitals_by_network.values() for h in hosps})
    return pd.DataFrame({
        "#": range(1, len(hospitals) + 1),
        "Emergency Department": hospitals,
    })


def table_2_calls_by_zone_and_slot(df_stations):
    """Table 2: number of historical calls by zone and time band."""
    slots = [name for name, _, _ in GENERATION_SLOTS]
    df_slots = df_stations.dropna(subset=["date", "zone"])

    table = (
        pd.pivot_table(df_slots, index="zone", columns="time_slot",
                       values="emergency_id", aggfunc="count", fill_value=0)
        .reindex(columns=slots, fill_value=0)
        .reset_index()
        .rename(columns={"zone": "Zone"})
    )
    table["Total"] = table[slots].sum(axis=1)
    return table.sort_values("Total", ascending=False).reset_index(drop=True)


def table_3_correction_factors():
    """
    Table 3: travel-time correction factors, by segment, urgency and time band.
    None if the file of the factors is not available.
    """
    folders = [globals().get("PARAMS_DIR"), globals().get("MODEL_DIR"), Path(".")]
    for folder in folders:
        if folder is None:
            continue
        for name in CORRECTION_FACTOR_FILES:
            path = Path(folder) / name
            if not path.exists():
                continue
            factors = (pd.read_excel(path) if path.suffix == ".xlsx"
                       else pd.read_csv(path))

            factors["Urgency"] = factors["category"].map(
                lambda c: CORRECTION_SEGMENTS.get(c, (None, None))[0])
            factors["Segment"] = factors["category"].map(
                lambda c: CORRECTION_SEGMENTS.get(c, (None, None))[1])
            factors = factors.dropna(subset=["Urgency", "Segment"])

            table = factors.pivot(index="time_band",
                                  columns=["Urgency", "Segment"],
                                  values="scaling_factor")
            table.columns = [f"{urgency} - {segment}"
                             for urgency, segment in table.columns]
            return table.reset_index().rename(columns={"time_band": "Time interval"})
    return None


def table_4_calls_validation(kpi, real_counts, n):
    """Table 4: number of historical and simulated calls, by urgency class."""
    labels = {"1_2": "Urgent", "3_4": "Non-urgent"}

    rows = []
    for group, codes in COLOUR_CODE_GROUPS.items():
        per_replication = [sum(d.get(c, 0) for c in codes) for d in kpi.counts_colour_code]
        mean, std = kpi.mean_std(per_replication)
        lb, ub = confidence_interval(mean, std, n)
        real = sum(int(real_counts.get(c, 0)) for c in codes)

        rows.append({
            "Urgency": labels[group],
            COL_MU: real,
            COL_AVG: mean,
            COL_LB: lb,
            COL_UB: ub,
            COL_DLB: abs(lb - real),
            COL_DUB: abs(ub - real),
        })
    return pd.DataFrame(rows)


def table_coverage(kpi, real_coverages, group, n):
    """Tables 5 and 6: historical and simulated coverage at every threshold."""
    rows = []
    for threshold in THRESHOLDS:
        mean, std = kpi.mean_std(kpi.coverage[(group, threshold)])
        lb, ub = confidence_interval(mean, std, n)
        real = real_coverages[threshold]

        rows.append({
            "Threshold (min)": threshold,
            COL_MU: real,
            COL_AVG: mean,
            COL_LB: lb,
            COL_UB: ub,
            COL_DLB: abs(lb - real),
            COL_DUB: abs(ub - real),
        })
    return pd.DataFrame(rows)


def table_7_stations(kpi, station_frequency, n):
    """Table 7: percentage of calls assigned at each station."""
    mean_stations, std_stations = kpi.mean_by_key(kpi.ambulances_used, sorted(STATIONS))
    total_sim = sum(mean_stations.values())

    real_values = normalise_capacities(station_frequency.to_dict())
    total_real = sum(real_values.values())

    def as_percentage(value, total):
        return value / total * 100 if total > 0 else 0.0

    rows = []
    for station in sorted(mean_stations):
        real = as_percentage(real_values.get(station, 0), total_real)
        if real == 0:
            continue                      # candidate cells, not actual stations

        lb_abs, ub_abs = confidence_interval(mean_stations[station],
                                             std_stations[station], n)
        mean = as_percentage(mean_stations[station], total_sim)
        lb = as_percentage(lb_abs, total_sim)
        ub = as_percentage(ub_abs, total_sim)

        rows.append({
            "Base": station_name(station),
            COL_MU: real,
            COL_AVG: mean,
            COL_LB: lb,
            COL_UB: ub,
            COL_DLB: abs(lb - real),
            COL_DUB: abs(ub - real),
        })
    return pd.DataFrame(rows).sort_values(COL_MU, ascending=False).reset_index(drop=True)


def figure_coverage(table, title, path):
    """
    Figures 6 and 7: historical and simulated coverage as the threshold varies,
    as paired bars; the simulated bars carry the 95% confidence interval.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(table))
    width = 0.38
    errors = (table[COL_UB] - table[COL_LB]) / 2

    ax.bar(x - width / 2, table[COL_AVG], width, yerr=errors, capsize=5,
           color=COLOUR_SIM, label="Simulated (95% CI)")
    ax.bar(x + width / 2, table[COL_MU], width,
           color=COLOUR_REAL, label="Real")

    ax.set_xticks(x)
    ax.set_xticklabels([f"<{threshold}" for threshold in table["Threshold (min)"]])
    ax.set_xlabel("Response time threshold (min)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_stations(table, path):
    """Figure 8: percentage of calls per station, historical and simulated."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(table))
    width = 0.38
    errors = (table[COL_UB] - table[COL_LB]) / 2

    ax.bar(x - width / 2, table[COL_AVG], width, yerr=errors, capsize=4,
           color=COLOUR_SIM, label="Simulated")
    ax.bar(x + width / 2, table[COL_MU], width,
           color=COLOUR_REAL, label="Real")

    ax.set_xticks(x)
    ax.set_xticklabels(table["Base"], rotation=45, ha="right")
    ax.set_ylabel("Calls assigned (%)")
    ax.set_title("Percentage of calls assigned at each base")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_calibration(num_repl=30, output_dir="report_calibration",
                    figure_format="png", extra=False, workers=None):
    """
    Runs `num_repl` replications and produces the validation tables and figures
    of the paper arXiv:2604.24394v2:

        report_calibration.xlsx    Tables 1-7, one per sheet
        figures/figure_06..08.<fmt> Figures 6, 7 and 8

    With `extra=True` it also saves the arrival diagnostics and the coverage of
    every replication, which do not appear in the paper.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Historical data ----
    df_real = load_real_data()
    df_stations = df_real[df_real["station"].isin(RI_STATIONS)]
    df_valid = df_stations[
        (df_stations["mission_outcome"] != "MISSIONE ANNULLATA") |
        df_stations["arrived_on_scene"]
    ]

    station_frequency = df_valid["station"].value_counts()

    def times_by_urgency(urgencies):
        mask = (
            df_valid["urgency_triage"].isin(urgencies) &
            df_valid["response_time_min"].notna() &
            (df_valid["response_time_min"] >= 0)
        )
        return df_valid.loc[mask, "response_time_min"]

    real_coverages = {
        "EMER": real_coverage(times_by_urgency(["R", "G"])),
        "NONEMER": real_coverage(times_by_urgency(["V", "B"])),
    }
    real_triage_counts = df_stations["urgency_triage"].map(CODE_BY_COLOUR).value_counts()

    # ---- Replications ----
    after_replication = None
    if extra:
        df_unique = prepare_real_arrivals(df_real)

        def after_replication(index, stats):
            arrival_diagnostics(dict(stats.arrival_counts), df_unique, SIM_DAYS).to_csv(
                output_dir / "arrival_diagnostics.csv", index=False
            )

    kpi = run_replications(num_repl, capacities_day=None, capacities_night=None,
                           seed=BASE_SEED, on_replication=after_replication,
                           workers=workers)
    n = kpi.num_repl

    # ---- Tables of the paper ----
    coverage_urgent = table_coverage(kpi, real_coverages["EMER"], "EMER", n)
    coverage_non_urgent = table_coverage(kpi, real_coverages["NONEMER"], "NONEMER", n)
    stations = table_7_stations(kpi, station_frequency, n)

    tables = {
        "T1 - Potential EDs": table_1_emergency_departments(),
        "T2 - Calls by zone and slot": table_2_calls_by_zone_and_slot(df_stations),
        "T3 - Correction factors": table_3_correction_factors(),
        "T4 - Calls validation": table_4_calls_validation(
            kpi, real_triage_counts, n),
        "T5 - Coverage urgent": coverage_urgent,
        "T6 - Coverage non-urgent": coverage_non_urgent,
        "T7 - Calls per base": stations,
    }
    tables = {name: t for name, t in tables.items() if t is not None}

    # ---- Figures of the paper (before the Excel: they are not lost if it is open) ----
    figures = {
        "figure_06_coverage_urgent": lambda p: figure_coverage(
            coverage_urgent, "Coverage for urgent calls: real vs simulated", p),
        "figure_07_coverage_non_urgent": lambda p: figure_coverage(
            coverage_non_urgent, "Coverage for non-urgent calls: real vs simulated", p),
        "figure_08_calls_per_base": lambda p: figure_stations(stations, p),
    }
    for name, draw in figures.items():
        draw(figures_dir / f"{name}.{figure_format}")

    excel_path = output_dir / "report_calibration.xlsx"
    write_excel_tables(tables, excel_path)

    if extra:
        kpi.coverage_frame().to_csv(output_dir / "coverage_per_replication.csv", index=False)

    print(f"\nTables ({len(tables)} sheets) -> {excel_path}")
    print(f"Figures ({len(figures)})          -> {figures_dir}")
    return str(excel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibration: validation tables and figures of the paper."
    )
    parser.add_argument("--replications", type=int, default=30,
                        help="number of replications (default: 30)")
    parser.add_argument("--output-dir", default="report_calibration",
                        help="output folder (default: report_calibration)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="format of the figures (default: png)")
    parser.add_argument("--extra", action="store_true",
                        help="also save the arrival diagnostics and the coverage per replication")
    parser.add_argument("--workers", type=int, default=None,
                        help="replications in parallel (default: 1, or EMS_WORKERS)")
    parser.add_argument("--debug", action="store_true",
                        help="write the event log to log.txt")
    args = parser.parse_args()

    DEBUG = args.debug
    run_calibration(num_repl=args.replications, output_dir=args.output_dir,
                    figure_format=args.format, extra=args.extra,
                    workers=args.workers)
