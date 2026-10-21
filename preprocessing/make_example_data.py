"""
Synthetic example dataset for the simulator, written to `example-data/`.

The real inputs are derived from data of ARES 118 and are not published. This
script builds a fictitious territory with the same file formats, so that the
simulator can be run end to end:

    python preprocessing/make_example_data.py
    python run.py scenarios --example-data

Every value is invented: the grid, the coordinates, the zones, the hospitals,
the travel times, the probabilities, the durations, the call volumes and the
"historical" instances used by the calibration. Only the names of the stations
are the real ones, because the scenarios of the paper refer to them. The results
obtained on these data say nothing about the real system.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "example-data"
SEED = 2026
rng = np.random.default_rng(SEED)

NETWORK_ORDER = [
    "Gen", "Rete Trauma", "Rete Ictus", "Sindrome Coronarica Acuta",
    "Aritmie Gravi e Scompenso Cardiaco", "Rete Emergenza Cardiochirurgica",
    "Rete Perinatale", "SPDC", "Non classificata",
]
BANDS = {  # time band -> multiplier of the nominal travel time
    "weekday_peak": 1.15, "weekday_normal": 1.00, "weekday_night": 0.90,
    "weekend_august_normal": 0.95, "weekend_august_night": 0.90,
}
SLOTS = {"00_07": 420, "07_12": 300, "12_18": 360, "18_24": 360}
SLOT_SHARE = {"00_07": 0.12, "07_12": 0.30, "12_18": 0.33, "18_24": 0.25}

# ---- Fictitious territory: a 10 x 8 grid of 6 x 6 km cells ------------------
COLS, ROWS, CELL_KM = 10, 8, 6.0
cells = pd.DataFrame(
    [(f"Q{c}_{r}", c, r) for c in range(1, COLS + 1) for r in range(1, ROWS + 1)],
    columns=["cell", "col", "row"],
)
cells["x_km"] = (cells["col"] - 0.5) * CELL_KM
cells["y_km"] = (cells["row"] - 0.5) * CELL_KM


def zone_of(col, row):
    if 4 <= col <= 7 and 3 <= row <= 6:
        return "CENTRE"
    if row >= 6:
        return "NORTH"
    if col <= 3:
        return "WEST"
    if col >= 8:
        return "EAST"
    return "SOUTH"


cells["zone"] = [zone_of(c, r) for c, r in zip(cells["col"], cells["row"])]

# Real station names (the scenarios use them) at invented positions
STATIONS = {
    "RIETI": (5, 4), "OSTERIA NUOVA": (6, 2), "POGGIO MIRTETO": (2, 3),
    "TORRI IN SABINA": (3, 5), "PASSO CORESE": (2, 1), "BORGO S.PIETRO": (9, 3),
    "STIMIGLIANO SCALO": (1, 4), "POSTA": (8, 7), "PAGANICO": (8, 1),
    "MAGLIANO": (1, 6), "LEONESSA": (5, 8), "AMATRICE": (10, 8),
}
FIRE_STATIONS = {  # code -> (name, x_km, y_km)
    "PMCASERMA": ("POGGIO MIRTETO FS", 13.6, 18.8),
    "PCASERMA": ("POSTA FS", 42.4, 37.8),
    "RCASERMA": ("RIETI FS", 31.2, 22.4),
}
stations = [(f"Q{c}_{r}", name, (c - 0.5) * CELL_KM, (r - 0.5) * CELL_KM) for name, (c, r) in STATIONS.items()]
stations += [(code, name, x, y) for code, (name, x, y) in FIRE_STATIONS.items()]
stations = pd.DataFrame(stations, columns=["code", "name", "x_km", "y_km"])

HOSPITALS = {  # name -> (x_km, y_km): a central general hospital, two small ones, a far hub
    "HOSPITAL_CENTRAL": (28.0, 22.0),
    "HOSPITAL_WEST": (4.0, 16.0),
    "HOSPITAL_EAST": (56.0, 12.0),
    "HUB_FAR_AWAY": (-30.0, -50.0),
}
hospitals = pd.DataFrame([(h, x, y) for h, (x, y) in HOSPITALS.items()], columns=["hospital", "x_km", "y_km"])
hospitals_by_network = {
    "Gen": ["HOSPITAL_CENTRAL", "HOSPITAL_WEST", "HOSPITAL_EAST"],
    "Rete Trauma": ["HOSPITAL_CENTRAL", "HUB_FAR_AWAY"],
    "Rete Ictus": ["HOSPITAL_CENTRAL", "HUB_FAR_AWAY"],
    "Sindrome Coronarica Acuta": ["HOSPITAL_CENTRAL"],
    "Aritmie Gravi e Scompenso Cardiaco": ["HOSPITAL_CENTRAL", "HOSPITAL_EAST"],
    "Rete Perinatale": ["HOSPITAL_CENTRAL"],
    "SPDC": ["HOSPITAL_CENTRAL"],
    "Non classificata": ["HOSPITAL_CENTRAL"],
}


def to_latlon(x_km, y_km):
    """Fictitious coordinates around (0, 0): they only label the points."""
    return round(y_km / 111.0, 6), round(x_km / 82.0, 6)


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---- spatial/ ----------------------------------------------------------------
lat, lon = zip(*(to_latlon(x, y) for x, y in zip(cells.x_km, cells.y_km)))
write_csv(pd.DataFrame({"cell": cells.cell, "lat": lat, "lon": lon}), OUT / "spatial/cells.csv")
lat, lon = zip(*(to_latlon(x, y) for x, y in zip(stations.x_km, stations.y_km)))
write_csv(pd.DataFrame({"code": stations.code, "name": stations.name, "lat": lat, "lon": lon}),
          OUT / "spatial/stations.csv")
lat, lon = zip(*(to_latlon(x, y) for x, y in zip(hospitals.x_km, hospitals.y_km)))
write_csv(pd.DataFrame({"hospital": hospitals.hospital, "lat": lat, "lon": lon}), OUT / "spatial/hospitals.csv")

# ---- travel-times/ -------------------------------------------------------------
points = {c: (x, y) for c, x, y in zip(cells.cell, cells.x_km, cells.y_km)}
station_points = {c: (x, y) for c, x, y in zip(stations.code, stations.x_km, stations.y_km)}
hospital_points = dict(zip(hospitals.hospital, zip(hospitals.x_km, hospitals.y_km)))


def minutes(a, b, km_per_min, band_factor):
    """Straight-line distance x 1.35 (road detour), plus 1 minute to get going."""
    d = math.dist(a, b) * 1.35
    return 0.0 if d == 0 else round((1.0 + d / km_per_min) * band_factor)


SPEED = {"EMER": 1.2, "NON_EMER": 0.95, "plain": 0.95}  # km per minute


def matrix(origins, destinations, speed, band_factor):
    rows = [(o, d, float(minutes(po, pd_, speed, band_factor)))
            for o, po in origins.items() for d, pd_ in destinations.items()]
    return pd.DataFrame(rows, columns=["origin", "destination", "minutes"])


(OUT / "travel-times").mkdir(parents=True, exist_ok=True)
for band, factor in BANDS.items():
    trips = {
        "target_base": (points, station_points, SPEED["plain"], 1.0),
        "hospital_base": (hospital_points, station_points, SPEED["plain"], 1.0),
    }
    for urgency in ("EMER", "NON_EMER"):
        trips[f"base_target_{urgency}"] = (station_points, points, SPEED[urgency], factor)
        trips[f"target_target_{urgency}"] = (points, points, SPEED[urgency], factor)
        trips[f"target_hospital_{urgency}"] = (points, hospital_points, SPEED[urgency], factor)
        trips[f"hospital_target_{urgency}"] = (hospital_points, points, SPEED[urgency], factor)
    for name, args in trips.items():
        matrix(*args).to_parquet(OUT / "travel-times" / f"{band}_{name}.parquet", index=False)

# ---- interarrivals/ ------------------------------------------------------------
CALLS_PER_YEAR = {"CENTRE": 3200, "NORTH": 900, "WEST": 1300, "EAST": 800, "SOUTH": 1200}
index_rows = []
for zone, per_year in CALLS_PER_YEAR.items():
    for slot, length in SLOTS.items():
        n_calls = int(round(per_year * SLOT_SHARE[slot]))
        mean_gap = 365 * length / n_calls
        x = np.sort(stats.expon(scale=mean_gap).ppf((np.arange(400) + 0.5) / 400))
        path = f"interarrivals/{zone}/ecdf_{slot}.csv"
        write_csv(pd.DataFrame({"x": x, "cdf": np.arange(1, 401) / 400}), OUT / path)
        index_rows.append({"zone": zone, "time_slot": slot, "n_calls": n_calls, "n_days_obs": 365,
                           "n_ecdf_points": 400, "ecdf_file": path})
write_csv(pd.DataFrame(index_rows), OUT / "interarrivals/index.csv")

# ---- parameters/duration_distributions.csv -------------------------------------
def histogram(mean, cv, n_bins=50):
    """Binned gamma distribution (mean, coefficient of variation) in the published format."""
    shape = 1 / cv ** 2
    edges = stats.gamma(shape, scale=mean / shape).ppf(np.linspace(0.001, 0.995, n_bins + 1))
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, np.full(n_bins, 1 / n_bins)


durations = []
phases = {  # phase -> (mean RG, mean VB, coefficient of variation), minutes
    "triage": (1.5, 1.4, 0.6), "assignment": (0.9, 0.8, 0.8), "departure": (1.8, 1.9, 0.6),
    "operating_to_hosp": (21.0, 15.0, 0.5), "operating": (40.0, 30.0, 0.5), "delivery_to_ED": (19.0, 17.0, 0.5),
}
for phase, (mean_rg, mean_vb, cv) in phases.items():
    for group, mean in (("RG", mean_rg), ("VB", mean_vb)):
        for v, p in zip(*histogram(mean, cv)):
            durations.append((phase, group, v, p))
for phase, mean in (("cancelled_after_dispatch", 1.5), ("cancelled_after_departure", 9.0),
                    ("cancelled_after_arrival", 12.0)):
    for v, p in zip(*histogram(mean, 0.7)):
        durations.append((phase, "", v, p))
for group, mean in (("HOSPITAL_CENTRAL", 50.0), ("HUB_FAR_AWAY", 120.0), ("ALL_HOSPITALS", 65.0)):
    for v, p in zip(*histogram(mean, 0.8)):
        durations.append(("ed_block", group, v, p))
write_csv(pd.DataFrame(durations, columns=["phase", "group", "minutes", "probability"]),
          OUT / "parameters/duration_distributions.csv")

# ---- parameters/model_parameters.json -------------------------------------------
cell_probabilities_by_zone = {}
for zone, rows in cells.groupby("zone"):
    w = rng.gamma(1.5, size=len(rows))
    cell_probabilities_by_zone[zone] = {"values": rows.cell.tolist(), "probabilities": (w / w.sum()).tolist()}

def network_distribution(severe_share):
    p = np.array([0.60, 0.20, 0.04, 0.04, 0.05, 0.0, 0.01, 0.04, 0.02])
    p[[2, 3, 4]] *= severe_share
    return (p / p.sum()).tolist()

parameters = {
    "province": "EXAMPLE",
    "vehicle_type": "ASI",
    "zones": list(CALLS_PER_YEAR),
    "hospitals": sorted(HOSPITALS),
    "criticality_order": ["R", "G", "V", "B"],
    "network_order": NETWORK_ORDER,
    "ambulances_order": [1, 2, 3, 4],
    "colour_code_probabilities": [0.15, 0.65, 0.18, 0.02],
    "network_probabilities_by_on_scene": {
        "R": network_distribution(3.0), "G": network_distribution(1.5), "V": network_distribution(0.3),
        "B": network_distribution(0.1), "N": network_distribution(1.0),
    },
    "hospitals_by_network": hospitals_by_network,
    "ambulances_per_emergency": [0.93, 0.06, 0.008, 0.002],
    "transport_to_hospital_pct_by_on_scene": {"R": 88.0, "G": 80.0, "V": 50.0, "B": 30.0, "N": 0.0},
    "mission_cancelled_pct": {"after_dispatch": 4.0, "after_departure": 3.5, "after_arrival": 1.0},
    "ed_block_pct_by_hospital": {"HOSPITAL_CENTRAL": 4.0, "HOSPITAL_WEST": 2.0, "HOSPITAL_EAST": 2.0,
                                 "HUB_FAR_AWAY": 15.0},
    "sanitation_pct": 88.0,
    "triage_transition_matrix": {
        "R": {"R": 0.35, "G": 0.40, "V": 0.12, "B": 0.03, "N": 0.10},
        "G": {"R": 0.08, "G": 0.66, "V": 0.24, "B": 0.015, "N": 0.005},
        "V": {"R": 0.02, "G": 0.35, "V": 0.59, "B": 0.04, "N": 0.0},
        "B": {"R": 0.0, "G": 0.15, "V": 0.45, "B": 0.40, "N": 0.0},
    },
    "cell_probabilities_by_zone": cell_probabilities_by_zone,
}
(OUT / "parameters").mkdir(parents=True, exist_ok=True)
with open(OUT / "parameters/model_parameters.json", "w", encoding="utf-8") as f:
    json.dump(parameters, f, indent=1)

# ---- anonymized-instances/emergencies.csv (invented "historical" records) -------
n = sum(CALLS_PER_YEAR.values())
station_weights = np.array([8, 2, 2, 1.5, 1.5, 1.5, 1, 1, 1, 0.8, 0.8, 0.5])
records = pd.DataFrame({
    "emergency_id": [f"E{i:05d}" for i in range(1, n + 1)],
    "date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "hour": rng.choice(24, n),
    "zone": rng.choice(list(CALLS_PER_YEAR), n, p=np.array(list(CALLS_PER_YEAR.values())) / n),
    "urgency_triage": rng.choice(["R", "G", "V", "B"], n, p=parameters["colour_code_probabilities"]),
    "station": rng.choice(list(STATIONS), n, p=station_weights / station_weights.sum()),
    "vehicle_type": "ASI",
})
records["time_slot"] = pd.cut(records["hour"], [-1, 6, 11, 17, 23], labels=list(SLOTS)).astype(str)
cancelled = rng.random(n) < 0.08
records["mission_outcome"] = np.where(
    cancelled, "MISSIONE ANNULLATA",
    np.where(rng.random(n) < 0.72, "MISSIONE CONCLUSA CON IL TRASPORTO IN OSPEDALE",
             "MISSIONE CONCLUSA SUL POSTO O CON TRATTAMENTO EFFETTUATO SUL POSTO"))
records["arrived_on_scene"] = ~cancelled
records["response_time_min"] = np.where(cancelled, np.nan, np.round(rng.gamma(4.0, 4.5, n), 2))
records["date"] = records["date"].dt.date
write_csv(records, OUT / "anonymized-instances/emergencies.csv")

(OUT / "README.md").write_text(
    "# Example data (synthetic)\n\n"
    "Fictitious inputs with the same format as the real `data/` folder, written by\n"
    "`preprocessing/make_example_data.py`. Every value is invented: only the names of\n"
    "the stations are real, because the scenarios refer to them. Results obtained on\n"
    "these data say nothing about the real system.\n\n"
    "    python run.py calibration --example-data\n"
    "    python run.py all --example-data\n",
    encoding="utf-8",
)
print(f"Example data written to {OUT}")
