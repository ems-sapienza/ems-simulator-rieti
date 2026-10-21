"""
Estimation of the simulation parameters from the raw mission-level data.

This module is the only place where the raw (non-shareable) dataset is read.
It turns mission records into the aggregate parameters consumed by the
simulator: probabilities, transition matrices and empirical distributions of
the duration of each phase of a mission.

The column names of the raw dataset are the Italian ones of the source system
and are used verbatim; everything this module produces is in English.

The simulator itself never imports this module: it reads the files written by
`estimate_parameters.py`.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

# Ordering conventions shared with the simulator ------------------------------
URGENCY_ORDER = ["R", "G", "V", "B"]
NETWORK_ORDER = [
    "Gen", "Rete Trauma", "Rete Ictus", "Sindrome Coronarica Acuta",
    "Aritmie Gravi e Scompenso Cardiaco", "Rete Emergenza Cardiochirurgica",
    "Rete Perinatale", "SPDC", "Non classificata",
]
AMBULANCES_ORDER = [1, 2, 3, 4]

MACRO_GROUPS = {"RG": ["R", "G"], "VB": ["V", "B"]}
URGENCY_COLUMN = "Criticità Al triage Telefonico"

# The phases after the arrival on scene are conditioned on the urgency assigned
# on scene (paper, Section 3.4.3); "N" (deceased on scene) goes with the urgent codes.
ON_SCENE_URGENCY_COLUMN = "Criticità sul Posto"
ON_SCENE_PHASES = ("operating_to_hosp", "operating", "delivery_to_ED")
ON_SCENE_GROUPS = {"RG": ["R", "G", "N"], "VB": ["V", "B"]}
ON_SCENE_URGENCY_ORDER = ["R", "G", "V", "B", "N"]

TRANSPORT_OUTCOME = "MISSIONE CONCLUSA CON IL TRASPORTO IN OSPEDALE"

# Duration columns: name used by the simulator -> column of the raw dataset
DELAY_COLUMNS_BY_URGENCY = {
    "triage": "diff_FineTriage_Chiamata",          # call -> end of triage
    "assignment": "diff_Invio_FineTriage",         # end of triage -> dispatch
    "departure": "diff_Partenza_Invio",            # dispatch -> departure
    "operating_to_hosp": "diff_Carico_Arrivo",     # arrival on scene -> loading
    "operating": "diff_Chiusura_Arrivo",           # mission closed on scene
    "delivery_to_ED": "diff_Chiusura_Ospedale",    # handover at the ED
}
DELAY_COLUMNS_FLAT = {
    "cancelled_after_dispatch": "diff_Chiusura_MissAnn(invio)",
    "cancelled_after_departure": "diff_Chiusura_MissAnn(partenza)",
    "cancelled_after_arrival": "diff_Chiusura_MissAnn(arrivo)",
}
BLOCK_COLUMN = "diff_Chiusura_Blocco"
# ED blocks are not trimmed with the IQR rule, which would cut the long blocks
# (offload delays of hours): only the implausible ones are dropped.
MAX_ED_BLOCK_MINUTES = 12 * 60
# Block distribution pooled over every hospital, for the hospitals with too few blocks
POOLED_BLOCK_GROUP = "ALL_HOSPITALS"

# Probability (%) that an ambulance does NOT need sanitation, by province
SANITATION_PCT_BY_PROVINCE = {
    "FR": 85.81, "LT": 92.43, "RM": 90.61, "VT": 87.57, "RI": 87.57, None: 0,
}


# =============================================================================
# Probabilities and transition matrices
# =============================================================================

def compute_colour_code_probabilities(df):
    """Province -> triage colour-code distribution (order R, G, V, B)."""
    colour_code = {}
    df_unique = df.drop_duplicates(subset="Numero Emergenza")

    for province in df_unique["Provincia Target"].unique():
        df_prov = df_unique[df_unique["Provincia Target"] == province]
        counts = df_prov[URGENCY_COLUMN].value_counts().to_dict()
        total = len(df_prov)

        if total > 0:
            colour_code[province] = [counts.get(c, 0) / total for c in URGENCY_ORDER]
        else:
            colour_code[province] = [0] * len(URGENCY_ORDER)

    return colour_code


def compute_network_probabilities_by_on_scene(df):
    """
    Province -> on-scene urgency -> hospital-network distribution of the
    patients transported to hospital (order NETWORK_ORDER): the network only
    matters for the choice of the hospital. Urgencies never transported (e.g.
    "N") get the distribution of all the transported patients.
    """
    network_probabilities = {}
    df_unique = df.drop_duplicates(subset="Numero Emergenza")
    df_transported = df_unique[df_unique["ESITO MISSIONE"] == TRANSPORT_OUTCOME]

    for province in df_unique["Provincia Target"].dropna().unique():
        df_prov = df_transported[df_transported["Provincia Target"] == province]

        def distribution(rows):
            counts = rows["rete ospedaliera"].value_counts(normalize=True).to_dict()
            return [counts.get(network, 0) for network in NETWORK_ORDER]

        overall = distribution(df_prov)
        network_probabilities[province] = {
            urgency: distribution(rows) if len(rows) else overall
            for urgency in ON_SCENE_URGENCY_ORDER
            for rows in [df_prov[df_prov[ON_SCENE_URGENCY_COLUMN] == urgency]]
        }

    return network_probabilities


def compute_hospitals_by_network(df):
    """Province -> hospital network -> hospitals that can receive the patient."""
    hospitals_by_network = {}
    df_unique = df.drop_duplicates(subset="Numero Emergenza")

    for province in df_unique["Provincia Target"].dropna().unique():
        hospitals_by_network[province] = {}
        df_prov = df_unique[df_unique["Provincia Target"] == province]

        all_hospitals = (
            df_prov.loc[df_prov["OSPEDALE"] != "ALTRA_DESTINAZIONE", "OSPEDALE"]
            .dropna().unique().tolist()
        )

        networks_present = df_prov["rete ospedaliera"].dropna().unique()
        for network in networks_present:
            mask = (
                (df_prov["rete ospedaliera"] == network) &
                (df_prov["OSPEDALE"] != "ALTRA_DESTINAZIONE")
            )
            hospitals_by_network[province][network] = (
                df_prov.loc[mask, "OSPEDALE"].dropna().unique().tolist()
            )

        # "Non classificata": in Rieti the single integrated hospital, elsewhere any
        if "Non classificata" in networks_present:
            if province == "RI":
                hospitals_by_network[province]["Non classificata"] = [
                    "RIETI_S_CAMILLO_DE_LELLIS_POLO_OSPEDALIERO_UNICO_INTEGRATO"
                ]
            else:
                hospitals_by_network[province]["Non classificata"] = all_hospitals

    return hospitals_by_network


def compute_transport_to_hospital_pct_by_on_scene(df):
    """
    Province -> on-scene urgency -> % of missions ending with transport to a
    hospital (cancelled missions excluded).
    """
    transport_pct = {}
    df_unique = df.drop_duplicates(subset="Numero Emergenza")
    df_filtered = df_unique[df_unique["ESITO MISSIONE"] != "MISSIONE ANNULLATA"]

    for province in df["Provincia Target"].dropna().unique():
        df_prov = df_filtered[df_filtered["Provincia Target"] == province]
        transported = df_prov["ESITO MISSIONE"] == TRANSPORT_OUTCOME
        by_urgency = transported.groupby(df_prov[ON_SCENE_URGENCY_COLUMN]).mean() * 100
        transport_pct[province] = {
            urgency: float(by_urgency.get(urgency, transported.mean() * 100))
            for urgency in ON_SCENE_URGENCY_ORDER
        }

    return transport_pct


def compute_ambulances_per_emergency(df):
    """Province -> distribution of the number of ambulances per emergency (1..4)."""
    df = df.copy()
    df["ambulances_per_emergency"] = (
        df.groupby("Numero Emergenza")["Numero Emergenza"].transform("count")
    )

    ambulances_per_emergency = {}
    for province in df["Provincia Target"].unique():
        df_prov = df[df["Provincia Target"] == province]
        unique_emergencies = df_prov.drop_duplicates(subset=["Numero Emergenza"])
        counts = unique_emergencies["ambulances_per_emergency"].value_counts().to_dict()

        values = [counts.get(n, 0) for n in AMBULANCES_ORDER]
        total = sum(values)
        ambulances_per_emergency[province] = (
            [v / total for v in values] if total > 0 else [0] * len(AMBULANCES_ORDER)
        )

    return ambulances_per_emergency


def compute_mission_cancelled_pct(df):
    """
    Province -> probability (%) that the mission is cancelled after dispatch,
    after departure and after arrival on scene (conditional, in cascade).
    """
    df_cancelled = df[df["ESITO MISSIONE"] == "MISSIONE ANNULLATA"]
    total_rows_by_province = df["Provincia Target"].value_counts().to_dict()

    cancelled = {}
    for province in df["Provincia Target"].unique():
        df_prov = df_cancelled[df_cancelled["Provincia Target"] == province]
        total = total_rows_by_province.get(province, 0)
        occurrences = df_prov["Ultima Colonna Non Vuota"].value_counts()

        occ_dispatch = occurrences.get("DATA_ORA_INVIO", 0)
        occ_departure = occurrences.get("DATA_ORA_PARTENZA", 0)
        occ_arrival = occurrences.get("DATA_ORA_ARRIVO", 0)

        den_dispatch = total
        den_departure = den_dispatch - occ_dispatch
        den_arrival = den_departure - occ_departure

        cancelled[province] = {
            "after_dispatch": (occ_dispatch / den_dispatch * 100) if den_dispatch > 0 else 0,
            "after_departure": (occ_departure / den_departure * 100) if den_departure > 0 else 0,
            "after_arrival": (occ_arrival / den_arrival * 100) if den_arrival > 0 else 0,
        }

    return cancelled


def compute_ed_block_pct_by_hospital(df):
    """Province -> hospital -> % of missions with a block at the ED."""
    totals = df.groupby(["Provincia Target", "OSPEDALE"]).size()
    with_block = (
        df[df["Data e Ora Inizio Blocco"].notna()]
        .groupby(["Provincia Target", "OSPEDALE"])
        .size()
        .reindex(totals.index, fill_value=0)
    )
    percentages = with_block / totals * 100

    block_pct = defaultdict(dict)
    for (province, hospital), value in percentages.items():
        block_pct[province][hospital] = value
    return dict(block_pct)


def compute_transition_matrix(df, col_from, col_to):
    """Row-normalised transition matrix as a nested dictionary."""
    df_clean = df.dropna(subset=[col_from, col_to])
    contingency = pd.crosstab(df_clean[col_from], df_clean[col_to], normalize="index")
    return contingency.to_dict(orient="index")


def compute_cell_probabilities_by_zone(df, col_emergency_id, col_cell, col_zone):
    """Zone -> {'values': [cells], 'probabilities': [weights]}."""
    df_unique = df.drop_duplicates(subset=col_emergency_id)
    df_filtered = df_unique[df_unique[col_cell].notnull()]

    counts = df_filtered.groupby([col_zone, col_cell]).size().unstack(fill_value=0)
    perc = counts.div(counts.sum(axis=1), axis=0)
    perc = perc.loc[:, (perc > 0).any(axis=0)]        # drop never-used cells
    perc = perc.div(perc.sum(axis=1), axis=0)         # renormalise

    return {
        zone: {
            "values": perc.columns[perc.loc[zone].notnull()].tolist(),
            "probabilities": perc.loc[zone].dropna().tolist(),
        }
        for zone in perc.index
    }


def assign_stations_to_grid(df, station_columns, valid_stations, grid):
    """Map each station to the grid cell that contains it."""
    df_station = (
        df[df["POSTAZIONE"].isin(valid_stations)]
        [["POSTAZIONE", station_columns["lat"], station_columns["lon"]]]
        .drop_duplicates(subset=["POSTAZIONE"])
    )

    def find_name(lat, lon):
        match = grid[
            (grid["bottom_right_lat"] <= lat) & (lat <= grid["top_left_lat"]) &
            (grid["top_left_lon"] <= lon) & (lon <= grid["bottom_right_lon"])
        ]
        return match.iloc[0]["name"] if not match.empty else None

    df_station["name"] = df_station.apply(
        lambda row: find_name(row[station_columns["lat"]], row[station_columns["lon"]]),
        axis=1,
    )
    return df_station


# =============================================================================
# Empirical distributions of the phase durations
# =============================================================================

def build_empirical_for_simpy(data, n_points=100, remove_outliers=True, min_samples=10):
    """
    Normalised histogram of a series of durations, ready for inverse-transform
    sampling. Returns (bin_centers, probabilities, mean, std).
    """
    data = data.dropna()
    data = data[data > 0]

    if remove_outliers and len(data) > 0:
        q_low, q_high = data.quantile(0.25), data.quantile(0.75)
        iqr = q_high - q_low
        data = data[(data >= q_low - 1.5 * iqr) & (data <= q_high + 1.5 * iqr)]

    if len(data) < min_samples:
        # Not enough observations: empty but consistent structure
        return np.array([]), np.array([]), np.nan, np.nan

    bins = min(n_points, len(data))
    hist_values, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    probabilities = hist_values.astype(float)
    probabilities /= probabilities.sum()

    return bin_centers, probabilities, data.mean(), data.std()


def build_empirical_distributions(df, n_points=50):
    """
    Empirical distributions of the phase durations:
    - columns conditioned on urgency        -> {'RG': (values, probs), 'VB': ...}
      (triage urgency before the arrival on scene, on-scene urgency after)
    - unconditioned columns                 -> (values, probs)
    - ED block duration                     -> hospital -> (values, probs)
    """
    distributions = {}

    for col in DELAY_COLUMNS_FLAT.values():
        values, probs, _, _ = build_empirical_for_simpy(df[col], n_points=n_points)
        if len(values):
            distributions[col] = (values, probs)

    for phase, col in DELAY_COLUMNS_BY_URGENCY.items():
        if phase in ON_SCENE_PHASES:
            urgency_column, groups = ON_SCENE_URGENCY_COLUMN, ON_SCENE_GROUPS
        else:
            urgency_column, groups = URGENCY_COLUMN, MACRO_GROUPS

        distributions[col] = {}
        for group_name, urgencies in groups.items():
            df_valid = df[df[col].notna() & (df[col] > 0) & df[urgency_column].isin(urgencies)]
            values, probs, _, _ = build_empirical_for_simpy(df_valid[col], n_points=n_points)
            if len(values):
                distributions[col][group_name] = (values, probs)

    distributions[BLOCK_COLUMN] = {}
    df_valid = df[df[BLOCK_COLUMN].notna() & (df[BLOCK_COLUMN] > 0)
                  & (df[BLOCK_COLUMN] <= MAX_ED_BLOCK_MINUTES)]
    for hospital, group in df_valid.groupby("OSPEDALE"):
        values, probs, _, _ = build_empirical_for_simpy(
            group[BLOCK_COLUMN], n_points=n_points, remove_outliers=False
        )
        if len(values):
            distributions[BLOCK_COLUMN][hospital] = (values, probs)

    values, probs, _, _ = build_empirical_for_simpy(
        df_valid[BLOCK_COLUMN], n_points=n_points, remove_outliers=False
    )
    if len(values):
        distributions[BLOCK_COLUMN][POOLED_BLOCK_GROUP] = (values, probs)

    return distributions
