"""
Estimate the simulation parameters from the raw mission-level dataset and write
them to `data/parameters/`.

The raw dataset is not part of this repository (it contains personal data): put
it in `raw/` or point `EMS_RAW_DATA` at it, then run

    python preprocessing/estimate_parameters.py

Outputs
-------
data/parameters/model_parameters.json   probabilities, transition matrices,
                                        hospital networks, cell weights
data/parameters/duration_distributions.csv
                                        empirical distribution of the duration
                                        of every phase of a mission
data/parameters/travel_time_correction_factors.xlsx
                                        copied from raw/scaling_factors.xlsx,
                                        if present
data/spatial/stations.csv               station code, name and coordinates
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

import estimation as est

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = Path(os.environ.get("EMS_RAW_DATA", ROOT / "raw" / "dataframe_RI.parquet"))
DEFAULT_GRID = Path(os.environ.get("EMS_RAW_GRID", ROOT / "raw" / "grid_RI.xlsx"))
DEFAULT_STATIONS = Path(os.environ.get("EMS_RAW_STATIONS", ROOT / "raw" / "unique_stations.csv"))

PROVINCE = "RI"
VEHICLE_TYPE = "ASI"

# Stations of the province as named in the raw data
PROVINCE_STATIONS = [
    "AMATRICE", "BORGO S.PIETRO", "LEONESSA", "MAGLIANO", "OSTERIA NUOVA",
    "PAGANICO", "PASSO CORESE", "POGGIO MIRTETO", "POSTA", "RIETI",
    "STIMIGLIANO SCALO", "TORRI IN SABINA",
]

# Stations hosted in fire-brigade barracks: no name in the raw data
BARRACK_NAMES = {
    "PMCASERMA": "POGGIO MIRTETO FS",
    "PCASERMA": "POSTA FS",
    "RCASERMA": "RIETI FS",
}


def durations_to_frame(distributions):
    """Flatten the empirical distributions into a tidy table."""
    rows = []

    for phase, col in est.DELAY_COLUMNS_BY_URGENCY.items():
        for group, (values, probs) in distributions[col].items():
            for value, prob in zip(values, probs):
                rows.append({"phase": phase, "group": group,
                             "minutes": value, "probability": prob})

    for phase, col in est.DELAY_COLUMNS_FLAT.items():
        if col not in distributions:
            continue
        values, probs = distributions[col]
        for value, prob in zip(values, probs):
            rows.append({"phase": phase, "group": "",
                         "minutes": value, "probability": prob})

    for hospital, (values, probs) in distributions[est.BLOCK_COLUMN].items():
        for value, prob in zip(values, probs):
            rows.append({"phase": "ed_block", "group": hospital,
                         "minutes": value, "probability": prob})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help="raw mission-level dataset (parquet)")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID,
                        help="grid definition (xlsx)")
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS,
                        help="candidate station locations (csv)")
    parser.add_argument("--out", type=Path, default=ROOT / "data",
                        help="output data directory")
    args = parser.parse_args()

    df = pd.read_parquet(args.raw, engine="pyarrow")
    df = df[df["TIPO MEZZO"] == VEHICLE_TYPE]
    # Only the missions of the province's own stations: the few emergencies
    # served by ambulances of neighbouring provinces are not demand on this fleet
    df = df[df["POSTAZIONE"].isin(PROVINCE_STATIONS)]
    df_province = df[df["Provincia Target"] == PROVINCE]
    grid = pd.read_excel(args.grid)
    stations = pd.read_csv(args.stations)

    cancelled = est.compute_mission_cancelled_pct(df)[PROVINCE]

    parameters = {
        "province": PROVINCE,
        "vehicle_type": VEHICLE_TYPE,
        # Order of appearance in the data: it determines the order in which the
        # generator processes are created, and therefore the sequence of random
        # numbers.
        "zones": df["Nucleo"].dropna().unique().tolist(),
        "hospitals": sorted(df["OSPEDALE"].dropna().unique().tolist()),
        "criticality_order": est.URGENCY_ORDER,
        "network_order": est.NETWORK_ORDER,
        "ambulances_order": est.AMBULANCES_ORDER,
        "colour_code_probabilities": est.compute_colour_code_probabilities(df)[PROVINCE],
        "network_probabilities_by_on_scene":
            est.compute_network_probabilities_by_on_scene(df)[PROVINCE],
        "hospitals_by_network": est.compute_hospitals_by_network(df)[PROVINCE],
        "ambulances_per_emergency": est.compute_ambulances_per_emergency(df)[PROVINCE],
        "transport_to_hospital_pct_by_on_scene":
            est.compute_transport_to_hospital_pct_by_on_scene(df)[PROVINCE],
        "mission_cancelled_pct": {
            "after_dispatch": cancelled["after_dispatch"],
            "after_departure": cancelled["after_departure"],
            "after_arrival": cancelled["after_arrival"],
        },
        "ed_block_pct_by_hospital": est.compute_ed_block_pct_by_hospital(df)[PROVINCE],
        "sanitation_pct": est.SANITATION_PCT_BY_PROVINCE[PROVINCE],
        "triage_transition_matrix": est.compute_transition_matrix(
            df_province,
            col_from="Criticità Al triage Telefonico",
            col_to="Criticità sul Posto",
        ),
        "cell_probabilities_by_zone": est.compute_cell_probabilities_by_zone(
            df, col_emergency_id="Numero Emergenza", col_cell="Quadrato", col_zone="Nucleo"
        ),
    }

    params_dir = args.out / "parameters"
    spatial_dir = args.out / "spatial"
    params_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)

    with open(params_dir / "model_parameters.json", "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=1, ensure_ascii=False)

    durations = durations_to_frame(est.build_empirical_distributions(df))
    durations.to_csv(params_dir / "duration_distributions.csv", index=False)

    # Station code -> name (only the cells that host an actual station)
    df_station = est.assign_stations_to_grid(
        df=df,
        station_columns={"lat": "LATITUDINE Postazione", "lon": "LONGITUDINE Postazione"},
        valid_stations=PROVINCE_STATIONS,
        grid=grid,
    )
    name_by_code = {
        row["name"]: row["POSTAZIONE"]
        for _, row in df_station.iterrows() if row["name"] is not None
    }
    name_by_code.update(BARRACK_NAMES)

    stations = stations.rename(columns={
        "POSTAZIONE": "code",
        "LATITUDINE Postazione": "lat",
        "LONGITUDINE Postazione": "lon",
    })
    stations["name"] = stations["code"].map(name_by_code).fillna("")
    stations[["code", "name", "lat", "lon"]].to_csv(spatial_dir / "stations.csv", index=False)

    # Travel-time correction factors (Table 3 of the paper): they are an input
    # of the preprocessing and are published exactly as they are
    factors = args.raw.parent / "scaling_factors.xlsx"
    if factors.exists():
        shutil.copyfile(factors, params_dir / "travel_time_correction_factors.xlsx")
        print(f"factors     -> {params_dir / 'travel_time_correction_factors.xlsx'}")

    print(f"parameters  -> {params_dir / 'model_parameters.json'}")
    print(f"durations   -> {params_dir / 'duration_distributions.csv'} "
          f"({len(durations)} rows)")
    print(f"stations    -> {spatial_dir / 'stations.csv'} "
          f"({(stations['name'] != '').sum()} named of {len(stations)})")


if __name__ == "__main__":
    main()
