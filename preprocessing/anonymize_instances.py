"""
Build the anonymised emergency instances published in `data/anonymized-instances/`.

What is removed
---------------
- direct identifiers: emergency and mission numbers, vehicle codes (replaced by
  a sequential id that carries no information)
- exact location of the emergency: address, municipality, locality, latitude,
  longitude, geometry (only the 2 km grid cell and the operational zone are kept)
- clinical information: pathology, pathology class, medical assessment
- exact timestamps: only the calendar day and the hour of the call are kept;
  everything else is expressed as a duration in minutes relative to the call

What is kept
------------
Everything the simulator needs to re-estimate its parameters and to reproduce
the calibration: grid cell, zone, urgency at triage and on scene, hospital and
hospital network, station, mission outcome, ED block, and the duration of each
phase of the mission.

The columns of the raw dataset carry the Italian names of the source system;
every published column is renamed to English here.

Usage
-----
    python preprocessing/anonymize_instances.py
"""

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = Path(os.environ.get("EMS_RAW_DATA", ROOT / "raw" / "dataframe_RI.parquet"))

VEHICLE_TYPE = "ASI"

# Duration columns (minutes between two phases of the mission): raw name -> published name
DURATION_RENAME = {
    "diff_FineTriage_Chiamata": "diff_triage_end_call",
    "diff_Invio_FineTriage": "diff_dispatch_triage_end",
    "diff_Partenza_Invio": "diff_departure_dispatch",
    "diff_Arrivo_Partenza": "diff_arrival_departure",
    "diff_Carico_Arrivo": "diff_loading_arrival",
    "diff_Ospedale_Carico": "diff_hospital_loading",
    "diff_Chiusura_Ospedale": "diff_close_hospital",
    "diff_Chiusura_Blocco": "diff_close_block",
    "diff_Chiusura_Arrivo": "diff_close_arrival",
    "diff_Chiusura_MissAnn(invio)": "diff_close_cancelled_dispatch",
    "diff_Chiusura_MissAnn(partenza)": "diff_close_cancelled_departure",
    "diff_Chiusura_MissAnn(arrivo)": "diff_close_cancelled_arrival",
    "diff_Chiusura_MissAnn(ospedale)": "diff_close_cancelled_hospital",
    "diff_Chiusura_MissAnn(carico)": "diff_close_cancelled_loading",
}
DURATION_COLUMNS = list(DURATION_RENAME.values())

# Columns published under a new name
RENAME = {
    "Provincia Target": "province",
    "Nucleo": "zone",
    "Quadrato": "cell",
    "Criticità Al triage Telefonico": "urgency_triage",
    "Criticità sul Posto": "urgency_on_scene",
    "POSTAZIONE": "station",
    "TIPO MEZZO": "vehicle_type",
    "OSPEDALE": "hospital",
    "rete ospedaliera": "hospital_network",
    "ESITO MISSIONE": "mission_outcome",
    "MOTIVO_ANNULLAMENTO": "cancellation_reason",
    "Ultima Colonna Non Vuota": "last_filled_field",
    **DURATION_RENAME,
}

COLUMN_ORDER = [
    "emergency_id", "date", "hour", "time_slot", "province", "zone", "cell",
    "urgency_triage", "urgency_on_scene", "station", "vehicle_type",
    "hospital", "hospital_network", "mission_outcome", "cancellation_reason",
    "last_filled_field", "arrived_on_scene", "ed_block", "response_time_min",
    "cycle_time_min",
]


def time_slot(hour):
    if hour < 7:
        return "00_07"
    if hour < 12:
        return "07_12"
    if hour < 18:
        return "12_18"
    return "18_24"


def anonymise(df):
    """Return the anonymised mission-level table."""
    for col in ["Giorno e OraChiamata", "DATA_ORA_ARRIVO", "DATA_ORA_CHIUSURA"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    out = df.rename(columns=RENAME).copy()

    # Sequential, meaningless emergency id (missions of the same emergency keep
    # the same id, which is what the parameter estimation needs)
    codes = {
        number: f"E{i:05d}"
        for i, number in enumerate(sorted(df["Numero Emergenza"].dropna().unique()), start=1)
    }
    out["emergency_id"] = df["Numero Emergenza"].map(codes)

    # Temporal information: calendar day and hour only
    call = df["Giorno e OraChiamata"]
    out["date"] = call.dt.date
    out["hour"] = call.dt.hour
    out["time_slot"] = out["hour"].map(time_slot, na_action="ignore")

    # Derived durations (minutes)
    out["response_time_min"] = (df["DATA_ORA_ARRIVO"] - call).dt.total_seconds() / 60
    out["cycle_time_min"] = (df["DATA_ORA_CHIUSURA"] - call).dt.total_seconds() / 60
    out["arrived_on_scene"] = df["DATA_ORA_ARRIVO"].notna()
    out["ed_block"] = df["Data e Ora Inizio Blocco"].notna()

    columns = COLUMN_ORDER + [c for c in DURATION_COLUMNS if c in out.columns]
    out = out[columns].sort_values(["date", "hour", "emergency_id"], kind="stable")
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "anonymized-instances")
    parser.add_argument("--year", type=int, default=None,
                        help="keep only the calls of this year")
    args = parser.parse_args()

    df = pd.read_parquet(args.raw, engine="pyarrow")
    df = df[df["TIPO MEZZO"] == VEHICLE_TYPE].copy()

    out = anonymise(df)
    if args.year is not None:
        out = out[pd.to_datetime(out["date"]).dt.year == args.year]

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "emergencies.csv"
    out.to_csv(destination, index=False)

    print(f"{len(out)} missions, {out['emergency_id'].nunique()} emergencies "
          f"-> {destination}")
    print(f"period: {out['date'].min()} .. {out['date'].max()}")
    print(f"columns: {', '.join(out.columns)}")


if __name__ == "__main__":
    main()
