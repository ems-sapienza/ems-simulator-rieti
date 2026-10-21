"""
Ambulance reallocation scenarios for the province of Rieti.

Stations are given by name (the model translates them into the code of the grid
cell, see `model.station_code`). "FS" marks the stations hosted in the
fire-brigade barracks.
"""

import copy
import os

from model import run_and_export_kpi

NUM_REPLICATIONS = 30
OUTPUT_DIR = os.environ.get("EMS_REPORT_DIR", "report_scenarios")  # set by run.py --example-data


# ============================================================
# AS-IS SCENARIO
# ============================================================

BASE_DAY = {
    "AMATRICE": 1,
    "BORGO S.PIETRO": 1,
    "LEONESSA": 1,
    "MAGLIANO": 1,
    "OSTERIA NUOVA": 1,
    "PAGANICO": 1,
    "PASSO CORESE": 1,
    "POGGIO MIRTETO": 1,
    "POSTA": 1,
    "RIETI": 2,
    "STIMIGLIANO SCALO": 1,
    "TORRI IN SABINA": 1,

    "POGGIO MIRTETO FS": 0,
    "POSTA FS": 0,
    "RIETI FS": 0,
}

BASE_NIGHT = copy.deepcopy(BASE_DAY)


# ============================================================
# SCENARIOS
# ============================================================

scenarios = {
    "as_is": {
        "day": BASE_DAY,
        "night": BASE_NIGHT,
    },

    # S1: as-is + 1 H12 in Rieti
    "S1": {
        "day": {**BASE_DAY, "RIETI": BASE_DAY["RIETI"] + 1},
        "night": BASE_NIGHT,
    },

    # S2: one of the two H24 in Rieti becomes H12 (12 H24 + 1 H12)
    "S2": {
        "day": {**BASE_DAY, "RIETI": 2},
        "night": {**BASE_NIGHT, "RIETI": 1},
    },

    # S3: as-is + 1 H24 in Poggio Mirteto FS
    "S3": {
        "day": {**BASE_DAY, "POGGIO MIRTETO FS": 1},
        "night": {**BASE_NIGHT, "POGGIO MIRTETO FS": 1},
    },

    # S4: as-is + 1 H24 in Posta FS
    "S4": {
        "day": {**BASE_DAY, "POSTA FS": 1},
        "night": {**BASE_NIGHT, "POSTA FS": 1},
    },

    # S5: as-is + 1 H24 in Rieti FS
    "S5": {
        "day": {**BASE_DAY, "RIETI FS": 1},
        "night": {**BASE_NIGHT, "RIETI FS": 1},
    },

    # S6: move Poggio Mirteto to Poggio Mirteto FS
    "S6": {
        "day": {**BASE_DAY, "POGGIO MIRTETO": 0, "POGGIO MIRTETO FS": 1},
        "night": {**BASE_NIGHT, "POGGIO MIRTETO": 0, "POGGIO MIRTETO FS": 1},
    },

    # S7: move Posta to Posta FS
    "S7": {
        "day": {**BASE_DAY, "POSTA": 0, "POSTA FS": 1},
        "night": {**BASE_NIGHT, "POSTA": 0, "POSTA FS": 1},
    },

    # S8: move one ambulance from Rieti to Rieti FS
    "S8": {
        "day": {**BASE_DAY, "RIETI": 1, "RIETI FS": 1},
        "night": {**BASE_NIGHT, "RIETI": 1, "RIETI FS": 1},
    },
}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    for name, sc in scenarios.items():
        print(f"\nRunning scenario: {name}")
        print(f"Total day:   {sum(sc['day'].values())}")
        print(f"Total night: {sum(sc['night'].values())}")

        run_and_export_kpi(
            num_repl=NUM_REPLICATIONS,
            capacities_day=sc["day"],
            capacities_night=sc["night"],
            output_dir=OUTPUT_DIR,
            label=name,
        )
