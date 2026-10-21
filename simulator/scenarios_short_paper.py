"""
Scenarios of the short paper: alternative configurations of the Rieti fleet.

Stations are given by name (the model translates them into the code of the grid
cell, see `model.station_code`). "FS" marks the stations hosted in the
fire-brigade barracks.
In every scenario the daytime and the night-time configurations coincide.
"""

import copy
import os

from model import run_and_export_kpi

NUM_REPLICATIONS = 30
OUTPUT_DIR = os.environ.get("EMS_REPORT_DIR", "report_scenarios_short_paper")  # set by run.py --example-data


BASE = {
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


scenario_config = {
    "as_is": BASE,

    # A: one ambulance in each of the three barracks, Rieti down to 1
    "A": {
        **BASE,
        "RIETI": 1,
        "POGGIO MIRTETO FS": 1,
        "POSTA FS": 1,
        "RIETI FS": 1,
    },

    # B: like A, without Amatrice and Posta
    "B": {
        **BASE,
        "AMATRICE": 0,
        "POSTA": 0,
        "RIETI": 1,
        "POGGIO MIRTETO FS": 1,
        "POSTA FS": 1,
        "RIETI FS": 1,
    },

    # C: like B, without Magliano and with 2 ambulances in Rieti FS
    "C": {
        **BASE,
        "AMATRICE": 0,
        "MAGLIANO": 0,
        "POSTA": 0,
        "RIETI": 1,
        "POGGIO MIRTETO FS": 1,
        "POSTA FS": 1,
        "RIETI FS": 2,
    },

    # D: like C, without Paganico and with Rieti back to 2
    "D": {
        **BASE,
        "AMATRICE": 0,
        "MAGLIANO": 0,
        "PAGANICO": 0,
        "POSTA": 0,
        "RIETI": 2,
        "POGGIO MIRTETO FS": 1,
        "POSTA FS": 1,
        "RIETI FS": 2,
    },

    # E: reinforced as-is (Osteria Nuova 2, Rieti 3), no barracks
    "E": {
        **BASE,
        "OSTERIA NUOVA": 2,
        "RIETI": 3,
    },

    # F: like E, without Amatrice and Magliano
    "F": {
        **BASE,
        "AMATRICE": 0,
        "MAGLIANO": 0,
        "OSTERIA NUOVA": 2,
        "RIETI": 3,
    },

    # G: like F, without Paganico and with 4 ambulances in Rieti
    "G": {
        **BASE,
        "AMATRICE": 0,
        "MAGLIANO": 0,
        "PAGANICO": 0,
        "OSTERIA NUOVA": 2,
        "RIETI": 4,
    },
}


scenarios = {
    name: {
        "day": copy.deepcopy(config),
        "night": copy.deepcopy(config),
    }
    for name, config in scenario_config.items()
}


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
