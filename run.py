#!/usr/bin/env python
"""
Entry point of the repository: runs every step from any working directory.

    python run.py calibration              # model vs. real data
    python run.py scenarios                # fleet-allocation scenarios
    python run.py ttest                    # paired t-test of the scenarios
    python run.py tables                   # summary tables
    python run.py all                      # scenarios + ttest + tables

Add --example-data to run on the synthetic data of example-data/ (reports go
to report_*_example), and --short-paper to work on the scenarios of the short
paper instead of the main ones. Every command accepts the options of the script
it runs, e.g.

    python run.py calibration --replications 5
    python run.py ttest --scenarios as_is S1 S2

Reports are always written under `simulator/`, whatever the current directory.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT / "simulator"
EXAMPLE_DATA = ROOT / "example-data"

REPORT_DIRS = {
    False: "report_scenarios",
    True: "report_scenarios_short_paper",
}


def run_script(script, extra=()):
    """Runs a script of `simulator/` with that folder as working directory."""
    command = [sys.executable, script, *extra]
    print(f"\n$ {' '.join(command)}   (in {SIMULATOR})")
    outcome = subprocess.run(command, cwd=SIMULATOR)
    if outcome.returncode != 0:
        sys.exit(outcome.returncode)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=["calibration", "scenarios", "ttest", "tables", "all"],
        help="step to run",
    )
    parser.add_argument("--short-paper", action="store_true",
                        help="use the scenarios of the short paper")
    parser.add_argument("--example-data", action="store_true",
                        help="run on the synthetic data of example-data/ "
                             "(reports go to report_*_example)")
    args, extra = parser.parse_known_args()

    if not SIMULATOR.is_dir():
        sys.exit(f"Folder not found: {SIMULATOR}")

    report_dir = REPORT_DIRS[args.short_paper]
    calibration_dir = "report_calibration"
    if args.example_data:
        if not EXAMPLE_DATA.is_dir():
            sys.exit("Synthetic data not found: run python preprocessing/make_example_data.py")
        # read by simulator/model.py and simulator/scenarios*.py in the child processes
        os.environ["EMS_DATA_DIR"] = str(EXAMPLE_DATA)
        report_dir += "_example"
        calibration_dir += "_example"
        os.environ["EMS_REPORT_DIR"] = report_dir
    scenarios_script = ("scenarios_short_paper.py" if args.short_paper else "scenarios.py")

    if args.command == "calibration":
        run_script("model.py", ["--output-dir", calibration_dir, *extra])
    elif args.command == "scenarios":
        run_script(scenarios_script, extra)
    elif args.command == "ttest":
        run_script("paired_t_test.py", ["--report-dir", report_dir, *extra])
    elif args.command == "tables":
        run_script("aggregate_scenario_tables.py", ["--report-dir", report_dir, *extra])
    elif args.command == "all":
        run_script(scenarios_script, extra)
        run_script("paired_t_test.py", ["--report-dir", report_dir])
        run_script("aggregate_scenario_tables.py", ["--report-dir", report_dir])

    print(f"\nDone. Output in {SIMULATOR}")


if __name__ == "__main__":
    main()
