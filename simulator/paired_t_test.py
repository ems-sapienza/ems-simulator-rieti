"""
Paired t-test on the per-replication coverages of the scenario reports.

Reads the report_<scenario>.xlsx files produced by model.run_and_export_kpi and
compares every scenario with the baseline (as_is) and every pair of scenarios.

    python paired_t_test.py                                    # report_scenarios
    python paired_t_test.py --report-dir report_scenarios_short_paper
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
BASELINE_NAME = "as_is"


def discover_scenarios(report_dir: Path) -> list[str]:
    """Scenarios found in the folder, with the baseline first."""
    names = [p.stem[len("report_"):] for p in sorted(report_dir.glob("report_*.xlsx"))]
    names = [n for n in names if n and n != "calibration"]
    if not names:
        raise FileNotFoundError(f"No report_*.xlsx in {report_dir}")
    return ([BASELINE_NAME] if BASELINE_NAME in names else []) + sorted(
        n for n in names if n != BASELINE_NAME
    )


def coverage_cols(df):
    return [c for c in df.columns if c.startswith("Coverage_")]


def paired_ttest(df_ref, df_alt, ref_name, alt_name, alpha=ALPHA):
    """Paired comparison, replication by replication, of every coverage."""
    rows = []
    metrics = sorted(set(coverage_cols(df_ref)).intersection(coverage_cols(df_alt)))

    for metric in metrics:
        merged = df_ref[["Replication", metric]].merge(
            df_alt[["Replication", metric]], on="Replication", suffixes=("_ref", "_alt")
        )

        ref = pd.to_numeric(merged[f"{metric}_ref"], errors="coerce")
        alt = pd.to_numeric(merged[f"{metric}_alt"], errors="coerce")
        valid = ref.notna() & alt.notna()
        ref = ref[valid].to_numpy(dtype=float)
        alt = alt[valid].to_numpy(dtype=float)

        diff = alt - ref
        n = len(diff)
        if n < 2:
            continue

        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        t_stat, p_value = stats.ttest_rel(alt, ref)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

        rows.append({
            "scenario_ref": ref_name,
            "scenario_alt": alt_name,
            "metric": metric,
            "n": n,
            "mean_ref": np.mean(ref),
            "mean_alt": np.mean(alt),
            "mean_diff_alt_minus_ref": mean_diff,
            "std_diff": std_diff,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_low": mean_diff - t_crit * se_diff,
            "ci_high": mean_diff + t_crit * se_diff,
            "significant": p_value < alpha,
            "improves": mean_diff > 0,
        })

    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("report_scenarios"),
                        help="folder with the report_<scenario>.xlsx files")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="scenarios to compare (default: every one found)")
    args = parser.parse_args(argv)

    output_dir = args.report_dir
    scenarios = args.scenarios or discover_scenarios(output_dir)

    # ---- Per-replication coverage of every scenario ----
    coverage_by_scenario = {}
    for name in scenarios:
        excel_path = output_dir / f"report_{name}.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"File not found: {excel_path}")
        coverage_by_scenario[name] = pd.read_excel(
            excel_path, sheet_name="Coverage_per_replication"
        )
        print(f"Read {name}: {excel_path}")

    df_all = pd.concat(
        [df.assign(scenario=name) for name, df in coverage_by_scenario.items()],
        ignore_index=True,
    )
    df_all = df_all[["scenario"] + [c for c in df_all.columns if c != "scenario"]]
    df_all.to_excel(output_dir / "coverage_per_replication_all_scenarios.xlsx", index=False)

    # ---- Every scenario against the baseline ----
    if BASELINE_NAME not in coverage_by_scenario:
        raise ValueError(f"The reference scenario '{BASELINE_NAME}' is missing")

    baseline_rows = []
    for name, df_cov in coverage_by_scenario.items():
        if name == BASELINE_NAME:
            continue
        baseline_rows.extend(paired_ttest(
            coverage_by_scenario[BASELINE_NAME], df_cov, BASELINE_NAME, name
        ))
    pd.DataFrame(baseline_rows).to_excel(
        output_dir / f"paired_ttest_vs_{BASELINE_NAME}.xlsx", index=False
    )

    # ---- Every pair ----
    pair_rows = []
    for s1, s2 in itertools.combinations(coverage_by_scenario, 2):
        pair_rows.extend(paired_ttest(
            coverage_by_scenario[s1], coverage_by_scenario[s2], s1, s2
        ))
    pd.DataFrame(pair_rows).to_excel(
        output_dir / "paired_ttest_all_pairs.xlsx", index=False
    )

    print(f"\nSaved in: {output_dir}")
    print("- coverage_per_replication_all_scenarios.xlsx")
    print(f"- paired_ttest_vs_{BASELINE_NAME}.xlsx")
    print("- paired_ttest_all_pairs.xlsx")


if __name__ == "__main__":
    main()
