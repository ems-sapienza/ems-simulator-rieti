"""
Aggregates the scenario reports into the summary tables of the paper.

Reads the report_<scenario>.xlsx files produced by model.run_and_export_kpi and
the output of the paired t-test, and writes one Excel file with the coverage,
vehicle-utilisation, response-time and t-test tables.

    python aggregate_scenario_tables.py                        # report_scenarios
    python aggregate_scenario_tables.py --report-dir report_scenarios_short_paper
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PARAMETERS
# ============================================================

# Report folder and scenarios: both can be given on the command line
# (--report-dir, --scenarios). By default report_scenarios is used and every
# scenario found in the folder is taken.
REPORT_DIR = Path("report_scenarios")
OUTPUT_FILE = REPORT_DIR / "scenario_summary_tables.xlsx"

# File already produced by the paired t-test
PAIRED_TTEST_FILE = REPORT_DIR / "paired_ttest_vs_as_is.xlsx"

SCENARIO_NAMES = []


ROUND_DIGITS = 3
ALPHA = 0.05
THRESHOLD_FOR_TTEST_TABLE = 20

# For the paired t-test:
# if diff was computed as scenario - as_is, leave "alt_minus_ref"
# if diff was computed as as_is - scenario, use "ref_minus_alt"
DIFF_DIRECTION = "alt_minus_ref"

# For coverage: True, because more coverage = better.
# For response time: False, because less time = better.
HIGHER_IS_BETTER = True


def discover_scenarios(report_dir: Path) -> list[str]:
    """Scenarios found in the folder, with as_is first."""
    names = [p.stem[len("report_"):] for p in sorted(report_dir.glob("report_*.xlsx"))]
    names = [n for n in names if n and n != "calibration"]
    if not names:
        raise FileNotFoundError(f"No report_*.xlsx in {report_dir}")
    return (["as_is"] if "as_is" in names else []) + sorted(n for n in names if n != "as_is")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("report_scenarios"),
                        help="folder with the report_<scenario>.xlsx files")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="scenarios to compare (default: every one found)")
    return parser.parse_args(argv)


# ============================================================
# EXCEL REPORT UTILITIES
# ============================================================

def report_path(scenario: str) -> Path:
    return REPORT_DIR / f"report_{scenario}.xlsx"


def read_report_sheet(scenario: str, sheet_name: str) -> pd.DataFrame:
    path = report_path(scenario)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(f"Sheet '{sheet_name}' not found in {path}") from exc


def mean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.Series({
        c: pd.to_numeric(df[c], errors="coerce").mean()
        for c in cols
    })


def std_numeric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.Series({
        c: pd.to_numeric(df[c], errors="coerce").std(ddof=1)
        for c in cols
    })


def ci95_half_width(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = {}

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        n = len(x)

        if n < 2:
            out[c] = np.nan
        else:
            try:
                from scipy.stats import t
                tcrit = t.ppf(0.975, df=n - 1)
            except Exception:
                tcrit = 1.96

            out[c] = tcrit * x.std(ddof=1) / np.sqrt(n)

    return pd.Series(out)


def clean_metric_name(col: str) -> str:
    name = col
    name = name.replace("Coverage_", "")
    name = name.replace("Mean_response_time_", "Mean RT ")
    name = name.replace("Worst_5pct_response_time_", "Worst 5% RT ")
    name = name.replace("_", " ")
    name = name.replace("lt", "<")
    return name


def threshold_from_coverage_col(col: str) -> int:
    m = re.search(r"_lt(\d+)", col)
    return int(m.group(1)) if m else 9999


def sort_coverage_cols(cols: list[str]) -> list[str]:
    return sorted(
        cols,
        key=lambda c: (
            "NON_EMER" in c,
            threshold_from_coverage_col(c),
        )
    )


def add_scenario_column_first(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Scenario"] + [c for c in df.columns if c != "Scenario"]
    return df[cols]


def build_mean_table_from_sheet(
    sheet_name: str,
    columns: list[str],
    rename_cols: bool = True,
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIO_NAMES:
        df = read_report_sheet(scenario, sheet_name)

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Sheet '{sheet_name}' of scenario '{scenario}' is missing columns: {missing}"
            )

        s = mean_numeric(df, columns)
        s["Scenario"] = scenario
        rows.append(s)

    out = pd.DataFrame(rows)
    out = add_scenario_column_first(out)

    if rename_cols:
        out = out.rename(columns={c: clean_metric_name(c) for c in columns})

    return out.round(ROUND_DIGITS)


def build_stats_long_from_sheet(
    sheet_name: str,
    columns: list[str],
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIO_NAMES:
        df = read_report_sheet(scenario, sheet_name)

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Sheet '{sheet_name}' of scenario '{scenario}' is missing columns: {missing}"
            )

        mean_s = mean_numeric(df, columns)
        std_s = std_numeric(df, columns)
        ci_s = ci95_half_width(df, columns)

        for col in columns:
            rows.append({
                "Scenario": scenario,
                "Metric": clean_metric_name(col),
                "Mean": mean_s[col],
                "Std": std_s[col],
                "CI95_low": mean_s[col] - ci_s[col],
                "CI95_high": mean_s[col] + ci_s[col],
                "CI95_half_width": ci_s[col],
            })

    return pd.DataFrame(rows).round(ROUND_DIGITS)


# ============================================================
# TABLES FROM THE KPIs ALREADY COMPUTED
# ============================================================

def build_coverage_20min_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    first_cov = read_report_sheet(SCENARIO_NAMES[0], "Coverage_per_replication")
    coverage_cols = [c for c in first_cov.columns if c.startswith("Coverage_")]

    coverage_20_cols = [
        c for c in coverage_cols
        if threshold_from_coverage_col(c) == 20
    ]

    coverage_20_cols = sort_coverage_cols(coverage_20_cols)

    wide = build_mean_table_from_sheet(
        sheet_name="Coverage_per_replication",
        columns=coverage_20_cols,
        rename_cols=True,
    )

    long = build_stats_long_from_sheet(
        sheet_name="Coverage_per_replication",
        columns=coverage_20_cols,
    )

    return wide, long


def build_coverage_all_thresholds_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    first_cov = read_report_sheet(SCENARIO_NAMES[0], "Coverage_per_replication")
    coverage_cols = [c for c in first_cov.columns if c.startswith("Coverage_")]
    coverage_cols = sort_coverage_cols(coverage_cols)

    wide = build_mean_table_from_sheet(
        sheet_name="Coverage_per_replication",
        columns=coverage_cols,
        rename_cols=True,
    )

    long = build_stats_long_from_sheet(
        sheet_name="Coverage_per_replication",
        columns=coverage_cols,
    )

    return wide, long


def build_vehicle_utilization_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    by_station_parts = []

    for scenario in SCENARIO_NAMES:
        df_raw = read_report_sheet(scenario, "Vehicle_Utilization")

        needed_raw = [
            "Replication",
            "Total_service_time_min",
            "Working_time_min",
        ]

        missing_raw = [c for c in needed_raw if c not in df_raw.columns]
        if missing_raw:
            raise KeyError(
                f"Sheet Vehicle_Utilization of scenario '{scenario}' is missing columns: {missing_raw}"
            )

        repl = (
            df_raw
            .groupby("Replication", as_index=False)
            .agg(
                Total_service_time_min=("Total_service_time_min", "sum"),
                Working_time_min=("Working_time_min", "sum"),
            )
        )

        repl["Vehicle_utilization_overall_pct"] = np.where(
            repl["Working_time_min"] > 0,
            repl["Total_service_time_min"] / repl["Working_time_min"] * 100,
            np.nan,
        )

        x = pd.to_numeric(repl["Vehicle_utilization_overall_pct"], errors="coerce").dropna()

        overall_rows.append({
            "Scenario": scenario,
            "Mean_vehicle_utilization_overall_pct": x.mean(),
            "Std_vehicle_utilization_overall_pct": x.std(ddof=1),
            "Min_vehicle_utilization_overall_pct": x.min(),
            "Max_vehicle_utilization_overall_pct": x.max(),
        })

        df_sum = read_report_sheet(scenario, "Vehicle_Util_summary")

        needed_sum = ["Station", "Mean_utilization_pct"]
        missing_sum = [c for c in needed_sum if c not in df_sum.columns]
        if missing_sum:
            raise KeyError(
                f"Sheet Vehicle_Util_summary of scenario '{scenario}' is missing columns: {missing_sum}"
            )

        tmp = df_sum[["Station", "Mean_utilization_pct"]].copy()
        tmp = tmp.rename(columns={"Mean_utilization_pct": scenario})
        by_station_parts.append(tmp)

    by_station = by_station_parts[0]

    for part in by_station_parts[1:]:
        by_station = by_station.merge(part, on="Station", how="outer")

    by_station = by_station.sort_values("Station").reset_index(drop=True)

    overall = pd.DataFrame(overall_rows)

    return overall.round(ROUND_DIGITS), by_station.round(ROUND_DIGITS)


def build_avg_response_time_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds:
    - Avg_response_time: short-paper table with Mean, Std, Min, Max, LB, UB
    - Avg_RT_stats: full long table, including Mean RT ALL
    """

    metric_map = {
        "Urgent": "Mean_response_time_EMER",
        "Non-Urgent": "Mean_response_time_NONEMER",
    }

    rows = []

    for scenario in SCENARIO_NAMES:
        df = read_report_sheet(scenario, "Extra_KPI")

        for class_name, col in metric_map.items():
            if col not in df.columns:
                raise KeyError(
                    f"Sheet Extra_KPI of scenario '{scenario}' is missing the column: {col}"
                )

            x = pd.to_numeric(df[col], errors="coerce").dropna()
            n = len(x)

            mean_val = x.mean()
            std_val = x.std(ddof=1)
            min_val = x.min()
            max_val = x.max()

            if n < 2:
                ci_low = np.nan
                ci_high = np.nan
            else:
                try:
                    from scipy.stats import t
                    tcrit = t.ppf(0.975, df=n - 1)
                except Exception:
                    tcrit = 1.96

                se = std_val / np.sqrt(n)
                ci_low = mean_val - tcrit * se
                ci_high = mean_val + tcrit * se

            rows.append({
                "Scenario": scenario,
                "Class": class_name,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
                "LB": ci_low,
                "UB": ci_high,
            })

    wide = pd.DataFrame(rows).round(ROUND_DIGITS)

    cols_stats = [
        "Mean_response_time_ALL",
        "Mean_response_time_EMER",
        "Mean_response_time_NONEMER",
    ]

    long = build_stats_long_from_sheet(
        sheet_name="Extra_KPI",
        columns=cols_stats,
    )

    return wide, long


def build_worst_5_response_time_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds:
    - Worst5_response_time: short-paper table with Mean, Std, Min, Max, LB, UB
    - Worst5_RT_stats: full long table, including Worst 5% RT ALL
    """

    metric_map = {
        "Urgent": "Worst_5pct_response_time_EMER",
        "Non-Urgent": "Worst_5pct_response_time_NONEMER",
    }

    rows = []

    for scenario in SCENARIO_NAMES:
        df = read_report_sheet(scenario, "Extra_KPI")

        for class_name, col in metric_map.items():
            if col not in df.columns:
                raise KeyError(
                    f"Sheet Extra_KPI of scenario '{scenario}' is missing the column: {col}"
                )

            x = pd.to_numeric(df[col], errors="coerce").dropna()
            n = len(x)

            mean_val = x.mean()
            std_val = x.std(ddof=1)
            min_val = x.min()
            max_val = x.max()

            if n < 2:
                ci_low = np.nan
                ci_high = np.nan
            else:
                try:
                    from scipy.stats import t
                    tcrit = t.ppf(0.975, df=n - 1)
                except Exception:
                    tcrit = 1.96

                se = std_val / np.sqrt(n)
                ci_low = mean_val - tcrit * se
                ci_high = mean_val + tcrit * se

            rows.append({
                "Scenario": scenario,
                "Class": class_name,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
                "LB": ci_low,
                "UB": ci_high,
            })

    wide = pd.DataFrame(rows).round(ROUND_DIGITS)

    cols_stats = [
        "Worst_5pct_response_time_ALL",
        "Worst_5pct_response_time_EMER",
        "Worst_5pct_response_time_NONEMER",
    ]

    long = build_stats_long_from_sheet(
        sheet_name="Extra_KPI",
        columns=cols_stats,
    )

    return wide, long

# ============================================================
# 6) RISK-AVERSE RESPONSE TIME
# Worst 10%, 5%, 1% by scenario and urgency class
# ============================================================

def build_risk_averse_rt_table() -> pd.DataFrame:
    """
    Builds a table shaped like:

    Scenario | Class | Worst 10% (CVaR 90%) | Worst 5% (CVaR 95%) | Worst 1% (CVaR 99%)

    The values are means over the replications, read from the Extra_KPI sheet.
    """

    metric_map = {
        "Urgent": {
            "Worst 10% (CVaR 90%)": "Worst_10pct_response_time_EMER",
            "Worst 5% (CVaR 95%)": "Worst_5pct_response_time_EMER",
            "Worst 1% (CVaR 99%)": "Worst_1pct_response_time_EMER",
        },
        "Non-Urgent": {
            "Worst 10% (CVaR 90%)": "Worst_10pct_response_time_NONEMER",
            "Worst 5% (CVaR 95%)": "Worst_5pct_response_time_NONEMER",
            "Worst 1% (CVaR 99%)": "Worst_1pct_response_time_NONEMER",
        },
    }

    rows = []

    for scenario in SCENARIO_NAMES:
        df = read_report_sheet(scenario, "Extra_KPI")

        for class_name, cols in metric_map.items():
            missing = [c for c in cols.values() if c not in df.columns]

            if missing:
                raise KeyError(
                    f"Sheet Extra_KPI of scenario '{scenario}' is missing columns: {missing}"
                )

            row = {
                "Scenario": scenario,
                "Class": class_name,
            }

            for output_col, source_col in cols.items():
                row[output_col] = pd.to_numeric(df[source_col], errors="coerce").mean()

            rows.append(row)

    out = pd.DataFrame(rows)

    out["_scenario_order"] = out["Scenario"].apply(scenario_order)
    out["_class_order"] = out["Class"].map({
        "Urgent": 0,
        "Non-Urgent": 1,
    })

    out = (
        out.sort_values(["_scenario_order", "Scenario", "_class_order"])
        .drop(columns=["_scenario_order", "_class_order"])
        .reset_index(drop=True)
    )

    return out.round(ROUND_DIGITS)

# ============================================================
# TABLES FROM THE PAIRED T-TEST
# ============================================================

def read_paired_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Paired t-test file not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Format not handled: {path.suffix}")


def get_scenario_col(df: pd.DataFrame) -> str:
    for c in ["scenario_alt", "scenario_b", "scenario"]:
        if c in df.columns:
            return c
    raise KeyError("Column scenario_alt / scenario_b / scenario not found.")


def get_metric_col(df: pd.DataFrame) -> str:
    if "metric" in df.columns:
        return "metric"
    raise KeyError("Column metric not found.")


def get_mean_diff_col(df: pd.DataFrame) -> str:
    for c in ["mean_diff_alt_minus_ref", "mean_diff", "Mean diff.", "Mean diff"]:
        if c in df.columns:
            return c
    raise KeyError("Column of the mean difference not found.")


def get_ci_low_col(df: pd.DataFrame) -> str:
    for c in ["ci_low", "CI_low", "ci95_low", "CI95_low"]:
        if c in df.columns:
            return c
    raise KeyError("Column ci_low not found.")


def get_ci_high_col(df: pd.DataFrame) -> str:
    for c in ["ci_high", "CI_high", "ci95_high", "CI95_high"]:
        if c in df.columns:
            return c
    raise KeyError("Column ci_high not found.")


def is_significant(row: pd.Series) -> bool:
    if "p_value" in row.index and pd.notna(row["p_value"]):
        return float(row["p_value"]) < ALPHA

    if "significant" in row.index and pd.notna(row["significant"]):
        val = str(row["significant"]).strip().lower()
        return val in ["true", "1", "yes", "y", "significant"]

    if "significance" in row.index and pd.notna(row["significance"]):
        val = str(row["significance"]).strip().lower()
        return "significant" in val and "not" not in val

    raise KeyError("None of p_value / significant / significance found.")


def urgency_from_metric(metric: str) -> str:
    m = str(metric).upper()

    if "NON_EMER" in m or "NON EMER" in m or "NON-EMER" in m:
        return "Non-urgent"

    if "EMER" in m or "URGENT" in m:
        return "Urgent"

    return "Unknown"


def threshold_from_metric(metric: str) -> int | None:
    text = str(metric)

    patterns = [
        r"lt\s*(\d+)",
        r"<\s*(\d+)",
        r"threshold[_\s]*(\d+)",
    ]

    for p in patterns:
        match = re.search(p, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def urgency_order(urgency: str) -> int:
    order = {
        "Non-urgent": 0,
        "Urgent": 1,
        "Unknown": 2,
    }
    return order.get(urgency, 99)


def scenario_order(scenario: str) -> int:
    match = re.search(r"(\d+)", str(scenario))
    return int(match.group(1)) if match else 0


def effect_alt_minus_ref(mean_diff: float) -> float:
    if DIFF_DIRECTION == "alt_minus_ref":
        return mean_diff

    if DIFF_DIRECTION == "ref_minus_alt":
        return -mean_diff

    raise ValueError("DIFF_DIRECTION must be 'alt_minus_ref' or 'ref_minus_alt'.")


def classify_result(mean_diff: float, significant: bool) -> str:
    if not significant:
        return "Not significant"

    effect = effect_alt_minus_ref(mean_diff)

    if HIGHER_IS_BETTER:
        if effect > 0:
            return "Significant improvement"
        if effect < 0:
            return "Significant worsening"
    else:
        if effect < 0:
            return "Significant improvement"
        if effect > 0:
            return "Significant worsening"

    return "Not significant"


def format_signed(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return ""

    if x > 0:
        return f"+{x:.{digits}f}"

    return f"{x:.{digits}f}"


def format_ci(low: float, high: float, digits: int = 2) -> str:
    return f"({format_signed(low, digits)}, {format_signed(high, digits)})"


def build_ttest_threshold_table(df: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    scenario_col = get_scenario_col(df)
    metric_col = get_metric_col(df)
    mean_col = get_mean_diff_col(df)
    low_col = get_ci_low_col(df)
    high_col = get_ci_high_col(df)

    work = df.copy()
    work["Threshold"] = work[metric_col].apply(threshold_from_metric)
    work["Urgency"] = work[metric_col].apply(urgency_from_metric)

    work = work[work["Threshold"] == threshold].copy()

    if work.empty:
        raise RuntimeError(f"No metric found for threshold = {threshold}.")

    rows = []

    for _, row in work.iterrows():
        mean_diff = float(row[mean_col])
        ci_low = float(row[low_col])
        ci_high = float(row[high_col])
        significant = is_significant(row)

        rows.append({
            "Scenario": row[scenario_col],
            "Urgency": row["Urgency"],
            "Mean diff.": format_signed(mean_diff, 2),
            "95% CI": format_ci(ci_low, ci_high, 2),
            "Result": classify_result(mean_diff, significant),
        })

    out = pd.DataFrame(rows)
    out["_scenario_order"] = out["Scenario"].apply(scenario_order)
    out["_urgency_order"] = out["Urgency"].apply(urgency_order)

    out = (
        out.sort_values(["_scenario_order", "Scenario", "_urgency_order"])
        .drop(columns=["_scenario_order", "_urgency_order"])
        .reset_index(drop=True)
    )

    return out


def interpret_summary(row: pd.Series, max_improvements_no_worsening: int, n_metrics: int) -> str:
    imp = int(row["Significant improvements"])
    wors = int(row["Significant worsenings"])

    if imp > 0 and wors > 0:
        return "Mixed (trade-offs)"

    if imp == 0 and wors == 0:
        return "Essentially neutral"

    if wors == 0:
        if imp == max_improvements_no_worsening and imp >= 0.75 * n_metrics:
            return "Dominant (robust-wide improvement)"
        if imp >= 0.75 * n_metrics:
            return "Strong improvement"
        if imp >= 0.30 * n_metrics:
            return "Selective improvement"
        return "Essentially neutral"

    if imp == 0:
        if wors >= 0.75 * n_metrics:
            return "Systematically worsening"
        return "Mild negative"

    return "Mixed (trade-offs)"


def build_ttest_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    scenario_col = get_scenario_col(df)
    mean_col = get_mean_diff_col(df)

    work = df.copy()

    work["Result_class"] = [
        classify_result(float(row[mean_col]), is_significant(row))
        for _, row in work.iterrows()
    ]

    summary = (
        work
        .groupby(scenario_col)
        .agg(
            **{
                "Significant improvements": (
                    "Result_class",
                    lambda x: int((x == "Significant improvement").sum())
                ),
                "Significant worsenings": (
                    "Result_class",
                    lambda x: int((x == "Significant worsening").sum())
                ),
                "Not significant": (
                    "Result_class",
                    lambda x: int((x == "Not significant").sum())
                ),
            }
        )
        .reset_index()
        .rename(columns={scenario_col: "Scenario"})
    )

    n_metrics = (
        summary["Significant improvements"]
        + summary["Significant worsenings"]
        + summary["Not significant"]
    ).max()

    no_wors = summary[summary["Significant worsenings"] == 0]

    max_imp_no_wors = (
        int(no_wors["Significant improvements"].max())
        if not no_wors.empty
        else 0
    )

    summary["Interpretation"] = summary.apply(
        lambda r: interpret_summary(r, max_imp_no_wors, n_metrics),
        axis=1,
    )

    summary = summary.drop(columns=["Not significant"])
    summary["_scenario_order"] = summary["Scenario"].apply(scenario_order)

    summary = (
        summary
        .sort_values(
            ["Significant improvements", "Significant worsenings", "_scenario_order"],
            ascending=[False, True, True],
        )
        .drop(columns=["_scenario_order"])
        .reset_index(drop=True)
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cov20, cov20_stats = build_coverage_20min_table()
    cov_all, cov_all_stats = build_coverage_all_thresholds_table()
    veh_overall, veh_by_station = build_vehicle_utilization_tables()
    avg_rt, avg_rt_stats = build_avg_response_time_table()
    worst5, worst5_stats = build_worst_5_response_time_table()
    risk_averse_rt = build_risk_averse_rt_table()

    paired_df = read_paired_results(PAIRED_TTEST_FILE)
    ttest_20 = build_ttest_threshold_table(
        paired_df,
        threshold=THRESHOLD_FOR_TTEST_TABLE,
    )
    ttest_summary = build_ttest_summary_table(paired_df)

    dfs = {
        "Coverage_20min": cov20,
        "Coverage_20min_stats": cov20_stats,
        "Coverage_all_thresholds": cov_all,
        "Coverage_all_stats": cov_all_stats,
        "Vehicle_util_overall": veh_overall,
        "Vehicle_util_station": veh_by_station,
        "Avg_response_time": avg_rt,
        "Avg_RT_stats": avg_rt_stats,
        "Worst5_response_time": worst5,
        "Worst5_RT_stats": worst5_stats,
        "Risk_averse_RT": risk_averse_rt,
        "Ttest_20min": ttest_20,
        "Ttest_summary": ttest_summary,
    }

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book

        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "align": "center",
        })

        cell_fmt = workbook.add_format({
            "align": "center",
        })

        num_fmt = workbook.add_format({
            "num_format": "0.000",
            "align": "center",
        })

        for sheet_name, df in dfs.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 1)

            for col_idx, col_name in enumerate(df.columns):
                ws.write(0, col_idx, col_name, header_fmt)
                width = max(12, min(45, len(str(col_name)) + 4))

                if pd.api.types.is_numeric_dtype(df[col_name]):
                    ws.set_column(col_idx, col_idx, width, num_fmt)
                else:
                    ws.set_column(col_idx, col_idx, width, cell_fmt)

    print(f"File saved in: {OUTPUT_FILE}")


if __name__ == "__main__":
    args = parse_args()
    REPORT_DIR = args.report_dir
    OUTPUT_FILE = REPORT_DIR / "scenario_summary_tables.xlsx"
    PAIRED_TTEST_FILE = REPORT_DIR / "paired_ttest_vs_as_is.xlsx"
    SCENARIO_NAMES = args.scenarios or discover_scenarios(REPORT_DIR)
    print(f"Scenarios: {', '.join(SCENARIO_NAMES)}")
    main()
