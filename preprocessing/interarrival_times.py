"""
Interarrival times — empirical ECDFs with Monte Carlo calibration
═══════════════════════════════════════════════════════════════════════

Goal:
  - Keep the empirical distributions of the interarrival times.
  - Do not use a scaling based on the mean alone, because over finite time
    bands it can badly underestimate the rare zones.
  - Calibrate instead a multiplicative factor for every (zone, time slot) by
    simulating the same mechanism used in model.py:

        while time < slot_end:
            iat = ECDF()
            if time + iat >= slot_end:
                break
            time += iat
            count += 1

  - The factor is chosen so that the mean simulated number of arrivals is close
    to the real yearly number of the (zone, time slot) pair.

Output:
  - interarrivals/<zone>/ecdf_<slot>.csv
  - index.csv
"""

import os
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

from estimate_parameters import PROVINCE_STATIONS

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
selected_province = "RI"
SIM_DAYS           = 365
MIN_OBS            = 10

# Monte Carlo calibration
N_CALIB_REP        = 400      # raise to 800/1000 for more stability
CALIB_MAX_ITER     = 28
CALIB_TOL_PCT      = 1.0      # target: estimated mean error within ±1%
CALIB_SEED         = 12345
MIN_IAT            = 1e-3     # avoids null interarrivals and infinite loops

slot_duration   = {"00_07": 420, "07_12": 300, "12_18": 360, "18_24": 360}
slot_start_hour = {"00_07":   0, "07_12":   7, "12_18":  12, "18_24":  18}
SLOTS_ORDERED   = ["00_07", "07_12", "12_18", "18_24"]

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_time_slot(hour):
    if   0 <= hour <  7: return "00_07"
    elif 7 <= hour < 12: return "07_12"
    elif 12 <= hour < 18: return "12_18"
    else:                 return "18_24"


def make_ecdf(values):
    """Returns a sorted x and the cdf expected by the loader of model.py."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.maximum(x, MIN_IAT)
    x = np.sort(x)
    cdf = np.arange(1, len(x) + 1) / len(x)
    return x, cdf


def estimate_generated_count(x, cdf, slot_len, sim_days, factor, n_rep, seed):
    """
    Estimates how many arrivals the truncated renewal process generates in the
    time band. Same logic as model.py, without the rest of the DES.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros(n_rep, dtype=float)

    x_scaled = np.maximum(x * factor, MIN_IAT)

    for r in range(n_rep):
        total = 0
        for _ in range(sim_days):
            t = 0.0
            while t < slot_len:
                u = rng.random()
                iat = float(np.interp(u, cdf, x_scaled, left=x_scaled[0], right=x_scaled[-1]))

                # same logic as model.py: if it overruns, nothing is generated
                if t + iat >= slot_len:
                    break

                t += iat
                total += 1

                # safety guard, in theory useless, but it avoids pathological loops
                if total > sim_days * slot_len / MIN_IAT:
                    raise RuntimeError("Too many arrivals generated: possible null interarrival.")

        counts[r] = total

    return float(counts.mean()), float(counts.std(ddof=1))


def calibrate_factor_to_count(x, cdf, slot_len, target_count, sim_days, zone, slot):
    """
    Finds a multiplicative factor of the interarrivals such that the mean number
    of arrivals generated is close to the real target.

    If the factor grows, the interarrivals grow and the number of arrivals drops.
    """
    stable_seed = CALIB_SEED + zlib.crc32(f"{zone}|{slot}".encode("utf-8"))

    def eval_factor(f):
        mean_count, std_count = estimate_generated_count(
            x=x,
            cdf=cdf,
            slot_len=slot_len,
            sim_days=sim_days,
            factor=f,
            n_rep=N_CALIB_REP,
            seed=stable_seed,
        )
        err_pct = 100.0 * (mean_count - target_count) / target_count
        return mean_count, std_count, err_pct

    # diagnostics at factor=1, that is purely empirical without any scaling
    base_mean, base_std, base_err = eval_factor(1.0)

    best_factor = 1.0
    best_mean   = base_mean
    best_std    = base_std
    best_err    = base_err

    # If it is already good, barely anything is touched
    if abs(base_err) <= CALIB_TOL_PCT:
        return best_factor, best_mean, best_std, best_err, base_mean, base_std, base_err

    # Bracketing: the count decreases as the factor grows
    low, high = 0.05, 20.0
    low_mean, low_std, low_err = eval_factor(low)
    high_mean, high_std, high_err = eval_factor(high)

    # If high still generates too much, raise high
    while high_mean > target_count and high < 500:
        high *= 2.0
        high_mean, high_std, high_err = eval_factor(high)

    # If low generates too little, lower low
    while low_mean < target_count and low > 1e-4:
        low /= 2.0
        low_mean, low_std, low_err = eval_factor(low)

    candidates = [
        (1.0, base_mean, base_std, base_err),
        (low, low_mean, low_std, low_err),
        (high, high_mean, high_std, high_err),
    ]

    for _ in range(CALIB_MAX_ITER):
        mid = (low + high) / 2.0
        mid_mean, mid_std, mid_err = eval_factor(mid)
        candidates.append((mid, mid_mean, mid_std, mid_err))

        if abs(mid_err) <= CALIB_TOL_PCT:
            break

        # If it generates too many calls, the interarrivals have to grow
        if mid_mean > target_count:
            low = mid
        else:
            high = mid

    best_factor, best_mean, best_std, best_err = min(
        candidates,
        key=lambda z: abs(z[3])
    )

    return best_factor, best_mean, best_std, best_err, base_mean, base_std, base_err

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADING AND FILTERING OF THE DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_parquet(f"dataframe_{selected_province}.parquet", engine="pyarrow")
df = df[df["TIPO MEZZO"] == "ASI"].copy()
# Only the missions of the province's own stations (as in estimate_parameters.py)
df = df[df["POSTAZIONE"].isin(PROVINCE_STATIONS)].copy()

df_unique = df.drop_duplicates(subset="Numero Emergenza").copy()
df_unique["Giorno e OraChiamata"] = pd.to_datetime(df_unique["Giorno e OraChiamata"], errors="coerce")
df_unique = df_unique[df_unique["Giorno e OraChiamata"].notna()].copy()

df_unique["TimeSlot"] = df_unique["Giorno e OraChiamata"].dt.hour.apply(get_time_slot)
df_unique["Date"]     = df_unique["Giorno e OraChiamata"].dt.date

# ─────────────────────────────────────────────────────────────────────────────
# 2. OBSERVED INTERARRIVALS
# ─────────────────────────────────────────────────────────────────────────────
interarrival_dict = defaultdict(list)
n_calls_dict      = defaultdict(int)

for (zone, slot), group in df_unique.groupby(["Nucleo", "TimeSlot"]):
    start_hour = slot_start_hour[slot]

    for date, day_group in group.groupby("Date"):
        day_times = np.sort(pd.to_datetime(day_group["Giorno e OraChiamata"].values))
        n_calls_dict[(zone, slot)] += len(day_times)

        slot_start = pd.Timestamp(date) + pd.Timedelta(hours=start_hour)

        # start of the band → first call
        first_iat = (day_times[0] - slot_start).total_seconds() / 60.0
        if first_iat >= 0 and np.isfinite(first_iat):
            interarrival_dict[(zone, slot)].append(first_iat)

        # consecutive calls in the same band/day
        if len(day_times) >= 2:
            iats = np.diff(day_times) / np.timedelta64(1, "m")
            interarrival_dict[(zone, slot)].extend(iats.astype(float))

# Days with at least one call per (zone, slot)
n_days_with_calls = (
    df_unique.groupby(["Nucleo", "TimeSlot"])["Date"]
    .nunique()
    .rename("n_days")
    .reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILDING THE ECDF + MONTE CARLO CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
results = []

for (zone, slot), raw in interarrival_dict.items():
    raw = np.array(raw, dtype=float)
    raw = raw[(raw >= 0) & np.isfinite(raw)]

    if len(raw) < MIN_OBS:
        print(f"  SKIP {zone:20s} {slot}: only {len(raw)} observations")
        continue

    n_calls  = int(n_calls_dict[(zone, slot)])
    slot_len = float(slot_duration[slot])

    n_days_obs = int(n_days_with_calls.loc[
        (n_days_with_calls["Nucleo"] == zone) &
        (n_days_with_calls["TimeSlot"] == slot),
        "n_days"
    ].values[0])

    n_empty_days = max(0, SIM_DAYS - n_days_obs)

    # Empty days are represented by an IAT equal to the duration of the band.
    # The factor is then chosen on the number of arrivals generated, not on the mean.
    if n_empty_days > 0:
        raw_aug = np.concatenate([raw, np.full(n_empty_days, slot_len)])
    else:
        raw_aug = raw.copy()

    raw_aug = np.maximum(raw_aug, MIN_IAT)
    x_raw, cdf_raw = make_ecdf(raw_aug)

    mean_observed = float(raw_aug.mean())
    mean_target   = float((SIM_DAYS * slot_len) / n_calls)
    mean_scaling_factor = mean_target / mean_observed

    factor_mc, mean_calib, std_calib, err_calib, mean_base, std_base, err_base = calibrate_factor_to_count(
        x=x_raw,
        cdf=cdf_raw,
        slot_len=slot_len,
        target_count=n_calls,
        sim_days=SIM_DAYS,
        zone=zone,
        slot=slot,
    )

    # Save the calibrated empirical ECDF
    data_sorted = np.sort(x_raw * factor_mc)
    data_sorted = np.maximum(data_sorted, MIN_IAT)
    n_ecdf = len(data_sorted)
    ecdf_probs = np.arange(1, n_ecdf + 1) / n_ecdf

    out_dir = os.path.join("interarrivals", zone)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ecdf_{slot}.csv")
    pd.DataFrame({"x": data_sorted, "cdf": ecdf_probs}).to_csv(out_path, index=False)

    print(
        f"{zone:20s} {slot}  "
        f"Real={n_calls:4d}  "
        f"Base_MC={mean_base:7.1f} ({err_base:+6.2f}%)  "
        f"Factor_MC={factor_mc:7.4f}  "
        f"Calib_MC={mean_calib:7.1f} ({err_calib:+6.2f}%)  "
        f"MeanFactor_old={mean_scaling_factor:7.4f}"
    )

    results.append({
        "zone": zone,
        "time_slot": slot,
        "n_calls": n_calls,
        "n_days_obs": n_days_obs,
        "n_empty_days": n_empty_days,
        "n_ecdf_points": n_ecdf,
        "mean_observed_raw_aug": round(mean_observed, 3),
        "mean_target_old": round(mean_target, 3),
        "mean_scaling_factor_old": round(mean_scaling_factor, 6),
        "mc_factor": round(factor_mc, 6),
        "mc_base_count": round(mean_base, 3),
        "mc_base_error_pct": round(err_base, 3),
        "mc_calibrated_count": round(mean_calib, 3),
        "mc_calibrated_std": round(std_calib, 3),
        "mc_calibrated_error_pct": round(err_calib, 3),
        "ecdf_file": out_path,
    })

df_res = pd.DataFrame(results)
df_res.to_csv("index.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FINAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("FINAL DIAGNOSTICS - ECDF calibration on the number of arrivals generated")
print("=" * 78)

if len(df_res) > 0:
    print(f"Mean base error, without calibration:      {df_res['mc_base_error_pct'].mean():+.2f}%")
    print(f"Mean calibrated error:                     {df_res['mc_calibrated_error_pct'].mean():+.2f}%")
    print(f"Median calibrated error:                   {df_res['mc_calibrated_error_pct'].median():+.2f}%")
    print(f"Min/max calibrated error:                  {df_res['mc_calibrated_error_pct'].min():+.2f}% / {df_res['mc_calibrated_error_pct'].max():+.2f}%")
    print(f"N with |calibrated error| > 5%:            {(abs(df_res['mc_calibrated_error_pct']) > 5).sum()} / {len(df_res)}")

    print("\nCalibrated error by time slot:")
    print(df_res.groupby("time_slot")["mc_calibrated_error_pct"].agg(["mean", "min", "max"]).round(3).to_string())

    print("\nCalibrated error by zone:")
    print(df_res.groupby("zone")["mc_calibrated_error_pct"].agg(["mean", "min", "max"]).round(3).to_string())

print("\nECDFs saved in: interarrivals/")
print("Index saved in: index.csv")
