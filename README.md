# EMS Simulator — Rieti

A discrete-event simulator of the Emergency Medical Service (EMS) of the
province of Rieti (Lazio, Italy), together with the aggregate parameters and
the anonymised emergency instances it was estimated and calibrated on.

The simulator reproduces one year of operations of the basic ambulance fleet
(*Ambulanze di Soccorso di base*, ASI): call arrivals by zone and time band,
telephone triage, dispatching of the closest available ambulance, travel to the
scene, treatment, transport to the hospital of the appropriate clinical
network, ambulance block at the emergency department, sanitation and return to
base. It is used to compare alternative allocations of the ambulance fleet over
the stations of the province.

This repository publishes the code of the model and of the estimation
procedure. The data belong to ARES 118 (the regional EMS authority of Lazio):
the raw mission-level records are personal data and are never published, and
the parameters estimated from them (`data/`) are not included in this release
either. The `data/` section below documents their format.

## Repository layout

```
run.py                        entry point: runs any step from anywhere
simulator/                    the model and the experiments
  model.py                    discrete-event model, KPIs and reports
  scenarios.py                fleet-allocation scenarios (as-is, S1..S8)
  scenarios_short_paper.py    scenarios of the short paper (as-is, A..G)
  aggregate_scenario_tables.py  aggregates the scenario reports into tables
  paired_t_test.py            paired comparison of two scenarios
preprocessing/                from raw records to the published parameters
  estimation.py               estimators of every simulation parameter
  estimate_parameters.py      writes data/parameters/ from the raw dataset
  anonymize_instances.py      writes data/anonymized-instances/
  call_zones.py               assigns municipalities to operational zones
  travel_times.py             builds the travel-time matrices (OpenRouteService)
  interarrival_times.py       builds the interarrival ECDFs
  make_example_data.py        writes the synthetic example-data/
example-data/                 synthetic inputs with the format of data/ (invented values)
data/                         inputs of the simulator (not included in this release)
  parameters/                 aggregate parameters read by the simulator
  travel-times/               travel-time matrices, by time band
  interarrivals/              interarrival ECDFs, by zone and time band
  spatial/                    grid cells, stations, hospitals
  anonymized-instances/       anonymised emergency records
raw/                          raw dataset (not published, git-ignored)
```

## Requirements

Python 3.11 or later and the packages in `requirements.txt`:

```sh
pip install -r requirements.txt
```

The simulator itself needs `simpy`, `numpy`, `pandas`, `scipy`, `polars`,
`matplotlib`, `xlsxwriter` and `pyarrow`. The preprocessing scripts that
rebuild the travel-time matrices also need `openrouteservice`, `requests` and
`tqdm`, plus an OpenRouteService API key in the `ORS_API_KEY` environment
variable.

## Running the simulator

The simulator reads its inputs from `data/` (see the note at the top: not
included in this release). With that folder in place, nothing else has to be
prepared. `run.py` is the entry point and works from any working directory:

```sh
python run.py calibration          # 1. model vs. real data
python run.py scenarios            # 2. fleet-allocation scenarios
python run.py ttest                # 3. paired t-test of the scenarios
python run.py tables               # 4. summary tables
python run.py all                  # steps 2, 3 and 4 in a row
```

### Trying it on synthetic data

`example-data/` holds a fictitious territory with the same file formats as
`data/`, written by `preprocessing/make_example_data.py`. Every value in it is
invented (only the station names are real, because the scenarios refer to
them), so the results say nothing about the real system, but the whole pipeline
runs on it:

```sh
python run.py calibration --example-data
python run.py all --example-data
```

The reports go to `simulator/report_calibration_example/` and
`simulator/report_scenarios_example/`, next to the published ones and without
overwriting them.

### Options

Add `--short-paper` to work on the scenarios of the short paper instead of the
main ones, and pass through any option of the underlying script, for instance
`python run.py calibration --replications 5`. Reports are always written under
`simulator/`.

The steps below describe what each one does and produces. They can also be run
directly, from the `simulator/` folder:

```sh
cd simulator
```

**1. Calibration** — checks that the model reproduces the real data. Writes
exactly the validation tables and figures of the paper, nothing else: the
tables in one Excel file (one sheet each) and the figures as image files.

```sh
python model.py                    # or: python model.py --replications 5
```

`report_calibration/report_calibration.xlsx`:

| sheet | paper |
| --- | --- |
| `T1 - Potential EDs` | Table 1, the 13 emergency departments reachable from the territory |
| `T2 - Calls by zone and slot` | Table 2, historical calls by zone and time band |
| `T3 - Correction factors` | Table 3, travel-time correction factors by segment, urgency and time band |
| `T4 - Calls validation` | Table 4, historical vs. simulated number of calls by urgency |
| `T5 - Coverage urgent` | Table 5, coverage of urgent calls at every threshold |
| `T6 - Coverage non-urgent` | Table 6, coverage of non-urgent calls |
| `T7 - Calls per base` | Table 7, percentage of calls assigned at each base |

Every validation table uses the columns of the paper: `mu` (historical value),
`AVGsim`, `LBsim`, `UBsim` (mean and 95% confidence interval over the
replications) and the two absolute deviations `|LBsim-mu|`, `|UBsim-mu|`.

`report_calibration/figures/`:

| file | paper |
| --- | --- |
| `figure_06_coverage_urgent.png` | Figure 6, coverage of urgent calls, real vs. simulated |
| `figure_07_coverage_non_urgent.png` | Figure 7, same for non-urgent calls |
| `figure_08_calls_per_base.png` | Figure 8, percentage of calls per base, real vs. simulated |

`--format pdf` (or `svg`) writes vector figures instead. `--extra` also saves
the arrival diagnostics and the per-replication coverage, which are not in the
paper.

**2. Scenarios** — runs every fleet allocation, 30 replications each. Writes one
report per scenario in `report_scenarios/` (`report_as_is.xlsx`,
`report_S1.xlsx` … `report_S8.xlsx`), each with the sheets `Summary`,
`Coverage_per_replication`, `Extra_KPI`, `Extra_KPI_summary`,
`Calls_per_replication`, `MaxResponseTime_raw`, `MaxResponseTime_stats`,
`Hospitals`, `Zones`, `Stations`, `Vehicle_Utilization`, `Vehicle_Util_summary`.

```sh
python scenarios.py                  # -> report_scenarios/
python scenarios_short_paper.py      # -> report_scenarios_short_paper/
```

**3. Statistical comparison** — paired t-test of every scenario against the
as-is, replication by replication. Reads the reports of step 2 and writes, in
the same folder, `coverage_per_replication_all_scenarios.xlsx`,
`paired_ttest_vs_as_is.xlsx` and `paired_ttest_all_pairs.xlsx`.

```sh
python paired_t_test.py
python paired_t_test.py --report-dir report_scenarios_short_paper
```

**4. Summary tables** — aggregates everything into
`scenario_summary_tables.xlsx` (coverage at 20 minutes, coverage at every
threshold, vehicle utilisation overall and by station, average and worst-5%
response time, t-test tables). It needs step 3 to have been run.

```sh
python aggregate_scenario_tables.py
python aggregate_scenario_tables.py --report-dir report_scenarios_short_paper
```

Both scripts of steps 3 and 4 take the scenarios from the reports they find;
`--scenarios as_is S1 S2` restricts the comparison to a subset.

### Replications run in parallel

Replications are independent and each one is seeded separately, so spreading
them over several processes does not change a single number. It is on by
default: the simulator leaves one core free and uses at most six processes,
since each of them loads its own copy of the travel-time matrices.

```sh
python model.py --replications 30 --workers 2      # fewer processes
python model.py --replications 30 --workers 1      # serial
set EMS_WORKERS=8 && python scenarios.py           # scripts without options (Windows)
EMS_WORKERS=8 python scenarios.py                  # (Linux, macOS)
```

Spawning processes needs an importable entry point, which `python -c "..."` and
interactive sessions do not provide: there the simulator says so and falls back
to a serial run. Launch the scripts (`python model.py ...`) to get the parallel
one.

Run a single configuration from Python:

```python
import model

model.run_and_export_kpi(
    num_repl=30,
    capacities_day={"RIETI": 2, "POSTA": 1, "AMATRICE": 1},   # station -> ambulances
    capacities_night=None,                                    # None = as-is
    output_dir="report_scenario",
    label="my_scenario",
)
```

Stations can be given by name or by grid-cell code (`"RIETI"` and `"Q17_12"`
are the same station); `model.STATION_NAME` lists the correspondence.

### How a replication works

Each replication simulates 15 days of warm-up plus 365 days of operations.
Calls are generated during the warm-up too, so that the system is already
loaded when the statistics start; only the calls started after the warm-up are
counted. Calls are generated per zone and time band (`00_07`, `07_12`, `12_18`,
`18_24`) by inverse transform of the empirical interarrival distributions, and
every duration of a mission is drawn the same way.

Each emergency draws a triage colour code, the number of ambulances, the grid
cell and the urgency assessed on scene; the clinical network and whether the
mission ends with transport to hospital depend on the on-scene urgency, and so
do the durations after the arrival on scene. The ambulance is chosen right after
the triage, so it stays busy during the assignment and the preparation of the
crew: it is the closest free one within a dynamic time limit (fixed at 60
minutes for non-urgent codes; for urgent codes, the travel time of the farthest
station within 20 minutes, or of the third closest one). Otherwise the call
waits, and when an ambulance frees up the waiting calls retry in order of
urgency. Every ambulance has its own position, also when a station hosts more
than one. Each trip uses the travel-time matrix of the time band of the instant
it starts; returns to the station use the matrices without lights and sirens.
Missions can be cancelled after dispatch, after departure (the ambulance then
drives back the stretch already covered) or on arrival, with the probabilities
estimated from the data.

## Data

### `data/parameters/`

`model_parameters.json` contains the aggregate parameters of the province,
estimated on the missions of its 12 stations (the few missions carried out by
ambulances of neighbouring provinces are left out): triage colour-code
distribution, the triage → on-scene urgency transition matrix, the
clinical-network distribution and the probability of transport to hospital by
on-scene urgency (`network_probabilities_by_on_scene`,
`transport_to_hospital_pct_by_on_scene`), the hospitals of each network, number
of ambulances per emergency, probability of cancelling the mission after
dispatch / departure / arrival, probability of a block at the emergency
department by hospital, probability that an ambulance does *not* need
sanitation (`sanitation_pct`), and the distribution of the calls over the grid
cells of each zone.

The sanitation time is not estimated from these data: it is a triangular
distribution (minimum 0.05, maximum 60, mode 38.57 minutes) set in
`simulator/model.py`.

`duration_distributions.csv` contains the empirical distribution of the
duration of every phase of a mission, as a binned histogram:

| column | meaning |
| --- | --- |
| `phase` | `triage`, `assignment`, `departure`, `operating_to_hosp`, `operating`, `delivery_to_ED`, `cancelled_after_dispatch`, `cancelled_after_departure`, `cancelled_after_arrival`, `ed_block` |
| `group` | `RG`/`VB`: urgency at triage for `triage`, `assignment`, `departure`, urgency on scene for `operating_to_hosp`, `operating`, `delivery_to_ED` (deceased on scene counts as `RG`); the hospital for `ed_block`, or `ALL_HOSPITALS` for the distribution pooled over every hospital, used where a hospital has fewer than 10 blocks; empty otherwise |
| `minutes` | centre of the histogram bin |
| `probability` | probability of the bin (sums to 1 within each phase and group) |

`travel_time_correction_factors.xlsx` holds the multiplicative factors applied
to the travel-time matrices, by segment (base → scene, scene → hospital),
urgency and time band, in the columns `category`, `time_band` and
`scaling_factor`. They are an input of the preprocessing, not of the simulator,
and are published because they are Table 3 of the paper. The base → scene
factors are estimated only on the missions whose ambulance left from its
station. Base → scene factors also scale the scene → scene and hospital → scene
matrices; return trips to the station are not scaled.

### `data/travel-times/`

Travel-time matrices in minutes, one parquet file per time band and type of
trip, named `<time band>_<trip type>.parquet`. Time bands are `weekday_peak`,
`weekday_normal`, `weekday_night`, `weekend_august_normal`,
`weekend_august_night`; trip types are `base_target`, `target_base`,
`target_target`, `target_hospital`, `hospital_target` and `hospital_base`
(`base` = station, `target` = grid cell of the emergency), with the
`EMER`/`NON_EMER` suffix where the driving profile differs between urgent and
non-urgent missions. Each file holds three columns: `origin`, `destination`,
`minutes`. The times come from OpenRouteService (`driving-hgv` profile, each
direction computed separately), multiplied by the correction factors and
rounded to the minute.

### `data/interarrivals/`

`index.csv` lists, for each zone (`zone`) and time band (`time_slot`), the
number of observed emergencies served by the stations of the province
(`n_calls`) and the ECDF file (`ecdf_file`); each ECDF file holds the
interarrival time `x` in minutes and the cumulative probability `cdf`. The
empirical interarrivals are rescaled by a factor, found by Monte Carlo
simulation, so that the number of calls generated in each zone and time band
matches the observed one.

### `data/spatial/`

`stations.csv` (grid-cell code, station name, coordinates — cells without a
name are candidate locations), `cells.csv` (centroid of each grid cell) and
`hospitals.csv` (coordinates of the hospitals).

### `data/anonymized-instances/`

`emergencies.csv` contains the mission-level records of one year (2023) of ASI
missions, anonymised. Direct identifiers, the exact location of the emergency
(address, coordinates, municipality, locality) and all clinical information
(pathology, pathology class, medical assessment) have been removed; of the
timestamps only the calendar day and the hour of the call are kept, every other
instant being expressed as a duration in minutes. What remains is the
information needed to re-estimate the parameters and to reproduce the
calibration:

| field | meaning |
| --- | --- |
| `emergency_id` | sequential identifier of the emergency (several missions can share it) |
| `date`, `hour`, `time_slot` | day and hour of the call, and the time band it falls in |
| `province`, `zone`, `cell` | province, operational zone and grid cell (about 2.3 × 3.2 km) of the emergency |
| `urgency_triage`, `urgency_on_scene` | urgency at telephone triage and on scene (R, G, V, B; N = deceased) |
| `station`, `vehicle_type` | station the ambulance came from, type of vehicle |
| `hospital`, `hospital_network` | destination hospital and its clinical network |
| `mission_outcome`, `cancellation_reason` | outcome of the mission and, if cancelled, its reason |
| `last_filled_field` | last phase reached by a cancelled mission |
| `arrived_on_scene`, `ed_block` | whether the ambulance reached the scene, whether it was blocked at the ED |
| `response_time_min`, `cycle_time_min` | call → arrival on scene, call → end of mission |
| `diff_*` | duration in minutes of each phase of the mission (`diff_triage_end_call`, `diff_dispatch_triage_end`, `diff_departure_dispatch`, `diff_arrival_departure`, `diff_loading_arrival`, `diff_hospital_loading`, `diff_close_hospital`, `diff_close_block`, `diff_close_arrival`, `diff_close_cancelled_*`) |

Column names, file names and parameter keys are in English throughout. The
*values* of the categorical fields are instead kept exactly as they are recorded
by the regional EMS system, so that every row stays traceable to its source:
`mission_outcome` (`MISSIONE CONCLUSA CON IL TRASPORTO IN OSPEDALE`, `MISSIONE
ANNULLATA`, …), `cancellation_reason`, `hospital` and the clinical networks in
`hospital_network` (`Rete Trauma` = trauma network, `Rete Ictus` = stroke
network, `Sindrome Coronarica Acuta` = acute coronary syndrome, `Aritmie Gravi e
Scompenso Cardiaco` = severe arrhythmias and heart failure, `Rete Emergenza
Cardiochirurgica` = cardiac-surgery emergency network, `Rete Perinatale` =
perinatal network, `SPDC` = psychiatric ward, `Gen` = general, `Non
classificata` = unclassified). Station and zone names are proper nouns and are
likewise unchanged.

## Re-estimating the parameters from the raw data

The raw dataset is not part of this repository. The pipeline starts from the
cleaned regional export (`dataframe_clean.parquet`, with the phase durations,
the grid cell and the clinical network already attached; the cleaning step
itself is not included here):

```sh
python preprocessing/call_zones.py              # dataframe_clean.parquet -> dataframe_RI.parquet (zones)
python preprocessing/travel_times.py            # travel-time matrices and correction factors (needs ORS_API_KEY)
python preprocessing/interarrival_times.py      # -> interarrivals/
```

These three scripts read and write in the current folder. Then, with
`dataframe_RI.parquet`, `grid_RI.xlsx`, `unique_stations.csv` and
`scaling_factors.xlsx` in `raw/` (or `EMS_RAW_DATA` pointing at the dataset):

```sh
python preprocessing/estimate_parameters.py     # -> data/parameters/, data/spatial/stations.csv
python preprocessing/anonymize_instances.py     # -> data/anonymized-instances/
```

The simulator never reads the raw dataset: estimation and simulation are
separate steps, which is what makes this repository shareable.

## Reproducing the results

The simulator is deterministic given its seed: running the four steps above
with the default 30 replications reproduces exactly the reports in
`simulator/report_calibration/` and `simulator/report_scenarios/`. Splitting
the replications over several processes does not change this, because every
replication is seeded independently.

These results differ from those of arXiv:2604.24394v2, which were produced by an
earlier version of the code. The main corrections since then: the travel times
follow the time band of each trip (before, the band of the first instant was
used for the whole year); each ambulance has its own position; the ambulance is
taken right after the triage; scenario S2 keeps one H24 and one H12 ambulance
at Rieti, as described in the paper; the durations after the arrival on scene,
the transport to hospital and the clinical network depend on the on-scene
urgency; ED blocks are no longer trimmed as outliers; return trips use their own
matrix; the warm-up generates calls; waiting calls are served by urgency; the
travel-time matrices were recomputed with OpenRouteService in both directions
with one vehicle profile; only the demand served by the 12 stations of the
province is generated.

If you use this code, please cite:

> A. De Santis, S. Iannazzo, F. Ingravalle, S. Lucidi, M. Maurici, G. Riccardi,
> M. Roma, A. Vinci. *A Realistic Discrete Event Simulation model for Ambulance
> Location and Deployment within a regional Emergency Medical Service*,
> arXiv:2604.24394.

## Licence

The code is released under the MIT licence (`LICENSE`). The data are not part
of this release and are not covered by that licence.
