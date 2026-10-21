# Raw data (not published)

This folder holds the raw mission-level dataset, which contains personal data
and is therefore **never committed** (see `.gitignore`). It is not needed to
run the simulator: the simulator only reads the aggregate parameters in
`data/parameters/`.

It is needed only to re-run the estimation step. Put here:

| file | content |
| --- | --- |
| `dataframe_RI.parquet` | mission-level records of the province |
| `grid_RI.xlsx` | definition of the grid cells (corners of each cell) |
| `unique_stations.csv` | code and coordinates of the existing and candidate stations |
| `scaling_factors.xlsx` | travel-time correction factors (copied to `data/parameters/`) |

then, from the root of the repository:

```sh
python preprocessing/estimate_parameters.py
python preprocessing/anonymize_instances.py
```

Alternatively, point the scripts at the files with `--raw`, `--grid`,
`--stations` or with the environment variables `EMS_RAW_DATA`, `EMS_RAW_GRID`,
`EMS_RAW_STATIONS`.
