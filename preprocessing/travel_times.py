#%%
# To change the province under consideration, edit selected_province.
#
# The column names of the raw dataset are the Italian ones of the source system
# and are used verbatim; everything this script produces is in English.
#
# The OpenRouteService API key is read from the environment:
#     set ORS_API_KEY=<your key>          (Windows)
#     export ORS_API_KEY=<your key>       (Linux, macOS)

selected_province = "RI"

# Import
import pandas as pd
import openrouteservice
import time
import numpy as np
from tqdm import tqdm
import requests
from scipy.optimize import minimize_scalar
import os
import re

# Name of the travel-time column in every table produced here
MINUTES = "minutes"

df = pd.read_parquet(f"dataframe_{selected_province}.parquet", engine="pyarrow")
df = df[df["TIPO MEZZO"] == "ASI" ]
grid = pd.read_excel(f"grid_{selected_province}.xlsx")

# Keep the rows that have coordinates
df_cells_valid = df.dropna(subset=['LATITUDINE Target', 'LONGITUDINE Target']).copy()

# Make sure the coordinates in df are floats
df_cells_valid['LATITUDINE Target'] = pd.to_numeric(df_cells_valid['LATITUDINE Target'], errors='coerce')
df_cells_valid['LONGITUDINE Target'] = pd.to_numeric(df_cells_valid['LONGITUDINE Target'], errors='coerce')
df_cells_valid = df_cells_valid.dropna(subset=['LATITUDINE Target', 'LONGITUDINE Target'])

# ========== 2) Clean the grid (decimal commas) and compute the centres ==========
def to_float(col):
    # converts "42,345" -> 42.345
    return col.astype(str).str.replace(',', '.', regex=False).astype(float)

for c in ['top_left_lat','top_left_lon','bottom_right_lat','bottom_right_lon']:
    grid[c] = to_float(grid[c])

grid['center_lat'] = (grid['top_left_lat'] + grid['bottom_right_lat']) / 2.0
grid['center_lon'] = (grid['top_left_lon'] + grid['bottom_right_lon']) / 2.0

# ========== 3) Keep only the cells present in df (inner join) ==========
# In df the column is called 'Quadrato'; in the grid it is 'name'
df_for_merge = df_cells_valid[['Quadrato','LATITUDINE Target','LONGITUDINE Target']].copy()
merged = df_for_merge.merge(
    grid[['name','center_lat','center_lon']],
    left_on='Quadrato', right_on='name', how='inner'
)

# ========== 4) Haversine distance (vectorised) and choice of the closest point ==========
R = 6371.0088

lat1 = np.deg2rad(merged['LATITUDINE Target'].astype(float).to_numpy())
lon1 = np.deg2rad(merged['LONGITUDINE Target'].astype(float).to_numpy())
lat2 = np.deg2rad(merged['center_lat'].astype(float).to_numpy())
lon2 = np.deg2rad(merged['center_lon'].astype(float).to_numpy())

dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
merged['dist_km'] = 2 * R * np.arcsin(np.sqrt(a))

# For every cell take the row with the smallest distance
idx_min = merged.groupby('Quadrato')['dist_km'].idxmin().to_numpy()

df_cells_unique = (
    merged.loc[idx_min, ['Quadrato', 'LATITUDINE Target', 'LONGITUDINE Target']]
    .sort_values('Quadrato')
    .reset_index(drop=True)
)

# Save without dist_km
df_cells_unique.to_csv('cells.csv', index=False)


#%%
# --- 0) NORMALISATION OF THE CELL NAMES ---
# Extracts "Q<col>_<row>" from strings such as "Q5_8 (foo)" or with spaces, etc.
_pat_q = re.compile(r"Q\d+_\d+")

from typing import Optional

def sanitize_cell(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).upper().strip()
    m = _pat_q.search(s)
    return m.group(0) if m else None

# Apply the normalisation to the call dataframe
if "Quadrato" not in df.columns or "Nucleo" not in df.columns:
    raise KeyError("The parquet is missing the required columns: 'Quadrato' and/or 'Nucleo'.")

df = df.copy()
df["Quadrato_norm"] = df["Quadrato"].apply(sanitize_cell)

# Normalise grid.name as well
if "name" not in grid.columns or not {"top_left_lat","top_left_lon","bottom_right_lat","bottom_right_lon"}.issubset(grid.columns):
    raise KeyError("The grid file is missing required columns: name/top_left_lat/top_left_lon/bottom_right_lat/bottom_right_lon.")

grid = grid.copy()
grid["name_norm"] = grid["name"].apply(sanitize_cell)

# Compute the centres of the cells from the grid (single, stable source of coordinates)
df_grid_centroids = grid.assign(
    CENTRE_LAT = (grid["top_left_lat"] + grid["bottom_right_lat"]) / 2.0,
    CENTRE_LON = (grid["top_left_lon"] + grid["bottom_right_lon"]) / 2.0
)[["name_norm","CENTRE_LAT","CENTRE_LON"]].dropna(subset=["name_norm"]).drop_duplicates(subset=["name_norm"])

# === Ordinary stations, by province ===
stations_by_province = {
    "FR": ["ALATRI","ALATRI DUE","ANAGNI","ARPINO","ATINA","CASSINO","CECCANO","CEPRANO","CEPRANO FERRARI","FERENTINO","FIUGGI","FROSINONE CO","ISOLA LIRI","PONTECORVO","SORA","VEROLI"],
    "LT": ["APRILIA","CASTELFORTE","CISTERNA","CORI","FONDI","FONDI MOF","FORMIA","FORMIA PENITRO","GAETA","LATINA","LATINA LESTRELLA","LATINA OSCI","LATINA SCALO","MINTURNO","PONTINIA","PONZA","PRIVERNO","SABAUDIA","SEZZE","SONNINO","TERRACINA","TERRACINA BADINO","VENTOTENE"],
    "RI": ["AMATRICE","BORGO S.PIETRO","LEONESSA","MAGLIANO","OSTERIA NUOVA","PAGANICO","PASSO CORESE","POGGIO MIRTETO","POSTA","RIETI","STIMIGLIANO SCALO","TORRI IN SABINA"],
    "RM": ["ACILIA","ALBANO","ANGUILLARA","ANZIO","ARDEA","ARSOLI","AURELIA HOSPITAL","BELTRAMELLI","BRACCIANO","BRIZIARELLI","CAMPUS","CANCELLERIA","CASAL PALOCCO","CASSIA","CESANO","CIAMPINO","CINECITTÀ","CIVITAVECCHIA","CIVITAVECCHIA PORTO","COLLEFERRO","CORTE DEI CONTI","CTO","FERONIA","FIUMICINO","FONTE NUOVA","FORI IMPERIALI","FRASCATI","FRATTOCCHIE","FREGENE","GENZANO","GUIDONIA","LA RUSTICA","LADISPOLI","LADISPOLI AUDITORIUM","LANUVIO","LARIANO","LUNGHEZZA","MARCIGLIANA","MARINA S.NICOLA","MARINO","MONTELANICO","MONTEROTONDO","MONTI TIBURTINI","MORLUPO","NETTUNO","NOMENTANO","NUOVO REGINA MARGHERITA","NUOVO SALARIO","OLEVANO","OSTIA GRASSI","PALIDORO","PALOMBARA","PARLAMENTO","PISPOLE","POMEZIA","PONTE GALERIA","POPOLO","PRENESTINO","PRIMA PORTA","REPUBBLICA","ROCCA PRIORA","ROMAGNOLI","ROMANINA","S.CAMILLO","S.EUGENIO","S.GIOVANNI","S.MARIA PIETA'","S.MARINELLA","S.SEVERA","SACROFANO","SETTECAMINI","SPINACETO","SUBIACO","SUBIACO PELLICCIA","TIBURTINO","TIVOLI","TOR BELLA MONACA","TREVISO","VALMONTONE","VELLETRI","VICOVARO","VITTOR PISANI","ZAGAROLO"],
    "VT": ["ACQUAPENDENTE","BOLSENA","CANINO","CIVITA CASTELLANA","CIVITELLA D'AGLIANO","MONTALTO CENTRO","MONTEFIASCONE","MONTEROSI","ORTE","RONCIGLIONE","TARQUINIA","TUSCANIA","VETRALLA CASSIA","VITERBO","VITERBO MAINELLA","VITERBO MAMMAGIALLA"]
}

# === 1) Ordinary stations & their cells (normalised) ===
df_stations_uni = (
    df[df['POSTAZIONE'].isin(stations_by_province[selected_province])]
    [['POSTAZIONE','LATITUDINE Postazione','LONGITUDINE Postazione']]
    .drop_duplicates(subset=['POSTAZIONE'])
    .reset_index(drop=True)
)

# Compute the cell from the grid (bbox), then normalise
def find_name(lat, lon, _grid):
    match = _grid[
        (_grid["bottom_right_lat"] <= lat) & (lat <= _grid["top_left_lat"]) &
        (_grid["top_left_lon"] <= lon) & (lon <= _grid["bottom_right_lon"])
    ]
    if not match.empty:
        return match.iloc[0]["name"]
    return None

df_stations_uni["name"] = df_stations_uni.apply(
    lambda r: find_name(r["LATITUDINE Postazione"], r["LONGITUDINE Postazione"], grid),
    axis=1
)
df_stations_uni["name_norm"] = df_stations_uni["name"].apply(sanitize_cell)

# set of the ordinary cells (normalised, without NaN)
ord_cells = set(df_stations_uni.loc[df_stations_uni["name_norm"].notna(),"name_norm"].unique())

# === 2) Non-adjacent, "king"-like selector ===
_q_pat = re.compile(r"^Q(\d+)_([0-9]+)$")
def _q_idx(name: str):
    if name is None:
        return None
    m = _q_pat.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _king_adjacent(a: str, b: str) -> bool:
    ia, ib = _q_idx(a), _q_idx(b)
    if ia is None or ib is None:
        return False
    dc = abs(ia[0]-ib[0]); dr = abs(ia[1]-ib[1])
    return max(dc, dr) == 1 and not (dc==0 and dr==0)

def top_n_non_adjacent_strict(cell_list, exclude, n):
    chosen = []
    exclude = set([q for q in exclude if q is not None])
    for q in cell_list:
        if q is None or q in exclude:
            continue
        if all(not _king_adjacent(q, s) for s in chosen):
            chosen.append(q)
            if len(chosen) == n:
                break
    return chosen

# === 3) Building 4 cells per zone (on the normalised names) ===
# === GLOBAL selection: 4 non-adjacent per zone, avoiding adjacency across zones too ===
N_PER_ZONE = 4

# 1) Normalised candidates, sorted by frequency, for every zone
candidates_per_zone = (
    df.dropna(subset=["Nucleo"])
      .groupby("Nucleo")["Quadrato_norm"]
      .apply(lambda s: s.dropna().value_counts().index.tolist())
      .to_dict()
)

# 2) Global round-robin with non-adjacency constraints
selected = {zone: [] for zone in candidates_per_zone.keys()}

def _adjacent_to_any(q, squares_iterable):
    for s in squares_iterable:
        if _king_adjacent(q, s):
            return True
    return False

ord_cells = set([q for q in ord_cells if q is not None])  # already defined above

def _invalid(q):
    # Hard exclusion: identical to an ordinary station or adjacent to ANY of them
    if q in ord_cells:
        return True
    if _adjacent_to_any(q, ord_cells):
        return True
    # Global exclusion: adjacent to ANY selected cell (of any zone)
    for lst in selected.values():
        if _adjacent_to_any(q, lst):
            return True
    return False

# Order of the zones: it can stay natural, or they can be sorted by "richness"
# (the ones with more candidates first)
zone_order = sorted(candidates_per_zone.keys(), key=lambda k: len(candidates_per_zone[k]), reverse=True)

progress = True
while progress:
    progress = False
    for zone in zone_order:
        if len(selected[zone]) >= N_PER_ZONE:
            continue
        cand_list = candidates_per_zone[zone]
        # look for the next valid candidate
        while cand_list:
            q = cand_list.pop(0)
            if q is None:
                continue
            if _invalid(q):
                continue
            selected[zone].append(q)
            progress = True
            break

# 3) Coverage check: exactly 4 per zone are wanted
counts = {n: len(v) for n, v in selected.items()}
if any(c < N_PER_ZONE for c in counts.values()):
    lacking = {n: c for n, c in counts.items() if c < N_PER_ZONE}
    raise ValueError(f"Cannot assign 4 non-adjacent cells to every zone under the global constraints.\nAssignments: {counts}\nIncomplete zones: {lacking}")

# 4) Build the final dataframe of the selections
rows = [(n, q) for n, lst in selected.items() for q in lst]
df_top_cells_per_zone = pd.DataFrame(rows, columns=["Nucleo", "Quadrato_norm"])


# Check: does every zone have 4 selections?
counts = df_top_cells_per_zone.groupby("Nucleo").size()
if not (counts == N_PER_ZONE).all():
    missing = counts[counts != N_PER_ZONE]
    raise ValueError(f"Some zones do not have 4 valid cells: \n{missing}")

# === Coordinates: use the centroids of the grid (normalised names) ===
_pat_q = re.compile(r"Q\d+_\d+")
def sanitize_cell(x):
    if pd.isna(x): return None
    m = _pat_q.search(str(x).upper().strip())
    return m.group(0) if m else None

grid = grid.copy()
grid["name_norm"] = grid["name"].apply(sanitize_cell)
df_grid_centroids = grid.assign(
    CENTRE_LAT = (grid["top_left_lat"] + grid["bottom_right_lat"]) / 2.0,
    CENTRE_LON = (grid["top_left_lon"] + grid["bottom_right_lon"]) / 2.0
)[["name_norm","CENTRE_LAT","CENTRE_LON"]].dropna(subset=["name_norm"]).drop_duplicates("name_norm")

# Map the selections → centroids
df_top_cells_coords = df_top_cells_per_zone.merge(
    df_grid_centroids, left_on="Quadrato_norm", right_on="name_norm", how="left"
)

missing = df_top_cells_coords[df_top_cells_coords["CENTRE_LAT"].isna()]["Quadrato_norm"].unique()
if len(missing) > 0:
    print("[WARN] Selected cells not present in the grid:", list(missing))

df_top_cells_coords_final = df_top_cells_coords.rename(columns={
    "Quadrato_norm": "POSTAZIONE",
    "CENTRE_LAT": "LATITUDINE Postazione",
    "CENTRE_LON": "LONGITUDINE Postazione"
})[["POSTAZIONE","LATITUDINE Postazione","LONGITUDINE Postazione"]].dropna()

# --- Ordinary stations (the normalised cell is used as unique key)
df_stations_uni["name_norm"] = df_stations_uni["name"].apply(sanitize_cell)
df_stations_uni_coords = pd.DataFrame({
    "POSTAZIONE": df_stations_uni["name_norm"],
    "LATITUDINE Postazione": df_stations_uni["LATITUDINE Postazione"],
    "LONGITUDINE Postazione": df_stations_uni["LONGITUDINE Postazione"]
}).dropna(subset=["POSTAZIONE"])

# --- Final union
df_unique_stations = pd.concat(
    [df_stations_uni_coords.reset_index(drop=True),
     df_top_cells_coords_final.reset_index(drop=True)],
    ignore_index=True
).drop_duplicates(subset=["POSTAZIONE"]).reset_index(drop=True)

# --- Manual addition of the stations hosted in the fire-brigade barracks ---
df_extra_stations = pd.DataFrame([
    {
        "POSTAZIONE": "PMCASERMA",
        "LATITUDINE Postazione": 42.25702604647878,
        "LONGITUDINE Postazione": 12.640740929571185
    },
    {
        "POSTAZIONE": "PCASERMA",
        "LATITUDINE Postazione": 42.53954846065864,
        "LONGITUDINE Postazione": 13.093441015429823
    },
    {
        "POSTAZIONE": "RCASERMA",
        "LATITUDINE Postazione": 42.3962913909882,
        "LONGITUDINE Postazione": 12.859146835372918
    }
])

# Union with the stations already present
df_unique_stations = pd.concat(
    [df_unique_stations, df_extra_stations],
    ignore_index=True
).drop_duplicates(subset=["POSTAZIONE"]).reset_index(drop=True)


print(df_unique_stations)
# --- Saving
df_unique_stations.to_csv("unique_stations.csv", index=False)


#%%
# Select the needed columns and drop the duplicates
df_unique_hospitals = df[['OSPEDALE', 'LATITUDINE Ospedale', 'LONGITUDINE Ospedale']].drop_duplicates(subset=['OSPEDALE'])

# Drop the rows with NaN in the hospital coordinates and the row holding
# "ALTRA_DESTINAZIONE" in the "OSPEDALE" column
df_unique_hospitals = df_unique_hospitals[df_unique_hospitals['OSPEDALE'] != 'ALTRA_DESTINAZIONE']
# Drop any other row with a NaN in any column
df_unique_hospitals = df_unique_hospitals.dropna()
df_unique_hospitals.to_csv("hospitals.csv", index=False)

df = df[df["TIPO MEZZO"] == "ASI"]

# # Distances with OpenRouteService
# With no real traffic data, and since the times given by Google Maps cannot be
# considered reliable (an ambulance with its siren on takes markedly less), the
# procedure is the following.
# 1. Compute the times between target-target, hospital-station, station-target,
#    target-station, target-hospital, hospital-target with OpenRouteService; for
#    the zones the central coordinates are used
# 2. Compare the computed times with the real ones available, identifying a
#    traffic index for every time band
#
# Traffic normally behaves like this:
# - weekdays: peak hours (7-9, 17-19), night hours (23-6) and normal hours
#   (6-7, 9-17, 19-23)
# - weekend and August: night hours (23-6) and normal hours (6-23)
#
# !!! Some hospitals have no coordinates: they are the hospitals outside the
# region, which are not considered in the study.

# ### Case target-target
#%%
# OpenRouteService API key, read from the environment
API_KEY = os.environ.get("ORS_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the ORS_API_KEY environment variable with your OpenRouteService key.")

# Initialise the OpenRouteService client
client = openrouteservice.Client(key=API_KEY)

# Extract the data
locations = df_cells_unique[['Quadrato', 'LATITUDINE Target', 'LONGITUDINE Target']].values

# Convert the coordinates into the required format (longitude, latitude)
coords = [(lon, lat) for _, lat, lon in locations]
n = len(coords)

# Matrix holding the results
travel_data = []

# Split the points into batches, respecting the API limits (40x40 per request)
batch_size = 40

# Total number of batches, for tqdm
total_batches = ((n + batch_size - 1) // batch_size) ** 2

# Every ordered pair is computed: the road network is not symmetric, so the
# time from A to B is not reused for B to A.
with tqdm(total=total_batches, desc="Computing travel times") as pbar:
    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            sub_coords_orig = coords[i:min(i+batch_size, n)]
            sub_coords_dest = coords[j:min(j+batch_size, n)]

            try:
                # API call computing the distance matrix
                matrix = client.distance_matrix(
                    locations=sub_coords_orig + sub_coords_dest,  # origins and destinations
                    profile="driving-hgv",
                    metrics=["duration"],
                    sources=list(range(len(sub_coords_orig))),  # origin indices
                    # the destinations follow the origins in `locations`
                    destinations=list(range(len(sub_coords_orig), len(sub_coords_orig) + len(sub_coords_dest)))
                )

                # Extract the matrix of the durations
                durations = np.array(matrix["durations"])

                # Store the data
                for idx1, src in enumerate(range(i, min(i+batch_size, n))):
                    for idx2, dst in enumerate(range(j, min(j+batch_size, n))):
                        if src != dst:  # the diagonal is added below
                            travel_data.append([
                                locations[src][0], locations[src][1], locations[src][2],  # cell 1
                                locations[dst][0], locations[dst][1], locations[dst][2],  # cell 2
                                durations[idx1][idx2]  # travel time
                            ])

                # Respect the API limit (40 requests per minute)
                time.sleep(1.5)

            except Exception as e:
                print(f"Error on batch {i}-{j}: {e}")

            # Update the progress bar
            pbar.update(1)

# Build the final dataframe
df_travel = pd.DataFrame(travel_data, columns=[
    "Cell 1", "LATITUDINE Target 1", "LONGITUDINE Target 1",
    "Cell 2", "LATITUDINE Target 2", "LONGITUDINE Target 2",
    MINUTES
])

# ✅ Convert the seconds into minutes
df_travel[MINUTES] = df_travel[MINUTES] / 60

# Extract the needed columns
target_target_1 = df_travel[['Cell 1', 'Cell 2', MINUTES]]

# New dataframe with cell 1 equal to cell 2 and a travel time of 0
diagonal = pd.DataFrame({
    'Cell 1': df_cells_unique['Quadrato'].values
})
diagonal['Cell 2'] = diagonal['Cell 1']    # cell 2 equals cell 1
diagonal[MINUTES] = 0                      # travel time set to 0

# Concatenate the new table with the existing one
target_target = pd.concat([target_target_1, diagonal], ignore_index=True)

# Export the table to a CSV file
target_target.to_csv('travel_times_target_target.csv', index=False)

print("\n✅ Table exported successfully to 'travel_times_target_target.csv'!")
#%% md
# ### Case hospital-station
#%%
# 1. Extract the coordinates
hospital_coords = df_unique_hospitals[['LONGITUDINE Ospedale', 'LATITUDINE Ospedale']].values.tolist()
station_coords = df_unique_stations[['LONGITUDINE Postazione', 'LATITUDINE Postazione']].values.tolist()

# Mapping from coordinates to names
hospital_map = {
    (row['LONGITUDINE Ospedale'], row['LATITUDINE Ospedale']): row['OSPEDALE']
    for _, row in df_unique_hospitals.iterrows()
}

station_map = {
    (row['LONGITUDINE Postazione'], row['LATITUDINE Postazione']): row['POSTAZIONE']
    for _, row in df_unique_stations.iterrows()
}

# 2. API parameters
headers = {
    'Accept': 'application/json',
    'Authorization': API_KEY,
    'Content-Type': 'application/json; charset=utf-8'
}

# Function calling the API
def calculate_travel_time_matrix(origins, destinations, headers):
    body = {
        "locations": origins + destinations,
        "sources": list(range(len(origins))),  # hospitals
        "destinations": list(range(len(origins), len(origins) + len(destinations)))  # stations
    }

    while True:
        response = requests.post('https://api.openrouteservice.org/v2/matrix/driving-hgv', json=body, headers=headers)

        if response.status_code == 200:
            return response.json()['durations']
        elif response.status_code == 429:
            print("Rate limit! Waiting 60s...")
            time.sleep(60)
        else:
            print(f"Error {response.status_code}")
            print(response.text)
            return None

# Function splitting a list into batches
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# 3. Computing the times
results = []
batch_num = 0

for station_batch in chunk_list(station_coords, 50):
    batch_num += 1

    durations = calculate_travel_time_matrix(hospital_coords, station_batch, headers)

    if durations:
        for i, hosp_coord in enumerate(hospital_coords):
            for j, station_coord in enumerate(station_batch):
                minutes = durations[i][j] / 60

                results.append({
                    'Hospital': hospital_map[tuple(hosp_coord)],
                    'Station': station_map.get(tuple(station_coord), f"Station_{j}"),
                    MINUTES: minutes,
                    'Hospital lon': hosp_coord[0],
                    'Hospital lat': hosp_coord[1],
                    'Station lon': station_coord[0],
                    'Station lat': station_coord[1],
                })

# 4. Build the final dataframe
df_travel_times_hosp_base = pd.DataFrame(results)
df_travel_times_hosp_base[MINUTES] = df_travel_times_hosp_base[MINUTES].round(2)
hospital_base = df_travel_times_hosp_base[['Hospital','Station',MINUTES]]

# Save the file
hospital_base.to_csv("travel_times_hospital_base.csv", index=False)


#%% md
# ### Case station-target
#%%
# === 1. FUNCTION CALLING THE MATRIX V2 API ===
def calculate_travel_time_matrix(station_coords, cell_coords, headers):
    body = {
        "locations": station_coords + cell_coords,
        "sources": list(range(len(station_coords))),  # station indices
        "destinations": list(range(len(station_coords), len(station_coords) + len(cell_coords)))
    }

    while True:
        response = requests.post(
            'https://api.openrouteservice.org/v2/matrix/driving-hgv',
            json=body,
            headers=headers
        )
        if response.status_code == 200:
            result = response.json()
            return result['durations']
        elif response.status_code == 429:
            print("🚨 Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
        else:
            print(f"❌ Error during the computation: {response.status_code}")
            print(response.text)
            return None

# === 2. FUNCTION SPLITTING INTO BATCHES ===
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# === 3. DATA PREPARATION ===
# Coordinates [lon, lat]
station_coords = df_unique_stations[['LONGITUDINE Postazione', 'LATITUDINE Postazione']].values.tolist()
cell_coords = df_cells_unique[['LONGITUDINE Target', 'LATITUDINE Target']].values.tolist()

# API parameters
headers = {
    'Accept': 'application/json',
    'Authorization': API_KEY,
    'Content-Type': 'application/json; charset=utf-8'
}

# chunk_size that respects the limit of 3500 routes
max_routes = 3500
n_stations = len(station_coords)
chunk_size = max_routes // n_stations

def station_cell_times(from_stations):
    """
    Travel times between stations and cells, in the requested direction
    (station -> cell if from_stations, else cell -> station): the two
    directions are computed separately, the road network is not symmetric.
    """
    results = []
    # === 4. LOOP OVER THE BATCHES ===
    for cell_batch in chunk_list(cell_coords, chunk_size):
        if from_stations:
            durations_matrix = calculate_travel_time_matrix(station_coords, cell_batch, headers)
        else:
            durations_matrix = calculate_travel_time_matrix(cell_batch, station_coords, headers)
        if not durations_matrix:
            continue

        for i, station in enumerate(station_coords):
            for j, cell in enumerate(cell_batch):
                seconds = durations_matrix[i][j] if from_stations else durations_matrix[j][i]
                results.append({
                    'Station lon': station[0],
                    'Station lat': station[1],
                    'Cell lon': cell[0],
                    'Cell lat': cell[1],
                    MINUTES: seconds / 60  # in minutes
                })

    # === 5. MERGE TO GET THE STATION AND CELL NAMES ===
    times = pd.DataFrame(results).merge(
        df_unique_stations[['POSTAZIONE', 'LATITUDINE Postazione', 'LONGITUDINE Postazione']],
        left_on=['Station lat', 'Station lon'],
        right_on=['LATITUDINE Postazione', 'LONGITUDINE Postazione'],
        how='left'
    ).merge(
        df_cells_unique[['Quadrato', 'LATITUDINE Target', 'LONGITUDINE Target']],
        left_on=['Cell lat', 'Cell lon'],
        right_on=['LATITUDINE Target', 'LONGITUDINE Target'],
        how='left'
    )
    return times

# === 6. FINAL TABLE AND CSV EXPORT ===
base_target = station_cell_times(from_stations=True)[['POSTAZIONE', 'Quadrato', MINUTES]].drop_duplicates()
base_target.to_csv('travel_times_base_target.csv', index=False)
print("\n✅ Table exported to 'travel_times_base_target.csv'!")

#%% md
# ### Case target-station
#%%
target_base = station_cell_times(from_stations=False)[['Quadrato', 'POSTAZIONE', MINUTES]].drop_duplicates()
target_base.to_csv('travel_times_target_base.csv', index=False)
#%% md
# ### Case target-hospital
#%%
# Function checking whether a list holds invalid values (NaN, infinite)
def check_invalid_coords(coords):
    return any(np.isnan(coord[0]) or np.isnan(coord[1]) or np.isinf(coord[0]) or np.isinf(coord[1]) for coord in coords)

# Function computing the travel time with Matrix V2
def calculate_travel_time_matrix(hospital_coords, cell_coords, headers):
    # Check whether there are invalid coordinates
    if check_invalid_coords(hospital_coords) or check_invalid_coords(cell_coords):
        print("🚨 The coordinates hold invalid values! Check the coordinates of the hospitals and of the cells.")
        return None

    body = {
        "locations": hospital_coords + cell_coords,
        "sources": list(range(len(hospital_coords))),  # hospital indices
        "destinations": list(range(len(hospital_coords), len(hospital_coords) + len(cell_coords)))  # cell indices
    }

    while True:
        # same vehicle profile as every other matrix
        response = requests.post('https://api.openrouteservice.org/v2/matrix/driving-hgv', json=body, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result['durations']  # matrix of the travel times
        elif response.status_code == 429:
            print("🚨 Error 429: rate limit exceeded. Wait 60 seconds before retrying.")
            time.sleep(60)  # wait 60 seconds
        else:
            print(f"❌ Error during the computation: {response.status_code}")
            print(response.text)
            return None

# Split the data into batches of 50 cells at a time
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# df_unique_stations and df_cells_unique are assumed to be already defined
hospital_coords = df_unique_hospitals[['LONGITUDINE Ospedale', 'LATITUDINE Ospedale']].values.tolist()
cell_coords = df_cells_unique[['LONGITUDINE Target', 'LATITUDINE Target']].values.tolist()

# API parameters
headers = {
    'Accept': 'application/json',
    'Authorization': API_KEY,
    'Content-Type': 'application/json; charset=utf-8'
}

def hospital_cell_times(from_hospitals):
    """
    Travel times between hospitals and cells, in the requested direction
    (hospital -> cell if from_hospitals, else cell -> hospital): the two
    directions are computed separately, the road network is not symmetric.
    """
    results = []
    # Process the data in batches of 50 cells at a time
    for cell_batch in chunk_list(cell_coords, 50):
        if from_hospitals:
            durations_matrix = calculate_travel_time_matrix(hospital_coords, cell_batch, headers)
        else:
            durations_matrix = calculate_travel_time_matrix(cell_batch, hospital_coords, headers)
        if not durations_matrix:
            continue

        for i, hospital in enumerate(hospital_coords):
            for j, cell in enumerate(cell_batch):
                seconds = durations_matrix[i][j] if from_hospitals else durations_matrix[j][i]
                results.append({
                    'Hospital lon': hospital[0],
                    'Hospital lat': hospital[1],
                    'Cell lon': cell[0],
                    'Cell lat': cell[1],
                    MINUTES: seconds / 60  # convert to minutes
                })

    # Merge with the unique hospitals and cells to attach the names
    times = pd.DataFrame(results).merge(
        df_unique_hospitals[['OSPEDALE', 'LATITUDINE Ospedale', 'LONGITUDINE Ospedale']],
        left_on=['Hospital lat', 'Hospital lon'],
        right_on=['LATITUDINE Ospedale', 'LONGITUDINE Ospedale'],
        how='left',
    ).merge(
        df_cells_unique[['Quadrato', 'LATITUDINE Target', 'LONGITUDINE Target']],
        left_on=['Cell lat', 'Cell lon'],
        right_on=['LATITUDINE Target', 'LONGITUDINE Target'],
        how='left',
    )
    return times

# Cell -> hospital
target_hospital = hospital_cell_times(from_hospitals=False)[['Quadrato', 'OSPEDALE', MINUTES]]
target_hospital.to_csv('travel_times_target_hospital.csv', index=False)

print("\n✅ Table exported successfully to 'travel_times_target_hospital.csv'!")
#%% md
# ### Case hospital-target
#%%
hospital_target = hospital_cell_times(from_hospitals=True)[['OSPEDALE', 'Quadrato', MINUTES]]
hospital_target.to_csv('travel_times_hospital_target.csv', index=False)
#%% md
# ## Open the files created with OpenRouteService
#%%
hospital_target = pd.read_csv('travel_times_hospital_target.csv')
target_target = pd.read_csv('travel_times_target_target.csv')
hospital_base = pd.read_csv('travel_times_hospital_base.csv')
base_target = pd.read_csv('travel_times_base_target.csv')
base_target_EMER = pd.read_csv('travel_times_base_target.csv')
base_target_NON_EMER = pd.read_csv('travel_times_base_target.csv')
target_base = pd.read_csv('travel_times_target_base.csv')
target_hospital = pd.read_csv('travel_times_target_hospital.csv')
target_hospital_EMER = pd.read_csv('travel_times_target_hospital.csv')
target_hospital_NON_EMER = pd.read_csv('travel_times_target_hospital.csv')

print(base_target)
#%% md
# # Computing the scaling factors
#%% md
# ## Real data
#%%
# Missions whose ambulance did not leave from its station: it was sent before
# it could have driven back from where its previous mission ended (hospital or
# scene). Their observed time is not a station -> cell time, so they are left
# out of the station -> target calibration.
_cell_of_station = dict(zip(df_stations_uni["POSTAZIONE"], df_stations_uni["name"]))
_return_time = {
    **dict(zip(zip(target_base["Quadrato"], target_base["POSTAZIONE"]), target_base[MINUTES])),
    **dict(zip(zip(hospital_base["Hospital"], hospital_base["Station"]), hospital_base[MINUTES])),
}
for _col in ["DATA_ORA_INVIO", "DATA_ORA_CHIUSURA", "DATA_ORA_OSPEDALE"]:
    df[_col] = pd.to_datetime(df[_col], errors="coerce")
df = df.sort_values(["CODICE MEZZO", "DATA_ORA_INVIO"])
_previous = df.groupby("CODICE MEZZO")
_previous_place = np.where(_previous["DATA_ORA_OSPEDALE"].shift().notna(),
                           _previous["OSPEDALE"].shift(), _previous["Quadrato"].shift())
_back_home = pd.Series(
    [_return_time.get((place, _cell_of_station.get(station)), np.nan)
     for place, station in zip(_previous_place, df["POSTAZIONE"])],
    index=df.index,
)
_gap = (df["DATA_ORA_INVIO"] - _previous["DATA_ORA_CHIUSURA"].shift()).dt.total_seconds() / 60
df["not_from_base"] = _gap < _back_home.fillna(0)
print(f"Missions not leaving from their station: {df['not_from_base'].mean() * 100:.1f}%")
#%% md
# ### Case station-target NON EMER
#%%
# --- 1. Conversion and initial filter ---
df["DATA_ORA_PARTENZA"] = pd.to_datetime(df["DATA_ORA_PARTENZA"])


# --- 2. Function classifying the time bands ---
def classify_time_slot(row):
    weekday = row["DATA_ORA_PARTENZA"].weekday()
    hour = row["DATA_ORA_PARTENZA"].hour
    month = row["DATA_ORA_PARTENZA"].month

    is_weekday = weekday < 5 and month != 8
    is_weekend_or_august = weekday >= 5 or month == 8

    if is_weekday:
        if (6 <= hour < 7) or (9 <= hour < 17) or (19 <= hour < 23):
            return "weekday_normal"
        elif (23 <= hour) or (hour < 6):
            return "weekday_night"
        elif (7 <= hour < 9) or (17 <= hour < 19):
            return "weekday_peak"
    elif is_weekend_or_august:
        return "weekend_august_normal" if 6 <= hour < 23 else "weekend_august_night"
    return None

# --- 3. Applying the classification and the filters ---
df["time_band"] = df.apply(classify_time_slot, axis=1)
df_filtered = df.dropna(subset=["time_band"])
df_filtered = df_filtered[df_filtered["Criticità Al triage Telefonico"].isin(["B", "V"])]
df_filtered = df_filtered[~df_filtered["not_from_base"]]

# --- 4. Building one table per time band ---
base_target_real_tables = {}

for band, df_band in df_filtered.groupby("time_band"):
    # Pivot by POSTAZIONE and Quadrato
    province_pivot = df_band.pivot_table(
        values="diff_Arrivo_Partenza",
        index="POSTAZIONE",
        columns="Quadrato",
        aggfunc="mean",
        fill_value=None
    )

    # Drop the useless columns
    province_pivot = province_pivot.drop(columns=["mean_None"], errors="ignore")
    province_pivot.columns = [col.replace("mean_", "") for col in province_pivot.columns]

    # Turn into long format
    df_long = province_pivot.reset_index().melt(
        id_vars="POSTAZIONE",
        var_name="Quadrato",
        value_name="Mean_diff_Arrivo_Partenza"
    )
    df_long = df_long.dropna(subset=["Mean_diff_Arrivo_Partenza"])

    # --- 5. Replace POSTAZIONE with the cell name ---
    df_long = df_long.merge(
        df_stations_uni[["POSTAZIONE", "name"]],
        on="POSTAZIONE",
        how="left"
    )
    df_long = df_long.drop(columns=["POSTAZIONE"])          # drop the original column
    df_long = df_long.rename(columns={"name": "POSTAZIONE"})  # rename with the cell name


    # --- 6. Final columns ---
    df_long = df_long[["POSTAZIONE", "Quadrato", "Mean_diff_Arrivo_Partenza"]]

    # Store in the dictionary
    base_target_real_tables[band] = df_long

    # Print for inspection
    print(f"\n##### Table base_target_real - Band: {band} #####")
    print(df_long.head())

#%%
# Comparison between the dictionary base_target_real_tables and base_target
scaling_factors_base_target_NON_EMER = {}

# Rename the columns if needed, so that the join works automatically
base_target_renamed = base_target.rename(columns={MINUTES: 'mean_time'})

for band, table in base_target_real_tables.items():
    print(f"Processing band: {band}")

    # Merge the table of the band with the target table by POSTAZIONE and Quadrato
    merged = pd.merge(
        table,
        base_target_renamed,
        on=['POSTAZIONE', 'Quadrato'],
        how='inner'
    )

    # Drop the rows with NaN
    merged = merged.dropna(subset=['Mean_diff_Arrivo_Partenza', 'mean_time'])

    if merged.empty:
        print(f"Warning: no valid data for {band}, skipping.")
        continue

    original_valid = merged['mean_time'].values
    target_valid = merged['Mean_diff_Arrivo_Partenza'].values
    n = len(original_valid)

    original_valid_cut = []
    target_valid_cut = []

    MAX_ABS_DIFF = np.inf  # minutes

    for i in range(len(original_valid)):
        ratio = original_valid[i] / target_valid[i]
        diff = abs(original_valid[i] - target_valid[i])
        if target_valid[i] != 0 and 0.5 < ratio < 2 and diff < MAX_ABS_DIFF:
            original_valid_cut.append(original_valid[i])
            target_valid_cut.append(target_valid[i])

    original_valid_cut = np.array(original_valid_cut)
    target_valid_cut = np.array(target_valid_cut)

    n = len(original_valid)
    valid_n = len(original_valid_cut)

    l = [o - t for o, t in zip(original_valid_cut, target_valid_cut)]
    for i in range(len(l)):
        if l[i] == max(l):
            print("max", l[i], original_valid_cut[i], target_valid_cut[i])
        if l[i] == min(l):
            print("min", l[i], original_valid_cut[i], target_valid_cut[i])

    print("Excluded values:", n - valid_n, "out of", n)


    res = minimize_scalar(
        lambda alpha: np.sum((abs(original_valid_cut * alpha - target_valid_cut))), method="Golden"
    )

    if not res.success:
        print(f"Optimisation failed for {band}.")
        continue

    scaling_factors_base_target_NON_EMER[band] = res.x
    print(f"Optimal scaling factor for {band}: {res.x}")
    print(f"Objective value at the optimum: {res.fun}")

#%% md
# ### Case station-target EMER
#%%
# --- 1. Conversion and initial filter ---
df["DATA_ORA_PARTENZA"] = pd.to_datetime(df["DATA_ORA_PARTENZA"])

# --- 2. Function classifying the time bands ---
def classify_time_slot(row):
    weekday = row["DATA_ORA_PARTENZA"].weekday()
    hour = row["DATA_ORA_PARTENZA"].hour
    month = row["DATA_ORA_PARTENZA"].month

    is_weekday = weekday < 5 and month != 8
    is_weekend_or_august = weekday >= 5 or month == 8

    if is_weekday:
        if (6 <= hour < 7) or (9 <= hour < 17) or (19 <= hour < 23):
            return "weekday_normal"
        elif (23 <= hour) or (hour < 6):
            return "weekday_night"
        elif (7 <= hour < 9) or (17 <= hour < 19):
            return "weekday_peak"
    elif is_weekend_or_august:
        return "weekend_august_normal" if 6 <= hour < 23 else "weekend_august_night"
    return None

# --- 3. Applying the classification and the filters ---
df["time_band"] = df.apply(classify_time_slot, axis=1)
df_filtered = df.dropna(subset=["time_band"])
df_filtered = df_filtered[df_filtered["Criticità Al triage Telefonico"].isin(["R", "G"])]
df_filtered = df_filtered[~df_filtered["not_from_base"]]

# --- 4. Building one table per time band ---
base_target_real_tables = {}

for band, df_band in df_filtered.groupby("time_band"):
    # Pivot by POSTAZIONE and Quadrato
    province_pivot = df_band.pivot_table(
        values="diff_Arrivo_Partenza",
        index="POSTAZIONE",
        columns="Quadrato",
        aggfunc="mean",
        fill_value=None
    )

    # Drop the useless columns
    province_pivot = province_pivot.drop(columns=["mean_None"], errors="ignore")
    province_pivot.columns = [col.replace("mean_", "") for col in province_pivot.columns]

    # Turn into long format
    df_long = province_pivot.reset_index().melt(
        id_vars="POSTAZIONE",
        var_name="Quadrato",
        value_name="Mean_diff_Arrivo_Partenza"
    )
    df_long = df_long.dropna(subset=["Mean_diff_Arrivo_Partenza"])

    # --- 5. Replace POSTAZIONE with the cell name ---
    df_long = df_long.merge(
        df_stations_uni[["POSTAZIONE", "name"]],
        on="POSTAZIONE",
        how="left"
    )
    df_long = df_long.drop(columns=["POSTAZIONE"])          # drop the original column
    df_long = df_long.rename(columns={"name": "POSTAZIONE"})  # use the cell name as the new column

    # --- 6. Final columns ---
    df_long = df_long[["POSTAZIONE", "Quadrato", "Mean_diff_Arrivo_Partenza"]]

    # Store in the dictionary
    base_target_real_tables[band] = df_long

    # Print for inspection
    print(f"\n##### Table base_target_real - Band: {band} #####")
    print(df_long.head())

#%%
# Comparison between the dictionary base_target_real_tables and base_target
scaling_factors_base_target_EMER = {}

# Rename the columns if needed, so that the join works automatically
base_target_renamed = base_target.rename(columns={MINUTES: 'mean_time'})

for band, table in base_target_real_tables.items():
    print(f"Processing band: {band}")

    # Merge the table of the band with the target table by POSTAZIONE and Quadrato
    merged = pd.merge(
        table,
        base_target_renamed,
        on=['POSTAZIONE', 'Quadrato'],
        how='inner'
    )

    # Drop the rows with NaN
    merged = merged.dropna(subset=['Mean_diff_Arrivo_Partenza', 'mean_time'])

    if merged.empty:
        print(f"Warning: no valid data for {band}, skipping.")
        continue

    original_valid = merged['mean_time'].values
    target_valid = merged['Mean_diff_Arrivo_Partenza'].values
    n = len(original_valid)

    original_valid_cut = []
    target_valid_cut = []

    MAX_ABS_DIFF = np.inf  # minutes

    for i in range(len(original_valid)):
        ratio = original_valid[i] / target_valid[i]
        diff = abs(original_valid[i] - target_valid[i])
        if target_valid[i] != 0 and 0.5 < ratio < 2 and diff < MAX_ABS_DIFF:
            original_valid_cut.append(original_valid[i])
            target_valid_cut.append(target_valid[i])

    original_valid_cut = np.array(original_valid_cut)
    target_valid_cut = np.array(target_valid_cut)

    n = len(original_valid)
    valid_n = len(original_valid_cut)

    l = [o - t for o, t in zip(original_valid_cut, target_valid_cut)]
    for i in range(len(l)):
        if l[i] == max(l):
            print("max", l[i], original_valid_cut[i], target_valid_cut[i])
        if l[i] == min(l):
            print("min", l[i], original_valid_cut[i], target_valid_cut[i])

    print("Excluded values:", n - valid_n, "out of", n)

    res = minimize_scalar(
        lambda alpha: np.sum((abs(original_valid_cut * alpha - target_valid_cut))), method="Golden"
    )

    if not res.success:
        print(f"Optimisation failed for {band}.")
        continue

    scaling_factors_base_target_EMER[band] = res.x
    print(f"Optimal scaling factor for {band}: {res.x}")
    print(f"Objective value at the optimum: {res.fun}")

#%% md
# ### Case target-hospital NON EMER
#%%
# 'DATA_ORA_CARICO' is assumed to be in datetime format already
df["DATA_ORA_CARICO"] = pd.to_datetime(df["DATA_ORA_CARICO"])

# Function classifying the time bands
def classify_time_slot(row):
    weekday, hour, month = row["DATA_ORA_CARICO"].weekday(), row["DATA_ORA_CARICO"].hour, row["DATA_ORA_CARICO"].month
    is_weekday = weekday < 5 and month != 8
    is_weekend_or_august = weekday >= 5 or month == 8

    if is_weekday:
        if (6 <= hour < 7) or (9 <= hour < 17) or (19 <= hour < 23):
            return "weekday_normal"
        elif (23 <= hour) or (hour < 6):
            return "weekday_night"
        elif (7 <= hour < 9) or (17 <= hour < 19):
            return "weekday_peak"
    elif is_weekend_or_august:
        return "weekend_august_normal" if 6 <= hour < 23 else "weekend_august_night"
    return None

# Apply the classification and drop the rows without a time band
df["time_band"] = df.apply(classify_time_slot, axis=1)
df_filtered = df.dropna(subset=["time_band"])
# Filter on the on-scene urgency "B" or "V" as well
df_filtered = df_filtered[df_filtered["Criticità sul Posto"].isin(["B", "V"])]


# Build the final pivot table in long format
target_hospital_real_tables = {}

for band, df_band in df_filtered.groupby("time_band"):
    # Build the pivot for every band and province
    hospital_pivot = df_band.pivot_table(
        values="diff_Ospedale_Carico", index="Quadrato", columns="OSPEDALE", aggfunc="mean", fill_value=None
    )

    # Drop any useless column and reorganise the data
    hospital_pivot = hospital_pivot.drop(columns=["mean_None"], errors="ignore")
    hospital_pivot.columns = [col.replace("mean_", "") for col in hospital_pivot.columns]

    # Turn into long format
    df_long = hospital_pivot.reset_index().melt(id_vars="Quadrato", var_name="OSPEDALE", value_name="Mean_diff_Ospedale_Carico")
    df_long = df_long.dropna(subset=["Mean_diff_Ospedale_Carico"])

    # Store in the dictionary
    target_hospital_real_tables[band] = df_long[["Quadrato", "OSPEDALE", "Mean_diff_Ospedale_Carico"]]

    # Print for every band
    print(f"\n##### Table target_hospital_real_tables - Band: {band} #####")
    print(df_long.head())
#%%
# Comparison between the dictionary target_hospital_real_tables and target_hospital
scaling_factors_target_hosp_NON_EMER = {}

# Rename the columns if needed, so that the join works automatically
target_hospital_renamed = target_hospital.rename(columns={MINUTES: 'mean_time'})

for band, table in target_hospital_real_tables.items():
    print(f"Processing band: {band}")

    # Merge the table of the band with the target table by Quadrato and OSPEDALE
    merged = pd.merge(
        table,
        target_hospital_renamed,
        on=['Quadrato', 'OSPEDALE'],
        how='inner'
    )
    # Drop the rows with NaN
    merged = merged.dropna(subset=['Mean_diff_Ospedale_Carico', 'mean_time'])

    if merged.empty:
        print(f"Warning: no valid data for {band}, skipping.")
        continue

    original_valid = merged['mean_time'].values
    target_valid = merged['Mean_diff_Ospedale_Carico'].values

    original_valid_cut = []
    target_valid_cut = []

    MAX_ABS_DIFF = np.inf  # minutes

    for i in range(len(original_valid)):
        ratio = original_valid[i] / target_valid[i]
        diff = abs(original_valid[i] - target_valid[i])
        if target_valid[i] != 0 and 0.5 < ratio < 2 and diff < MAX_ABS_DIFF:
            original_valid_cut.append(original_valid[i])
            target_valid_cut.append(target_valid[i])

    original_valid_cut = np.array(original_valid_cut)
    target_valid_cut = np.array(target_valid_cut)
    n = len(original_valid)
    valid_n = len(original_valid_cut)

    l = [o - t for o, t in zip(original_valid_cut, target_valid_cut)]
    for i in range(len(l)):
        if l[i] == max(l):
            print("max", l[i], original_valid_cut[i], target_valid_cut[i])
        if l[i] == min(l):
            print("min", l[i], original_valid_cut[i], target_valid_cut[i])

    print("Excluded values:", n - valid_n, "out of", n)

    res = minimize_scalar(
        lambda alpha: np.sum((abs(original_valid_cut * alpha - target_valid_cut))), method="Golden"
    )

    if not res.success:
        print(f"Optimisation failed for {band}.")
        continue

    scaling_factors_target_hosp_NON_EMER[band] = res.x
    print(f"Optimal scaling factor for {band}: {res.x}")
    print(f"Objective value at the optimum: {res.fun}")
#%% md
# ### Case target-hospital EMER
#%%
# 'DATA_ORA_CARICO' is assumed to be in datetime format already
df["DATA_ORA_CARICO"] = pd.to_datetime(df["DATA_ORA_CARICO"])

# Function classifying the time bands
def classify_time_slot(row):
    weekday, hour, month = row["DATA_ORA_CARICO"].weekday(), row["DATA_ORA_CARICO"].hour, row["DATA_ORA_CARICO"].month
    is_weekday = weekday < 5 and month != 8
    is_weekend_or_august = weekday >= 5 or month == 8

    if is_weekday:
        if (6 <= hour < 7) or (9 <= hour < 17) or (19 <= hour < 23):
            return "weekday_normal"
        elif (23 <= hour) or (hour < 6):
            return "weekday_night"
        elif (7 <= hour < 9) or (17 <= hour < 19):
            return "weekday_peak"
    elif is_weekend_or_august:
        return "weekend_august_normal" if 6 <= hour < 23 else "weekend_august_night"
    return None

# Apply the classification and drop the rows without a time band
df["time_band"] = df.apply(classify_time_slot, axis=1)
df_filtered = df.dropna(subset=["time_band"])
# Filter on the on-scene urgency "R" or "G" as well
df_filtered = df_filtered[df_filtered["Criticità sul Posto"].isin(["R", "G"])]

# Build the final pivot table in long format
target_hospital_real_tables = {}

for band, df_band in df_filtered.groupby("time_band"):
    # Build the pivot for every band and province
    hospital_pivot = df_band.pivot_table(
        values="diff_Ospedale_Carico", index="Quadrato", columns="OSPEDALE", aggfunc="mean", fill_value=None
    )

    # Drop any useless column and reorganise the data
    hospital_pivot = hospital_pivot.drop(columns=["mean_None"], errors="ignore")
    hospital_pivot.columns = [col.replace("mean_", "") for col in hospital_pivot.columns]

    # Turn into long format
    df_long = hospital_pivot.reset_index().melt(id_vars="Quadrato", var_name="OSPEDALE", value_name="Mean_diff_Ospedale_Carico")
    df_long = df_long.dropna(subset=["Mean_diff_Ospedale_Carico"])

    # Store in the dictionary
    target_hospital_real_tables[band] = df_long[["Quadrato", "OSPEDALE", "Mean_diff_Ospedale_Carico"]]

    # Print for every band
    print(f"\n##### Table target_hospital_real_tables - Band: {band} #####")
    print(df_long.head())
#%%
# Comparison between the dictionary target_hospital_real_tables and target_hospital
scaling_factors_target_hosp_EMER = {}

# Rename the columns if needed, so that the join works automatically
target_hospital_renamed = target_hospital.rename(columns={MINUTES: 'mean_time'})

for band, table in target_hospital_real_tables.items():
    print(f"Processing band: {band}")

    # Merge the table of the band with the target table by Quadrato and OSPEDALE
    merged = pd.merge(
        table,
        target_hospital_renamed,
        on=['Quadrato', 'OSPEDALE'],
        how='inner'
    )
    # Drop the rows with NaN
    merged = merged.dropna(subset=['Mean_diff_Ospedale_Carico', 'mean_time'])

    if merged.empty:
        print(f"Warning: no valid data for {band}, skipping.")
        continue

    original_valid = merged['mean_time'].values
    target_valid = merged['Mean_diff_Ospedale_Carico'].values

    original_valid_cut = []
    target_valid_cut = []

    MAX_ABS_DIFF = np.inf  # minutes

    for i in range(len(original_valid)):
        ratio = original_valid[i] / target_valid[i]
        diff = abs(original_valid[i] - target_valid[i])
        if target_valid[i] != 0 and 0.5 < ratio < 2 and diff < MAX_ABS_DIFF:
            original_valid_cut.append(original_valid[i])
            target_valid_cut.append(target_valid[i])

    original_valid_cut = np.array(original_valid_cut)
    target_valid_cut = np.array(target_valid_cut)

    n = len(original_valid)
    valid_n = len(original_valid_cut)

    l = [o - t for o, t in zip(original_valid_cut, target_valid_cut)]
    for i in range(len(l)):
        if l[i] == max(l):
            print("max", l[i], original_valid_cut[i], target_valid_cut[i])
        if l[i] == min(l):
            print("min", l[i], original_valid_cut[i], target_valid_cut[i])

    print("Excluded values:", n - valid_n, "out of", n)

    def objective(alpha):
        return np.sum(abs(original_valid_cut * alpha - target_valid_cut))

    res = minimize_scalar(
        lambda alpha: np.sum((abs(original_valid_cut * alpha - target_valid_cut))), method="Golden"
    )

    if not res.success:
        print(f"Optimisation failed for {band}.")
        continue

    scaling_factors_target_hosp_EMER[band] = res.x
    print(f"Optimal scaling factor for {band}: {res.x}")
    print(f"Objective value at the optimum: {res.fun}")

print("BASE-TARGET NON EMER:",scaling_factors_base_target_NON_EMER,"TARGET-HOSPITAL NON EMER:", scaling_factors_target_hosp_NON_EMER,"BASE-TARGET EMER:",scaling_factors_base_target_EMER,"TARGET-HOSPITAL EMER:", scaling_factors_target_hosp_EMER)

factors_rows = []
for category, factors in [
    ("base_target_NON_EMER", scaling_factors_base_target_NON_EMER),
    ("target_hospital_NON_EMER", scaling_factors_target_hosp_NON_EMER),
    ("base_target_EMER", scaling_factors_base_target_EMER),
    ("target_hospital_EMER", scaling_factors_target_hosp_EMER),
]:
    for band, value in factors.items():
        factors_rows.append({
            "category": category,
            "time_band": band,
            "scaling_factor": value,
        })

df_factors = pd.DataFrame(factors_rows).sort_values(["category", "time_band"])
df_factors.to_excel("scaling_factors.xlsx", index=False)

with open("scaling_factors.txt", "w", encoding="utf-8") as f:
    f.write(f"Province: {selected_province}\n")
    current_category = None
    for _, row in df_factors.iterrows():
        category = row["category"]
        if category != current_category:
            if current_category is not None:
                f.write("\n")
            f.write(f"[{category}]\n")
            current_category = category
        f.write(f"- {row['time_band']}: {row['scaling_factor']}\n")

time_band_list = ['weekday_peak', 'weekday_normal', 'weekday_night', 'weekend_august_normal', 'weekend_august_night']


def create_scaled_tables_for_time_band(
    time_band,
    base_target_NON_EMER,
    target_hospital_NON_EMER,
    base_target_EMER,
    target_hospital_EMER,
    target_base,
    hospital_base,
    target_target,
    scaling_factors_base_target_NON_EMER,
    scaling_factors_target_hosp_NON_EMER,
    scaling_factors_base_target_EMER,
    scaling_factors_target_hosp_EMER
):
    # Safe copies
    base_target_scaled_NON_EMER = base_target_NON_EMER.copy()
    target_hospital_scaled_NON_EMER = target_hospital_NON_EMER.copy()
    base_target_scaled_EMER = base_target_EMER.copy()
    target_hospital_scaled_EMER = target_hospital_EMER.copy()
    target_base_scaled = target_base.copy()
    hospital_base_scaled = hospital_base.copy()
    target_target_scaled_NON_EMER = target_target.copy()
    target_target_scaled_EMER = target_target.copy()
    hospital_target_scaled_NON_EMER = hospital_target.copy()
    hospital_target_scaled_EMER = hospital_target.copy()

    # Scaling
    base_target_scaled_NON_EMER[MINUTES] *= scaling_factors_base_target_NON_EMER[time_band]
    target_hospital_scaled_NON_EMER[MINUTES] *= scaling_factors_target_hosp_NON_EMER[time_band]
    base_target_scaled_EMER[MINUTES] *= scaling_factors_base_target_EMER[time_band]
    target_hospital_scaled_EMER[MINUTES] *= scaling_factors_target_hosp_EMER[time_band]
    target_target_scaled_NON_EMER[MINUTES] *= scaling_factors_base_target_NON_EMER[time_band]
    hospital_target_scaled_NON_EMER[MINUTES] *= scaling_factors_base_target_NON_EMER[time_band]
    target_target_scaled_EMER[MINUTES] *= scaling_factors_base_target_EMER[time_band]
    hospital_target_scaled_EMER[MINUTES] *= scaling_factors_base_target_EMER[time_band]

    # Rounding
    for table in [
        base_target_scaled_NON_EMER, target_hospital_scaled_NON_EMER,
        base_target_scaled_EMER, target_hospital_scaled_EMER,
        target_base_scaled, hospital_base_scaled,
        target_target_scaled_NON_EMER, target_target_scaled_EMER,
        hospital_target_scaled_NON_EMER, hospital_target_scaled_EMER
    ]:
        table[MINUTES] = table[MINUTES].round(0)

    # Return every scaled dataframe as a dictionary
    return {
        'base_target_NON_EMER': base_target_scaled_NON_EMER,
        'target_hospital_NON_EMER': target_hospital_scaled_NON_EMER,
        'base_target_EMER': base_target_scaled_EMER,
        'target_hospital_EMER': target_hospital_scaled_EMER,
        'target_base': target_base_scaled,
        'hospital_base': hospital_base_scaled,
        'target_target_NON_EMER': target_target_scaled_NON_EMER,
        'target_target_EMER': target_target_scaled_EMER,
        'hospital_target_NON_EMER': hospital_target_scaled_NON_EMER,
        'hospital_target_EMER': hospital_target_scaled_EMER
    }


def with_standard_columns(table):
    """The published matrices always hold (origin, destination, minutes)."""
    table = table.copy()
    table.columns = ["origin", "destination", MINUTES]
    return table

#%%
for time_band in time_band_list:
    filename = f"scaled_tables_{time_band}.xlsx"

    # Get the dictionary of the scaled dataframes
    scaled_tables_dict = create_scaled_tables_for_time_band(
        time_band,
        base_target_NON_EMER,
        target_hospital_NON_EMER,
        base_target_EMER,
        target_hospital_EMER,
        target_base,
        hospital_base,
        target_target,
        scaling_factors_base_target_NON_EMER,
        scaling_factors_target_hosp_NON_EMER,
        scaling_factors_base_target_EMER,
        scaling_factors_target_hosp_EMER
    )

    # Save to Excel, one sheet per table
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, table in scaled_tables_dict.items():
            # Make sure the sheet name does not exceed 31 characters (Excel limit)
            safe_sheet_name = sheet_name[:31]
            with_standard_columns(table).to_excel(writer, sheet_name=safe_sheet_name, index=False)

print("Excel files with separate sheets created successfully.")
#%%
# Output directory
output_dir = "scaled_tables_parquet"
os.makedirs(output_dir, exist_ok=True)

for time_band in time_band_list:
    # Get the dictionary of the scaled dataframes
    scaled_tables_dict = create_scaled_tables_for_time_band(
        time_band,
        base_target_NON_EMER,
        target_hospital_NON_EMER,
        base_target_EMER,
        target_hospital_EMER,
        target_base,
        hospital_base,
        target_target,
        scaling_factors_base_target_NON_EMER,
        scaling_factors_target_hosp_NON_EMER,
        scaling_factors_base_target_EMER,
        scaling_factors_target_hosp_EMER
    )

    # Save in parquet format (one file per table)
    for sheet_name, table in scaled_tables_dict.items():
        parquet_filename = f"{time_band}_{sheet_name}.parquet"
        parquet_path = os.path.join(output_dir, parquet_filename)

        with_standard_columns(table).to_parquet(parquet_path, index=False)

print(".parquet files created successfully.")
