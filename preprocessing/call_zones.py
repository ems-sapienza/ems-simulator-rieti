"""
Assign the municipalities of the province to the operational zones ("Nuclei").

This has to be adapted for every province: the zones below are the ones of
Rieti, taken from the two health districts of the local health authority.

The column names of the raw dataset are the Italian ones of the source system
and are used verbatim.
"""

import pandas as pd

selected_province = "RI"
df = pd.read_parquet("dataframe_clean.parquet", engine="pyarrow")
df = df[df["Provincia Target"] == selected_province]

# Areas a call can come from
# District no. 1 - Rieti, Antrodoco, S. Elpidio

zone_rieti = [
    "ASCREA", "BELMONTE IN SABINA", "CANTALICE", "CASTEL DI TORA", "CITTADUCALE",
    "COLLALTO SABINO", "COLLE DI TORA", "COLLEGIOVE", "COLLI SUL VELINO",
    "CONTIGLIANO", "GRECCIO", "LABRO", "LEONESSA", "LONGONE SABINO",
    "MONTENERO SABINO", "MONTE SAN GIOVANNI IN SABINA", "MORRO REATINO", "NESPOLO",
    "PAGANICO SABINO", "POGGIO BUSTONE", "RIETI", "RIVODUTRI", "ROCCA SINIBALDA",
    "TORRICELLA IN SABINA", "TURANIA"
]

zone_antrodoco = [
    "ACCUMOLI", "AMATRICE", "ANTRODOCO", "BORBONA", "BORGO VELINO",
    "CASTEL SANT'ANGELO", "CITTAREALE", "MICIGLIANO", "POSTA"
]

zone_s_elpidio = [
    "BORGOROSE", "CONCERVIANO", "FIAMIGNANO", "MARCETELLI", "PESCOROCCHIANO",
    "PETRELLA SALTO", "VARCO SABINO"
]

# District no. 2 - Salario-Mirtense

zone_mirtense = [
    "CANTALUPO IN SABINA", "CASPERIA", "COLLEVECCHIO", "CONFIGNI", "COTTANELLO",
    "FORANO", "MAGLIANO SABINA", "MOMPEO", "MONTASOLA", "MONTEBUONO",
    "MONTOPOLI DI SABINA", "POGGIO CATINO", "POGGIO MIRTETO", "ROCCANTICA",
    "SALISANO", "SELCI", "STIMIGLIANO", "TARANO", "TORRI IN SABINA", "VACONE"
]

zone_salario = [
    "CASAPROTA", "CASTELNUOVO DI FARFA", "FARA IN SABINA", "FRASSO SABINO", "MONTELEONE SABINO",
    "ORVINIO", "POGGIO MOIANO", "POGGIO NATIVO", "POGGIO SAN LORENZO",
    "POZZAGLIA SABINA", "SCANDRIGLIA", "TOFFIA"
]

# Normalise the municipality names
df["Comune Target"] = df["Comune Target"].astype("string").str.strip().str.upper()

# Municipality -> zone
zone_by_municipality = {}

for municipality in zone_rieti:
    zone_by_municipality[municipality] = "RIETI"

for municipality in zone_antrodoco:
    zone_by_municipality[municipality] = "ANTRODOCO"

for municipality in zone_s_elpidio:
    zone_by_municipality[municipality] = "S. ELPIDIO"

for municipality in zone_mirtense:
    zone_by_municipality[municipality] = "MIRTENSE"

for municipality in zone_salario:
    zone_by_municipality[municipality] = "SALARIO"

# Assign the zone
df["Nucleo"] = df["Comune Target"].map(zone_by_municipality).fillna("NON ASSEGNATO")

print(df["Nucleo"].value_counts(dropna=False))


df.to_parquet(f"dataframe_{selected_province}.parquet", engine="pyarrow", compression="snappy")
