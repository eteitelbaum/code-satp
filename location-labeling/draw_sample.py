"""Draw the incidents to label for the location answer set.

Reproduces the temporal split used by the location extraction notebooks
(models/location-models/location_extraction_seq2seq.ipynb: location_info_augmented.csv
sorted by date, 80/10/10), keeps validation and test incidents that name a place
below the district, and puts each split in a random labeling order. Label in that
order and stop at the target size (about 150 validation and 300 test).

Run from the repository root after build_lookup.R:  python location-labeling/draw_sample.py

Output: location-labeling/data/sample.csv
"""

import pandas as pd

SEED = 2026
OUT = "location-labeling/data/sample.csv"

# --- temporal split, as in the notebooks ---
data = pd.read_csv("data/location_info_augmented.csv", dtype=str)
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.sort_values("date").reset_index(drop=True)
train_end, val_end = int(len(data) * 0.8), int(len(data) * 0.9)
data["split"] = "train"
data.loc[train_end:val_end - 1, "split"] = "validation"
data.loc[val_end:, "split"] = "test"

# --- incidents that name a place below the district ---
def named(col):
    return data[col].fillna("").str.strip() != ""

sample = data[data["split"].isin(["validation", "test"]) & (named("village_name") | named("other_locations"))].copy()

# random labeling order within each split
sample["order"] = (
    sample.groupby("split")["incident_number"]
    .transform(lambda s: s.sample(frac=1, random_state=SEED).rank(method="first"))
    .astype(int)
)

# --- 2011 district and the hand-entered pin's SHRUG place ---
crosswalk = pd.read_csv("location-labeling/data/district_crosswalk.csv", dtype=str)
sample = sample.merge(
    crosswalk[["state", "district", "state_id", "district_id_2011", "district_name_2011", "review"]],
    on=["state", "district"], how="left",
)

pins = pd.read_parquet("location-labeling/data/pin_places.parquet").dropna(subset=["incident_number"])
pins["incident_number"] = pins["incident_number"].astype("int64").astype(str)
sample = sample.merge(
    pins[["incident_number", "pin_state_id", "pin_district_id", "pin_subdistrict_id", "pin_place_id", "pin_place_name"]],
    on="incident_number", how="left",
)

# --- link to SATP's district-month page, e.g. .../india-maoistinsurgency-odisha-rayagada-jan-2016 ---
def satp_url(row):
    state = row["state"].lower().replace(" ", "")
    district = row["district"].lower()
    return (
        "https://www.satp.org/terrorist-activity/india-maoistinsurgency-"
        f"{state}-{district}-{row['date'].strftime('%b').lower()}-{row['date'].year}"
    )

sample["satp_url"] = sample.apply(satp_url, axis=1)

cols = [
    "split", "order", "incident_number", "date", "state", "district", "state_id",
    "district_id_2011", "district_name_2011", "review", "village_name", "other_locations",
    "incident_summary", "latitude", "longitude", "pin_state_id", "pin_district_id",
    "pin_subdistrict_id", "pin_place_id", "pin_place_name", "satp_url",
]
sample = sample[cols].sort_values(["split", "order"])
sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
sample.to_csv(OUT, index=False)

print(sample.groupby("split").size().to_string())
print(f"no 2011 district: {sample['district_id_2011'].isna().sum()}; crosswalk flagged for review: {(sample['review'] == 'TRUE').sum()}")
print(f"pin inside a SHRUG place: {sample['pin_place_id'].notna().sum()} of {len(sample)}")
