from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data_raw" / "HAM10000"
OUTPUT_DIR = PROJECT_ROOT / "dataset_real"

METADATA_FILE = RAW_DIR / "HAM10000_metadata.csv"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {
    "akiec": "akiec",
    "bcc": "bcc",
    "bkl": "bkl",
    "df": "df",
    "mel": "mel",
    "nv": "nv",
    "vasc": "vasc",
}


# ============================================================
# CHECK METADATA
# ============================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )


print("=" * 60)
print("DermaAgent - REAL HAM10000 DATASET SPLIT")
print("=" * 60)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_FILE)

print(f"\nTotal metadata rows: {len(df)}")

required_columns = [
    "lesion_id",
    "image_id",
    "dx",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# CHECK IMAGE FILES
# ============================================================

image_files = list(RAW_DIR.rglob("*.jpg"))

image_map = {
    image.stem: image
    for image in image_files
}

print(f"Total JPG files found: {len(image_files)}")


# ============================================================
# REMOVE MISSING IMAGES
# ============================================================

df["image_path"] = df["image_id"].map(image_map)

missing_images = df["image_path"].isna().sum()

if missing_images > 0:
    print(
        f"\nWARNING: {missing_images} metadata rows "
        "do not have corresponding JPG files."
    )

    df = df.dropna(subset=["image_path"]).copy()


print(f"Usable images: {len(df)}")


# ============================================================
# LESION-LEVEL GROUPING
# ============================================================

lesion_df = (
    df[["lesion_id", "dx"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Unique lesions: {len(lesion_df)}")


# ============================================================
# TRAIN / TEMP SPLIT
# ============================================================

train_lesions, temp_lesions = train_test_split(
    lesion_df,
    test_size=(VAL_SIZE + TEST_SIZE),
    stratify=lesion_df["dx"],
    random_state=RANDOM_STATE,
)


# ============================================================
# VALIDATION / TEST SPLIT
# ============================================================

relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

val_lesions, test_lesions = train_test_split(
    temp_lesions,
    test_size=relative_test_size,
    stratify=temp_lesions["dx"],
    random_state=RANDOM_STATE,
)


# ============================================================
# CREATE LESION SETS
# ============================================================

train_set = set(train_lesions["lesion_id"])
val_set = set(val_lesions["lesion_id"])
test_set = set(test_lesions["lesion_id"])


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
val_test_overlap = val_set & test_set

print("\n===== LEAKAGE CHECK =====")

print("Train AND Validation:", len(train_val_overlap))
print("Train AND Test      :", len(train_test_overlap))
print("Validation AND Test :", len(val_test_overlap))

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "DATA LEAKAGE DETECTED!"
    )

print("No lesion-level overlap detected.")


# ============================================================
# ASSIGN SPLIT
# ============================================================

def get_split(lesion_id):

    if lesion_id in train_set:
        return "train"

    if lesion_id in val_set:
        return "val"

    if lesion_id in test_set:
        return "test"

    raise ValueError(
        f"Unknown lesion_id: {lesion_id}"
    )


df["split"] = df["lesion_id"].apply(get_split)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    for class_name in CLASS_MAPPING.values():

        directory = (
            OUTPUT_DIR
            / split
            / class_name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY IMAGES
# ============================================================

print("\n===== COPYING IMAGES =====")

copied = 0

for _, row in df.iterrows():

    source = Path(row["image_path"])

    split = row["split"]

    class_name = CLASS_MAPPING[row["dx"]]

    destination = (
        OUTPUT_DIR
        / split
        / class_name
        / f"{row['image_id']}.jpg"
    )

    if not destination.exists():

        shutil.copy2(
            source,
            destination
        )

        copied += 1

        if copied % 500 == 0:
            print(
                f"Copied {copied} images..."
            )


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET DISTRIBUTION")
print("=" * 60)

for split in ["train", "val", "test"]:

    split_df = df[df["split"] == split]

    print(f"\n{split.upper()}")
    print("-" * 30)

    print(
        split_df["dx"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Total images: {len(split_df)}"
    )

    print(
        f"Unique lesions: "
        f"{split_df['lesion_id'].nunique()}"
    )


# ============================================================
# SAVE SPLIT METADATA
# ============================================================

split_metadata = df[
    [
        "lesion_id",
        "image_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
        "split",
    ]
]

split_metadata.to_csv(
    OUTPUT_DIR / "metadata_split.csv",
    index=False
)


print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)

print(
    f"Images copied: {copied}"
)

print(
    f"Metadata saved to: "
    f"{OUTPUT_DIR / 'metadata_split.csv'}"
)

print(
    f"Dataset location: {OUTPUT_DIR}"
)
