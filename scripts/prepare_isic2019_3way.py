import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_raw" / "ISIC2019"

GT = DATA / "ISIC_2019_Training_GroundTruth.csv"
META = DATA / "ISIC_2019_Training_Metadata.csv"

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def main():
    print("Loading ISIC 2019 CSV files...")

    df_gt = pd.read_csv(GT)
    df_meta = pd.read_csv(META)

    df = df_gt.merge(df_meta, on="image", how="inner")

    if df["image"].duplicated().any():
        raise ValueError("Duplicate image IDs found. Stop and inspect the CSVs.")

    # Normalize lesion IDs. Missing IDs get a unique image-based group.
    lesion = df["lesion_id"].astype("string").str.strip()
    lesion = lesion.replace("", pd.NA)
    df["group_id"] = lesion.fillna(
        "IMAGE_" + df["image"].astype(str)
    )

    # Convert one-hot labels into a single target class.
    label_values = df[CLASSES].fillna(0).to_numpy()
    positive_counts = (label_values == 1).sum(axis=1)

    if not (positive_counts == 1).all():
        raise ValueError(
            "Some images do not have exactly one positive class. "
            "Inspect GroundTruth before splitting."
        )

    df["target_class"] = [
        CLASSES[i] for i in label_values.argmax(axis=1)
    ]

    df = df[df["target_class"].isin(CLASSES)].copy()
    df = df.reset_index(drop=True)

    print(f"Valid images: {len(df)}")
    print(f"Unique lesion/image groups: {df['group_id'].nunique()}")

    # First reserve approximately 10% for the final test set.
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.10, random_state=42
    )
    trainval_idx, test_idx = next(
        first_split.split(df, groups=df["group_id"])
    )

    trainval = df.iloc[trainval_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    # Reserve approximately 10% of the full dataset for validation.
    # 1/9 of the remaining 90% is approximately 10% overall.
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=1 / 9, random_state=43
    )
    train_idx, val_idx = next(
        second_split.split(trainval, groups=trainval["group_id"])
    )

    train = trainval.iloc[train_idx].copy().reset_index(drop=True)
    val = trainval.iloc[val_idx].copy().reset_index(drop=True)

    splits = {"train": train, "val": val, "test": test}

    # Strict image and group leakage checks.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        images_a = set(splits[a]["image"])
        images_b = set(splits[b]["image"])
        groups_a = set(splits[a]["group_id"])
        groups_b = set(splits[b]["group_id"])

        image_overlap = images_a & images_b
        group_overlap = groups_a & groups_b

        print(
            f"{a} vs {b}: image overlap={len(image_overlap)}, "
            f"group overlap={len(group_overlap)}"
        )

        if image_overlap or group_overlap:
            raise RuntimeError("Data leakage detected. Manifests not saved.")

    columns = [
        "image", "target_class", "group_id",
        "age_approx", "anatom_site_general", "sex"
    ] + CLASSES
    columns = [c for c in columns if c in df.columns]

    # Use new filenames; preserve existing manifests untouched.
    for name, frame in splits.items():
        path = DATA / f"isic2019_{name}_split_v2.csv"
        frame[columns].to_csv(path, index=False)
        print(f"Saved: {path.name} ({len(frame)} images)")

    for name, frame in splits.items():
        print(f"\n{name.upper()} class counts:")
        print(frame["target_class"].value_counts().sort_index().to_string())

    print("\nSUCCESS: Three split manifests created.")
    print("Check class counts and loader compatibility before training.")


if __name__ == "__main__":
    main()
