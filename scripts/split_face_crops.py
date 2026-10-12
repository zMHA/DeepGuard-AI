
import pandas as pd
from pathlib import Path

MANIFEST = Path("data/processed/face_crops_manifest.csv")
OUTPUT_DIR = Path("data/processed")

VALIDATION_GROUPS = {"source_group_866_878"}

def main():
    df = pd.read_csv(MANIFEST)

    required = {"image_path", "label", "video_id", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    if df["group_id"].isna().any() or (df["group_id"].astype(str).str.strip() == "").any():
        raise ValueError("Every crop must have a non-empty group_id.")

    train = df[~df["group_id"].isin(VALIDATION_GROUPS)].copy()
    val = df[df["group_id"].isin(VALIDATION_GROUPS)].copy()

    if train.empty or val.empty:
        raise ValueError("Training or validation split is empty.")

    overlap = set(train["group_id"]) & set(val["group_id"])
    if overlap:
        raise ValueError(f"Group leakage detected: {sorted(overlap)}")

    train.to_csv(OUTPUT_DIR / "face_crops_train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "face_crops_val.csv", index=False)

    print(f"Total face crops: {len(df)}")
    print(f"Training crops: {len(train)}")
    print(f"Validation crops: {len(val)}")

    print("\nTraining labels:")
    print(train["label"].value_counts().sort_index())

    print("\nValidation labels:")
    print(val["label"].value_counts().sort_index())

    print("\nTraining groups:")
    print(sorted(train["group_id"].unique()))

    print("\nValidation groups:")
    print(sorted(val["group_id"].unique()))

    print("\nGroup overlap:", overlap)
    print("\nSaved:")
    print("  data/processed/face_crops_train.csv")
    print("  data/processed/face_crops_val.csv")

if __name__ == "__main__":
    main()