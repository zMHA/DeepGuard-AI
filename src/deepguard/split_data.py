
"""Create leakage-aware train/validation splits from a frame manifest."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"image_path", "label", "video_id", "group_id"}


def split_manifest(
    manifest_path,
    output_dir="data/processed/splits",
    validation_fraction=0.2,
    seed=42,
):
    """Split frames by source group, never by individual frame or video."""
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    df = pd.read_csv(manifest_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Manifest contains no samples.")

    for column in ("image_path", "video_id", "group_id"):
        if df[column].isna().any() or (
            df[column].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"Manifest contains empty values in {column}.")

    labels = pd.to_numeric(df["label"], errors="coerce")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Labels must be numeric 0 (real) or 1 (fake).")

    df["label"] = labels.astype(int)

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Manifest must contain both real and fake samples.")

    # Each source group must have a consistent set of samples.
    group_ids = sorted(df["group_id"].astype(str).unique())
    if len(group_ids) < 2:
        raise ValueError(
            "At least two distinct source groups are required "
            "for a train/validation split."
        )

    # Keep group identifiers together. A seeded RNG makes the split repeatable.
    rng = np.random.default_rng(seed)
    shuffled_groups = np.array(group_ids, dtype=object)
    rng.shuffle(shuffled_groups)

    n_validation_groups = max(
        1, int(round(len(shuffled_groups) * validation_fraction))
    )
    n_validation_groups = min(
        n_validation_groups, len(shuffled_groups) - 1
    )

    validation_groups = set(
        shuffled_groups[:n_validation_groups].tolist()
    )
    train_groups = set(shuffled_groups[n_validation_groups:].tolist())

    train_df = df[df["group_id"].astype(str).isin(train_groups)].copy()
    validation_df = df[
        df["group_id"].astype(str).isin(validation_groups)
    ].copy()

    if train_df.empty or validation_df.empty:
        raise RuntimeError("The split produced an empty dataset.")

    if set(train_df["label"].unique()) != {0, 1}:
        raise ValueError(
            "Training split does not contain both classes. "
            "Review the source-group composition."
        )

    if set(validation_df["label"].unique()) != {0, 1}:
        raise ValueError(
            "Validation split does not contain both classes. "
            "Review the source-group composition or validation fraction."
        )

    overlap = (
        set(train_df["group_id"].astype(str))
        & set(validation_df["group_id"].astype(str))
    )
    if overlap:
        raise AssertionError(
            f"Source-group leakage detected: {sorted(overlap)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    print("Split complete.")
    print(f"Source groups: {len(group_ids)}")
    print(f"Training groups: {len(train_groups)}")
    print(f"Validation groups: {len(validation_groups)}")
    print(f"Training frames: {len(train_df)}")
    print(f"Validation frames: {len(validation_df)}")
    print(f"Training class counts:\n{train_df['label'].value_counts().sort_index()}")
    print(
        "Validation class counts:\n"
        f"{validation_df['label'].value_counts().sort_index()}"
    )
    print(f"Training manifest: {train_path}")
    print(f"Validation manifest: {validation_path}")

    return train_df, validation_df


def main():
    parser = argparse.ArgumentParser(
        description="Split a frame manifest by source group."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Input frame manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
        help="Directory for train.csv and validation.csv.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of source groups assigned to validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()