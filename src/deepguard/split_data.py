
import argparse
import csv
from pathlib import Path

import numpy as np


def split_manifest(
    manifest_path,
    output_dir="data/processed/splits",
    validation_fraction=0.2,
    seed=42,
):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir)

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    with manifest_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required = {"image_path", "label", "video_id"}
        if not reader.fieldnames or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "Manifest must contain image_path, label, "
                "and video_id columns."
            )

        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("At least two images are required.")

    groups = {}
    for row in rows:
        video_id = row["video_id"].strip()

        if not video_id:
            raise ValueError("Every row must have a video_id.")

        try:
            label = int(row["label"])
        except ValueError as exc:
            raise ValueError("Labels must be integers 0 or 1.") from exc

        if label not in (0, 1):
            raise ValueError("Labels must be 0 (real) or 1 (fake).")

        row["label"] = str(label)
        groups.setdefault(video_id, []).append(row)

    group_ids = np.array(sorted(groups.keys()), dtype=object)

    if len(group_ids) < 2:
        raise ValueError(
            "At least two distinct video groups are required."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(group_ids)

    val_count = max(
        1,
        min(
            len(shuffled) - 1,
            round(len(shuffled) * validation_fraction),
        ),
    )

    val_groups = set(shuffled[:val_count])
    train_groups = set(shuffled[val_count:])

    train_rows = [
        row for row in rows if row["video_id"].strip() in train_groups
    ]
    val_rows = [
        row for row in rows if row["video_id"].strip() in val_groups
    ]

    assert train_groups.isdisjoint(val_groups)

    output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["image_path", "label", "video_id"]

    for filename, subset in (
        ("train.csv", train_rows),
        ("validation.csv", val_rows),
    ):
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {column: row[column] for column in columns}
                for row in subset
            )

    print(f"Training images: {len(train_rows)}")
    print(f"Validation images: {len(val_rows)}")
    print(f"Training video groups: {len(train_groups)}")
    print(f"Validation video groups: {len(val_groups)}")
    print("Video-group overlap: 0")
    print(f"Splits saved to: {output_dir.resolve()}")

    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_manifest(
        args.manifest,
        args.output_dir,
        args.validation_fraction,
        args.seed,
    )


if __name__ == "__main__":
    main()