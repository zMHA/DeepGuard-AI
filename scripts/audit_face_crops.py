
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "processed" / "face_crops_manifest.csv"
FAILURES = PROJECT_ROOT / "data" / "processed" / "crop_failures.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "face_crops"

SAMPLES_PER_CLASS = 12
SEED = 42


def main():
    df = pd.read_csv(MANIFEST)

    required = {"image_path", "label", "video_id", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    print("\n=== DATASET SUMMARY ===")
    print(f"Total crops: {len(df)}")
    print("\nClass counts (0=real, 1=fake):")
    print(df["label"].value_counts().sort_index())

    print("\nCrops per source group:")
    print(pd.crosstab(df["group_id"], df["label"]))

    print("\nUnique videos per class:")
    print(df.groupby("label")["video_id"].nunique())

    # Check whether manifest paths point to existing images.
    def resolve_image_path(value):
        path = Path(str(value))
        if path.is_absolute():
            return path
        return IMAGE_ROOT / path

    df["resolved_path"] = df["image_path"].map(resolve_image_path)
    df["exists"] = df["resolved_path"].map(lambda p: p.is_file())

    missing_files = df.loc[~df["exists"]]
    print(f"\nMissing image files: {len(missing_files)}")
    if not missing_files.empty:
        print(missing_files[["image_path", "label", "video_id"]].to_string(index=False))

    # Check whether a file path appears more than once.
    duplicate_paths = df[df["image_path"].duplicated(keep=False)]
    print(f"\nRows with duplicate image paths: {len(duplicate_paths)}")
    if not duplicate_paths.empty:
        print(duplicate_paths[["image_path", "label", "video_id"]].to_string(index=False))

    # Review crop failures, if the report exists.
    if FAILURES.is_file():
        failures = pd.read_csv(FAILURES)
        print(f"\nCrop failure report rows: {len(failures)}")
        if not failures.empty:
            print(failures.head(20).to_string(index=False))
    else:
        print("\nNo crop_failures.csv report found.")

    # Sample crops from both classes.
    rng = random.Random(SEED)
    samples = []

    for label in (0, 1):
        candidates = df[
            (df["label"] == label) & df["exists"]
        ]

        count = min(SAMPLES_PER_CLASS, len(candidates))
        if count:
            chosen_indices = rng.sample(list(candidates.index), count)
            samples.extend(df.loc[chosen_indices].to_dict("records"))

    if not samples:
        print("\nNo existing crops were available to display.")
        return

    columns = 4
    rows = (len(samples) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, item in zip(axes, samples):
        try:
            with Image.open(item["resolved_path"]) as img:
                ax.imshow(img.convert("RGB"))

            label_name = "REAL" if item["label"] == 0 else "FAKE"
            ax.set_title(
                f"{label_name}\n"
                f"Video: {item['video_id']}\n"
                f"Group: {item['group_id']}",
                fontsize=8,
            )
            ax.axis("off")
        except Exception as exc:
            ax.set_title(f"Could not read image\n{exc}", fontsize=8)
            ax.axis("off")

    fig.suptitle("DeepGuard-AI Face Crop Audit", fontsize=14)
    fig.tight_layout()

    output_path = PROJECT_ROOT / "results" / "face_crop_audit.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"\nVisual audit saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()