import argparse
import csv
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {"image_path", "label", "video_id"}


def validate_manifest(manifest_path, image_root=None):
    manifest_path = Path(manifest_path).resolve()

    root = (
        Path(image_root).resolve()
        if image_root
        else manifest_path.parent
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    counts = Counter()
    missing_files = []
    total = 0

    with manifest_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if (
            not reader.fieldnames
            or not REQUIRED_COLUMNS.issubset(reader.fieldnames)
        ):
            raise ValueError(
                "Manifest must contain image_path, label, "
                "and video_id columns."
            )

        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            raw_label = (row.get("label") or "").strip()
            raw_video_id = (row.get("video_id") or "").strip()

            if not raw_path:
                raise ValueError(
                    f"Empty image_path at CSV row {row_number}"
                )

            if not raw_video_id:
                raise ValueError(
                    f"Empty video_id at CSV row {row_number}"
                )

            try:
                label = int(raw_label)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at CSV row {row_number}: "
                    f"{raw_label!r}"
                ) from exc

            if label not in (0, 1):
                raise ValueError(
                    f"Label must be 0 (real) or 1 (fake), "
                    f"found {label} at row {row_number}"
                )

            image_path = Path(raw_path)

            if not image_path.is_absolute():
                image_path = root / image_path

            if not image_path.is_file():
                missing_files.append(str(image_path))

            counts[label] += 1
            total += 1

    if total == 0:
        raise ValueError("Manifest contains no samples.")

    print(f"Manifest: {manifest_path}")
    print(f"Total samples: {total}")
    print(f"Real (0): {counts[0]}")
    print(f"Fake (1): {counts[1]}")
    print(f"Missing images: {len(missing_files)}")

    for path in missing_files[:20]:
        print(f"  MISSING: {path}")

    if missing_files:
        raise ValueError(
            f"{len(missing_files)} image file(s) are missing."
        )

    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            "Both real and fake classes must be present."
        )

    print("Manifest validation passed.")
    return counts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the DeepGuard-AI image manifest, "
            "including source-video identifiers."
        )
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Path to the CSV manifest.",
    )

    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "Root directory for relative image paths. "
            "Defaults to the manifest's parent directory."
        ),
    )

    args = parser.parse_args()

    validate_manifest(
        manifest_path=args.manifest,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()