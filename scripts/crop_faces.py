
import argparse
from pathlib import Path

import cv2
import pandas as pd


def crop_faces(manifest_path, image_root, output_dir, output_manifest):
    manifest_path = Path(manifest_path)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_manifest = Path(output_manifest)

    df = pd.read_csv(manifest_path)
    required = {"image_path", "label", "video_id", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    cascade_path = (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {cascade_path}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = []

    for _, row in df.iterrows():
        relative_path = Path(row["image_path"])
        source_path = image_root / relative_path

        image = cv2.imread(str(source_path))
        if image is None:
            failed.append((str(source_path), "image_read_failed"))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) == 0:
            failed.append((str(source_path), "no_face_detected"))
            continue

        # Keep the largest detected face.
        x, y, width, height = max(
            faces, key=lambda box: int(box[2]) * int(box[3])
        )

        margin_x = int(width * 0.20)
        margin_y = int(height * 0.20)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + width + margin_x)
        y2 = min(image.shape[0], y + height + margin_y)

        crop = image[y1:y2, x1:x2]
        relative_output = Path(str(row["label"])) / row["video_id"] / (
            Path(row["image_path"]).stem + ".jpg"
        )
        destination = output_dir / relative_output
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(destination), crop):
            failed.append((str(source_path), "crop_write_failed"))
            continue

        rows.append({
            "image_path": relative_output.as_posix(),
            "label": int(row["label"]),
            "video_id": row["video_id"],
            "group_id": row["group_id"],
        })

    result = pd.DataFrame(
        rows, columns=["image_path", "label", "video_id", "group_id"]
    )
    result.to_csv(output_manifest, index=False)

    print(f"Input frames: {len(df)}")
    print(f"Face crops saved: {len(result)}")
    print(f"Failed frames: {len(failed)}")
    print(f"Crop manifest: {output_manifest}")

    if failed:
        failure_path = output_manifest.with_name("crop_failures.csv")
        pd.DataFrame(failed, columns=["image_path", "reason"]).to_csv(
            failure_path, index=False
        )
        print(f"Failure report: {failure_path}")

    if not result.empty:
        print("\nCrop counts by label:")
        print(result.groupby("label").size())
        print("\nCrop counts by source group:")
        print(result.groupby("group_id").size())

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create face crops from the frame manifest."
    )
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--image-root", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed/face_crops")
    parser.add_argument(
        "--output-manifest",
        default="data/processed/face_crops_manifest.csv",
    )
    args = parser.parse_args()

    crop_faces(
        args.manifest,
        args.image_root,
        args.output_dir,
        args.output_manifest,
    )


if __name__ == "__main__":
    main()