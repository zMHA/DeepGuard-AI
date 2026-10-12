
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FAILURES_CSV = ROOT / "data" / "processed" / "crop_failures.csv"
OUTPUT = ROOT / "results" / "crop_failure_inspection.png"


def main():
    failures = pd.read_csv(FAILURES_CSV)

    if failures.empty:
        print("No crop failures found.")
        return

    fig, axes = plt.subplots(
        len(failures), 2,
        figsize=(12, 4 * len(failures)),
        squeeze=False,
    )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    for row_idx, row in failures.iterrows():
        source_path = Path(str(row["image_path"]))
        if not source_path.is_absolute():
            source_path = ROOT / source_path

        original_ax = axes[row_idx, 0]
        detection_ax = axes[row_idx, 1]

        if not source_path.is_file():
            original_ax.set_title(f"Missing: {source_path}")
            original_ax.axis("off")
            detection_ax.axis("off")
            continue

        image = cv2.imread(str(source_path))
        if image is None:
            original_ax.set_title("Could not read image")
            original_ax.axis("off")
            detection_ax.axis("off")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
        )

        original_ax.imshow(rgb)
        original_ax.set_title(
            f"Original: {source_path.name}\n"
            f"Reason: {row['reason']}"
        )
        original_ax.axis("off")

        annotated = rgb.copy()
        for x, y, w, h in faces:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3,
            )

        detection_ax.imshow(annotated)
        detection_ax.set_title(
            f"Retry detection: {len(faces)} face(s)"
        )
        detection_ax.axis("off")

        print(
            f"{source_path}\n"
            f"  Image size: {image.shape[1]}x{image.shape[0]}\n"
            f"  Retry detections: {len(faces)}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"\nInspection image saved to: {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()