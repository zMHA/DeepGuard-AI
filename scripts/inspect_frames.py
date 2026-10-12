
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/processed/frames")
SPLITS = [
    ("Real", ROOT / "real"),
    ("Fake", ROOT / "fake"),
]

fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for row, (label, directory) in enumerate(SPLITS):
    images = sorted(directory.glob("*/*.jpg"))[:5]

    for col, ax in enumerate(axes[row]):
        ax.axis("off")

        if col >= len(images):
            continue

        with Image.open(images[col]) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(f"{label}: {images[col].parent.name}")

plt.tight_layout()
plt.savefig("results/frame_inspection.png", dpi=150)
print("Saved contact sheet: results/frame_inspection.png")