
from pathlib import Path
import csv

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """Load labeled real and manipulated facial images from a CSV manifest."""

    VALID_LABELS = {0, 1}

    def __init__(
        self,
        manifest_path,
        image_size=224,
        image_root=None,
    ):
        self.manifest_path = Path(manifest_path).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.image_root = (
            Path(image_root).resolve()
            if image_root is not None
            else self.manifest_path.parent
        )

        self.samples = []

        with self.manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("The manifest is empty.")

            required = {"image_path", "label"}
            missing = required - set(reader.fieldnames)

            if missing:
                raise ValueError(
                    f"Manifest is missing columns: {sorted(missing)}"
                )

            for row_number, row in enumerate(reader, start=2):
                raw_path = (row.get("image_path") or "").strip()
                raw_label = (row.get("label") or "").strip()

                if not raw_path:
                    raise ValueError(
                        f"Missing image_path at CSV row {row_number}"
                    )

                try:
                    label = int(raw_label)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid label at CSV row {row_number}: "
                        f"{raw_label!r}"
                    ) from exc

                if label not in self.VALID_LABELS:
                    raise ValueError(
                        f"Label must be 0 (real) or 1 (fake); "
                        f"found {label} at row {row_number}"
                    )

                image_path = Path(raw_path)

                if not image_path.is_absolute():
                    image_path = self.image_root / image_path

                image_path = image_path.resolve()

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found at CSV row {row_number}: "
                        f"{image_path}"
                    )

                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError("The manifest contains no image samples.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)