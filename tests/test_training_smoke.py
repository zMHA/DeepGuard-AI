import csv

import torch
from PIL import Image

from src.deepguard.train import train


def test_training_completes_and_saves_checkpoint(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create six synthetic images for a pipeline test only.
    for index in range(6):
        label = index % 2
        color = (240, 240, 240) if label == 0 else (20, 20, 20)

        Image.new("RGB", (64, 64), color).save(
            image_dir / f"sample_{index}.jpg"
        )

    train_manifest = tmp_path / "train.csv"
    validation_manifest = tmp_path / "validation.csv"

    # Four images for training.
    with train_manifest.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])

        for index in range(4):
            writer.writerow(
                [f"images/sample_{index}.jpg", index % 2]
            )

    # Two separate images for validation.
    with validation_manifest.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])

        for index in range(4, 6):
            writer.writerow(
                [f"images/sample_{index}.jpg", index % 2]
            )

    checkpoint_dir = tmp_path / "checkpoints"

    checkpoint_path = train(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        image_root=tmp_path,
        output_dir=checkpoint_dir,
        epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        seed=42,
        num_workers=0,
    )

    assert checkpoint_path.is_file()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert "model_state_dict" in checkpoint
    assert checkpoint["epoch"] == 1
    assert checkpoint["class_mapping"] == {
        "real": 0,
        "fake": 1,
    }