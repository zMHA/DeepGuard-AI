
import csv

from PIL import Image

from src.deepguard.dataset import DeepfakeDataset


def test_dataset_loads_real_and_fake_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    Image.new("RGB", (64, 64), "white").save(
        image_dir / "real.jpg"
    )
    Image.new("RGB", (64, 64), "black").save(
        image_dir / "fake.jpg"
    )

    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        writer.writerow(["images/real.jpg", 0])
        writer.writerow(["images/fake.jpg", 1])

    dataset = DeepfakeDataset(
        manifest_path=manifest,
        image_size=224,
    )

    assert len(dataset) == 2

    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label.item() == 0

    image, label = dataset[1]
    assert image.shape == (3, 224, 224)
    assert label.item() == 1


def test_dataset_rejects_invalid_label(tmp_path):
    Image.new("RGB", (32, 32), "white").save(
        tmp_path / "sample.jpg"
    )

    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        writer.writerow(["sample.jpg", 2])

    try:
        DeepfakeDataset(manifest)
    except ValueError as exc:
        assert "Label must be 0" in str(exc)
    else:
        raise AssertionError("Expected invalid label to be rejected")