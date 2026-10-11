
import csv

from src.deepguard.split_data import split_manifest


def test_split_keeps_video_groups_separate(tmp_path):
    manifest = tmp_path / "manifest.csv"

    # Four source-video groups, with two frames per group.
    rows = [
        ("a_001.jpg", 0, "video_a"),
        ("a_002.jpg", 0, "video_a"),
        ("b_001.jpg", 0, "video_b"),
        ("b_002.jpg", 0, "video_b"),
        ("c_001.jpg", 1, "video_c"),
        ("c_002.jpg", 1, "video_c"),
        ("d_001.jpg", 1, "video_d"),
        ("d_002.jpg", 1, "video_d"),
    ]

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "video_id"])
        writer.writerows(rows)

    output_dir = tmp_path / "splits"

    train_rows, val_rows = split_manifest(
        manifest,
        output_dir=output_dir,
        validation_fraction=0.5,
        seed=42,
    )

    train_groups = {row["video_id"] for row in train_rows}
    val_groups = {row["video_id"] for row in val_rows}

    assert train_groups.isdisjoint(val_groups)
    assert len(train_rows) + len(val_rows) == len(rows)

    assert (output_dir / "train.csv").is_file()
    assert (output_dir / "validation.csv").is_file()

    for filename in ("train.csv", "validation.csv"):
        with (output_dir / filename).open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            assert set(reader.fieldnames) == {
                "image_path", "label", "video_id"
            }


def test_split_rejects_missing_video_id(tmp_path):
    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "video_id"])
        writer.writerow(["one.jpg", 0, ""])
        writer.writerow(["two.jpg", 1, "video_b"])

    try:
        split_manifest(manifest, output_dir=tmp_path / "splits")
    except ValueError as exc:
        assert "video_id" in str(exc)
    else:
        raise AssertionError("Expected missing video_id to be rejected")