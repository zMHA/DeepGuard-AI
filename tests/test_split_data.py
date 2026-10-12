
import csv

import pandas as pd
import pytest

from src.deepguard.split_data import split_manifest


def create_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "video_id", "group_id"])
        writer.writerows(rows)


def test_split_keeps_source_groups_separate(tmp_path):
    manifest = tmp_path / "manifest.csv"

    # Four source groups. Each group contains both real and fake samples.
    rows = [
        ("a_real_1.jpg", 0, "real_a", "group_a"),
        ("a_real_2.jpg", 0, "real_a", "group_a"),
        ("a_fake_1.jpg", 1, "fake_a", "group_a"),
        ("a_fake_2.jpg", 1, "fake_a", "group_a"),
        ("b_real_1.jpg", 0, "real_b", "group_b"),
        ("b_real_2.jpg", 0, "real_b", "group_b"),
        ("b_fake_1.jpg", 1, "fake_b", "group_b"),
        ("b_fake_2.jpg", 1, "fake_b", "group_b"),
        ("c_real_1.jpg", 0, "real_c", "group_c"),
        ("c_real_2.jpg", 0, "real_c", "group_c"),
        ("c_fake_1.jpg", 1, "fake_c", "group_c"),
        ("c_fake_2.jpg", 1, "fake_c", "group_c"),
        ("d_real_1.jpg", 0, "real_d", "group_d"),
        ("d_real_2.jpg", 0, "real_d", "group_d"),
        ("d_fake_1.jpg", 1, "fake_d", "group_d"),
        ("d_fake_2.jpg", 1, "fake_d", "group_d"),
    ]
    create_manifest(manifest, rows)

    train_df, val_df = split_manifest(
        manifest,
        output_dir=tmp_path / "splits",
        validation_fraction=0.5,
        seed=42,
    )

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])

    assert train_groups.isdisjoint(val_groups)
    assert len(train_groups) == 2
    assert len(val_groups) == 2
    assert set(train_df["label"]) == {0, 1}
    assert set(val_df["label"]) == {0, 1}

    assert (tmp_path / "splits" / "train.csv").is_file()
    assert (tmp_path / "splits" / "validation.csv").is_file()


def test_split_rejects_missing_group_id(tmp_path):
    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "video_id"])
        writer.writerow(["one.jpg", 0, "video_a"])
        writer.writerow(["two.jpg", 1, "video_b"])

    with pytest.raises(ValueError, match="group_id"):
        split_manifest(manifest, output_dir=tmp_path / "splits")


def test_split_rejects_empty_group_id(tmp_path):
    manifest = tmp_path / "manifest.csv"

    rows = [
        ("a_real.jpg", 0, "real_a", ""),
        ("a_fake.jpg", 1, "fake_a", ""),
        ("b_real.jpg", 0, "real_b", "group_b"),
        ("b_fake.jpg", 1, "fake_b", "group_b"),
    ]
    create_manifest(manifest, rows)

    with pytest.raises(ValueError, match="group_id"):
        split_manifest(manifest, output_dir=tmp_path / "splits")