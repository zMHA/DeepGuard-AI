
"""Extract representative frames from FaceForensics++ videos.

Labels:
    0 = real
    1 = fake

The manifest records a shared group_id for related source videos and
their manipulated versions to support leakage-aware dataset splitting.
"""

import os
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames and generate a labeled CSV manifest."
    )
    parser.add_argument(
        "--real-dir",
        default="data/raw/faceforensics/"
        "original_sequences/youtube/c40/videos",
        help="Directory containing authentic videos.",
    )
    parser.add_argument(
        "--fake-dir",
        default="data/raw/faceforensics/"
        "manipulated_sequences/Deepfakes/c40/videos",
        help="Directory containing manipulated videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/frames",
        help="Directory in which extracted frames will be saved.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of evenly spaced frames to extract per video.",
    )
    return parser.parse_args()


def find_videos(directory):
    """Return supported video files in deterministic order."""
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Video directory does not exist: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def source_ids_from_fake_name(video_path):
    """Extract source video IDs from a name such as 183_253.mp4."""
    match = re.fullmatch(r"(\d+)_(\d+)", video_path.stem)

    if not match:
        raise ValueError(
            "Cannot infer source IDs from manipulated video filename "
            f"'{video_path.name}'. Expected a name like 183_253.mp4."
        )

    return match.group(1), match.group(2)


def build_source_groups(real_videos, fake_videos):
    """Group connected source IDs to prevent cross-split leakage.

    Each manipulated filename A_B connects source videos A and B.
    Connected components receive the same group_id. This is conservative:
    overlapping source relationships are kept in one group.
    """
    parent = {}

    def find(item):
        parent.setdefault(item, item)

        if parent[item] != item:
            parent[item] = find(parent[item])

        return parent[item]

    def union(first, second):
        root_first = find(first)
        root_second = find(second)

        if root_first != root_second:
            # Deterministic representative for repeatable group IDs.
            low, high = sorted((root_first, root_second))
            parent[high] = low

    real_ids = set()

    for video in real_videos:
        if not re.fullmatch(r"\d+", video.stem):
            raise ValueError(
                f"Unexpected authentic video filename: {video.name}. "
                "Expected a numeric filename such as 183.mp4."
            )
        real_ids.add(video.stem)
        find(video.stem)

    fake_pairs = {}

    for video in fake_videos:
        first, second = source_ids_from_fake_name(video)
        fake_pairs[video] = (first, second)
        union(first, second)

    # Every connected component gets a stable ID derived from its members.
    components = {}

    for source_id in list(parent):
        root = find(source_id)
        components.setdefault(root, []).append(source_id)

    source_to_group = {}

    for members in components.values():
        members = sorted(members, key=lambda value: int(value))
        group_id = "source_group_" + "_".join(members)

        for source_id in members:
            source_to_group[source_id] = group_id

    # Require both source videos for each selected manipulation to exist.
    for video, (first, second) in fake_pairs.items():
        if first not in real_ids or second not in real_ids:
            raise ValueError(
                f"Missing authentic source video for {video.name}. "
                f"Expected both {first}.mp4 and {second}.mp4 in "
                "the authentic video directory. Download the matching "
                "source videos before extracting frames."
            )

    return source_to_group, fake_pairs


def extract_video_frames(
    video_path,
    label,
    video_id,
    group_id,
    output_dir,
    num_frames,
):
    """Extract evenly spaced frames from one video.

    Returns manifest rows for successfully written frames.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Could not open video: {video_path}")
        cap.release()
        return []

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            print(f"WARNING: Video contains no readable frames: {video_path}")
            return []

        # Include the beginning and end of the video, with unique indices.
        frame_indices = np.unique(
            np.linspace(
                0,
                frame_count - 1,
                num=min(num_frames, frame_count),
                dtype=int,
            )
        )

        video_output_dir = output_dir / (
            "real" if label == 0 else "fake"
        ) / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success or frame is None:
                print(
                    f"WARNING: Failed to read frame {frame_index} "
                    f"from {video_path.name}"
                )
                continue

            image_name = f"frame_{frame_index:06d}.jpg"
            image_path = video_output_dir / image_name

            if not cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                print(f"WARNING: Failed to save image: {image_path}")
                continue

            rows.append(
                {
                    "image_path": image_path.as_posix(),
                    "label": label,
                    "video_id": video_id,
                    "group_id": group_id,
                }
            )

        return rows

    finally:
        cap.release()


def main():
    args = parse_args()

    if args.num_frames < 1:
        raise ValueError("--num-frames must be at least 1.")

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    real_videos = find_videos(real_dir)
    fake_videos = find_videos(fake_dir)

    if not real_videos:
        raise ValueError(f"No supported authentic videos found in {real_dir}")

    if not fake_videos:
        raise ValueError(
            f"No supported manipulated videos found in {fake_dir}"
        )

    source_to_group, fake_pairs = build_source_groups(
        real_videos, fake_videos
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"Authentic videos: {len(real_videos)}")
    print(f"Manipulated videos: {len(fake_videos)}")
    print(f"Frames requested per video: {args.num_frames}")

    for video in real_videos:
        source_id = video.stem
        rows.extend(
            extract_video_frames(
                video_path=video,
                label=0,
                video_id=f"real_{source_id}",
                group_id=source_to_group[source_id],
                output_dir=output_dir,
                num_frames=args.num_frames,
            )
        )

    for video in fake_videos:
        first, second = fake_pairs[video]
        rows.extend(
            extract_video_frames(
                video_path=video,
                label=1,
                video_id=f"fake_{video.stem}",
                group_id=source_to_group[first],
                output_dir=output_dir,
                num_frames=args.num_frames,
            )
        )

    if not rows:
        raise RuntimeError(
            "No frames were extracted successfully. "
            "Check video paths and OpenCV video decoding."
        )

    for row in rows:
        row["image_path"] = os.path.relpath(
        Path(row["image_path"]).resolve(),
        start=manifest_path.parent.resolve(),
    ).replace("\\", "/")

    manifest = pd.DataFrame(
        rows,
        columns=["image_path", "label", "video_id", "group_id"],
    )

    manifest.to_csv(manifest_path, index=False)

    print("\nExtraction complete.")
    print(f"Frames saved: {len(manifest)}")
    print(f"Real frames: {(manifest['label'] == 0).sum()}")
    print(f"Fake frames: {(manifest['label'] == 1).sum()}")
    print(f"Unique videos: {manifest['video_id'].nunique()}")
    print(f"Source groups: {manifest['group_id'].nunique()}")
    print(f"Manifest: {manifest_path}")
    print("\nFrames per class:")
    print(manifest.groupby("label").size().to_string())


if __name__ == "__main__":
    main()