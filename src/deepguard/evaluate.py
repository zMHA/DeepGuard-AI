
"""Evaluate a trained DeepGuard-AI classifier on a manifest."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.deepguard.classifier import build_classifier
from src.deepguard.dataset import DeepfakeDataset


def evaluate(
    manifest_path,
    checkpoint_path,
    image_root=None,
    batch_size=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    dataset = DeepfakeDataset(
        manifest_path=manifest_path,
        image_root=image_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_classifier(num_classes=2, pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    print(f"Device: {device}")
    print(f"Evaluation samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(
        matrix,
        index=["Actual Real", "Actual Fake"],
        columns=["Predicted Real", "Predicted Fake"],
    ).to_string())

    return all_labels, all_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a DeepGuard-AI checkpoint."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/validation.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--image-root",
        default="data/processed",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    evaluate(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        image_root=args.image_root,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()