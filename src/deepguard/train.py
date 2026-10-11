
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.deepguard.classifier import build_classifier
from src.deepguard.dataset import DeepfakeDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }



def train(
    train_manifest,
    validation_manifest,
    image_root=None,
    output_dir="results/checkpoints",
    epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    seed=42,
    num_workers=0,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    set_seed(seed)

    train_dataset = DeepfakeDataset(
        manifest_path=train_manifest,
        image_size=224,
        image_root=image_root,
    )
    val_dataset = DeepfakeDataset(
        manifest_path=validation_manifest,
        image_size=224,
        image_root=image_root,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation sets must be non-empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_classifier(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    checkpoint_path = output_path / "best_model.pt"

    print(f"Device: {device}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_val_loss,
                    "seed": seed,
                    "class_mapping": {"real": 0, "fake": 1},
                },
                checkpoint_path,
            )

    print(f"Best checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Train the DeepGuard-AI ResNet-18 baseline."
    )
    parser.add_argument(
        "--train-manifest",
        default="data/processed/splits/train.csv",
    )
    parser.add_argument(
        "--validation-manifest",
        default="data/processed/splits/validation.csv",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="results/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        image_root=args.image_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()