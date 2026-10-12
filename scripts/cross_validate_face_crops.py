
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupKFold

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepguard.classifier import build_classifier
from src.deepguard.dataset import DeepfakeDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )
            all_labels.extend(labels.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Validation loader is empty.")

    predictions = np.asarray(all_predictions)
    labels = np.asarray(all_labels)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    accuracy = float((predictions == labels).mean())

    return {
        "val_loss": total_loss / total_samples,
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_one_fold(
    train_df,
    val_df,
    image_root,
    output_dir,
    fold,
    epochs,
    batch_size,
    learning_rate,
    seed,
    pretrained,
    num_workers,
):
    set_seed(seed + fold)

    fold_dir = output_dir / f"fold_{fold}"
    manifest_dir = fold_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = manifest_dir / "train.csv"
    val_manifest = manifest_dir / "validation.csv"

    train_df.to_csv(train_manifest, index=False)
    val_df.to_csv(val_manifest, index=False)

    train_dataset = DeepfakeDataset(
        manifest_path=train_manifest,
        image_size=224,
        image_root=image_root,
    )
    val_dataset = DeepfakeDataset(
        manifest_path=val_manifest,
        image_size=224,
        image_root=image_root,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    generator = torch.Generator()
    generator.manual_seed(seed + fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_classifier(pretrained=pretrained).to(device)

    # Train the classification head first, matching the baseline.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    checkpoint_path = fold_dir / "best_model.pt"
    best_val_loss = float("inf")
    best_metrics = None
    best_epoch = 0

    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/5")
    print(f"Device: {device}")
    print(f"Training crops: {len(train_dataset)}")
    print(f"Validation crops: {len(val_dataset)}")
    print(f"Training groups: {sorted(train_df['group_id'].unique())}")
    print(f"Validation groups: {sorted(val_df['group_id'].unique())}")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_train_loss += loss.item() * batch_size_actual
            total_train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_train_samples += batch_size_actual

        train_loss = total_train_loss / total_train_samples
        train_acc = total_train_correct / total_train_samples

        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_acc={metrics['accuracy']:.4f} | "
            f"val_f1={metrics['f1_fake']:.4f}"
        )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_metrics = metrics.copy()
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "validation_loss": best_val_loss,
                    "seed": seed + fold,
                    "class_mapping": {"real": 0, "fake": 1},
                    "train_groups": sorted(
                        train_df["group_id"].unique()
                    ),
                    "validation_groups": sorted(
                        val_df["group_id"].unique()
                    ),
                },
                checkpoint_path,
            )

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "train_crops": len(train_dataset),
        "validation_crops": len(val_dataset),
        "train_groups": ";".join(
            sorted(train_df["group_id"].unique())
        ),
        "validation_groups": ";".join(
            sorted(val_df["group_id"].unique())
        ),
        **best_metrics,
        "checkpoint": str(checkpoint_path),
    }

    print(
        f"Fold {fold} best: epoch={best_epoch}, "
        f"accuracy={best_metrics['accuracy']:.4f}, "
        f"F1={best_metrics['f1_fake']:.4f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Five-fold group cross-validation for DeepGuard-AI."
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/face_crops_manifest.csv",
    )
    parser.add_argument(
        "--image-root",
        default="data/processed/face_crops",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cv_face_crops",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Do not use ImageNet-pretrained weights.",
    )
    parser.set_defaults(pretrained=True)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    manifest_path = Path(args.manifest)
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir)

    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = {"image_path", "label", "video_id", "group_id"}
    missing = required - set(df.columns)

    if missing:
        parser.error(f"Manifest missing columns: {sorted(missing)}")

    if df.empty:
        parser.error("Manifest contains no samples.")

    if df[list(required)].isna().any().any():
        parser.error("Required manifest fields contain missing values.")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        parser.error("Labels must be 0 (real) or 1 (fake).")

    if df["group_id"].nunique() != 5:
        parser.error(
            "Expected exactly 5 source groups for five-fold "
            f"evaluation; found {df['group_id'].nunique()}."
        )

    if set(df["label"].unique()) != {0, 1}:
        parser.error("Both real (0) and fake (1) samples are required.")

    print("Dataset:", manifest_path)
    print("Total crops:", len(df))
    print("Class counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nCrops per source group:")
    print(df.groupby("group_id").size().to_string())

    output_dir.mkdir(parents=True, exist_ok=True)

    group_kfold = GroupKFold(n_splits=5)
    results = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(df, df["label"], groups=df["group_id"]),
        start=1,
    ):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Explicitly reject any source-group overlap.
        overlap = set(train_df["group_id"]) & set(val_df["group_id"])

        if overlap:
            raise RuntimeError(
                f"Group leakage in fold {fold}: {sorted(overlap)}"
            )

        result = train_one_fold(
            train_df=train_df,
            val_df=val_df,
            image_root=image_root,
            output_dir=output_dir,
            fold=fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            pretrained=args.pretrained,
            num_workers=args.num_workers,
        )

        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            output_dir / "fold_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(results)
    metrics = [
        "val_loss",
        "accuracy",
        "precision_fake",
        "recall_fake",
        "f1_fake",
    ]

    summary = pd.DataFrame({
        "metric": metrics,
        "mean": [results_df[m].mean() for m in metrics],
        "std": [
            results_df[m].std(ddof=1) for m in metrics
        ],
    })

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("FIVE-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(
        results_df[
            ["fold", "best_epoch", "val_loss", "accuracy",
             "precision_fake", "recall_fake", "f1_fake"]
        ].to_string(index=False)
    )

    print("\nMean and standard deviation across folds:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nFold results: {output_dir / 'fold_results.csv'}")
    print(f"Summary:      {output_dir / 'summary.csv'}")
    print(f"Checkpoints:  {output_dir}")


if __name__ == "__main__":
    main()