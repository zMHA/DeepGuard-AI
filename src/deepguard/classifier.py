
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_classifier(
    num_classes=2,
    pretrained=False,
):
    """Create a ResNet-18 classifier for real/fake image detection."""

    if num_classes != 2:
        raise ValueError("DeepGuard-AI currently supports 2 classes.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_classifier(pretrained=False)
    sample = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(sample)

    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(logits.shape))
    print("Predicted classes:", logits.argmax(dim=1).tolist())