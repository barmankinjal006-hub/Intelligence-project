
"""
EcoSort Challenge - Advanced Lightweight CNN
=============================================
Classes: cardboard, glass, metal, paper, plastic, trash

Constraints:
    Parameters <= 350,000
    ONNX size <= 1.5 MB
    Input: (B, 3, 96, 96)
    Output: (B, 6)

Usage:
    python train_template.py --epochs 40 --batch_size 32 --lr 0.001

"""

import argparse
import copy
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# 1. CONFIGURATION
# ==========================================

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_SIZE = 96
MAX_PARAMS = 350000
MAX_ONNX_MB = 1.5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. ADVANCED CNN BUILDING BLOCKS
# ==========================================

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + SiLU."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """Attention mechanism for important feature channels."""

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.fc(self.pool(x))
        return x * attention


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise convolution followed by pointwise convolution.
    Reduces parameters while extracting useful image features.
    """

    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch)
        )

        self.se = (
            SqueezeExcitation(out_ch)
            if use_se else nn.Identity()
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch)
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.se(out)

        return self.activation(out + identity)


# ==========================================
# 3. ADVANCED ECO SORT CLASSIFIER
# ==========================================

class ImprovedClassifier(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.stem = ConvBNAct(
            3, 24, kernel=3, stride=2
        )

        self.features = nn.Sequential(

            # Stage 1
            DepthwiseSeparableBlock(
                24, 24, stride=1
            ),

            # Stage 2
            DepthwiseSeparableBlock(
                24, 32, stride=2
            ),
            DepthwiseSeparableBlock(
                32, 32, stride=1
            ),

            # Stage 3
            DepthwiseSeparableBlock(
                32, 64, stride=2
            ),
            DepthwiseSeparableBlock(
                64, 64, stride=1
            ),

            # Stage 4
            DepthwiseSeparableBlock(
                64, 96, stride=2
            ),
            DepthwiseSeparableBlock(
                96, 96, stride=1
            ),

            # Stage 5
            DepthwiseSeparableBlock(
                96, 128, stride=2
            ),
            DepthwiseSeparableBlock(
                128, 128, stride=1
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        return self.classifier(x)


# ==========================================
# 4. PARAMETER COUNT
# ==========================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

def get_dataloaders(data_dir, batch_size=32):

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}"
        )

    # Training augmentation
    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(12),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03
        ),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD),

        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.12)
        )
    ])

    # Validation: no random augmentation
    val_transform = transforms.Compose([

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    # Ensure labels match the required output order
    expected_mapping = {
        name: i for i, name in enumerate(CLASSES)
    }

    if train_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"Incorrect training classes: "
            f"{train_dataset.class_to_idx}"
        )

    if val_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"Incorrect validation classes: "
            f"{val_dataset.class_to_idx}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print("\nClass distribution:")

    for name, index in expected_mapping.items():

        count = sum(
            1 for _, label in train_dataset.samples
            if label == index
        )

        print(f"{name}: {count} training images")

    return train_loader, val_loader


# ==========================================
# 6. TRAINING AND VALIDATION
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision on GPU
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return epoch_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_accuracy = 100.0 * correct / max(total, 1)

    return val_loss, val_accuracy


# ==========================================
# 7. ONNX EXPORT
# ==========================================

def export_to_onnx(model, output_path):

    model.eval()
    model.cpu()

    dummy_input = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    )

    print("\nExporting ONNX model...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False
    )

    size_mb = os.path.getsize(output_path) / (
        1024 * 1024
    )

    print(f"ONNX file: {output_path}")
    print(f"ONNX size: {size_mb:.3f} MB")

    if size_mb > MAX_ONNX_MB:
        print("WARNING: ONNX size exceeds 1.5 MB!")

    else:
        print("ONNX size requirement satisfied!")


# ==========================================
# 8. MAIN TRAINING FUNCTION
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/public"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Select GPU or CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("       ADVANCED ECOSORT MODEL TRAINING")
    print("=" * 55)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load datasets
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        args.batch_size
    )

    print("\nTraining images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # Initialize model
    model = ImprovedClassifier(
        num_classes=len(CLASSES)
    ).to(device)

    params = count_parameters(model)

    print("\nTrainable parameters:", f"{params:,}")
    print("Parameter limit:", f"{MAX_PARAMS:,}")

    if params > MAX_PARAMS:
        raise RuntimeError(
            "Model exceeds parameter limit!"
        )

    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    # Best model tracking
    best_val_acc = 0.0
    best_val_loss = float("inf")

    best_weights = copy.deepcopy(model.state_dict())

    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        # Save the best model
        improved = (
            val_acc > best_val_acc
            or (
                val_acc == best_val_acc
                and val_loss < best_val_loss
            )
        )

        if improved:

            best_val_acc = val_acc
            best_val_loss = val_loss

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            torch.save(
                best_weights,
                "best_model.pth"
            )

            print("  -> Best model saved!")

        else:

            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:

            print("\nEarly stopping activated.")

            break

    # Restore best validation model
    model.load_state_dict(best_weights)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Export best model
    export_to_onnx(
        model,
        args.output
    )

    print("\nNext step:")
    print(
        f"python verify_submission.py --model {args.output}"
    )


if __name__ == "__main__":
    main()
