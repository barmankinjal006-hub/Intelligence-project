"""
EcoSort Challenge — Improved Model Training & Export Template
=============================================================
Usage:
    python train_template.py --epochs 10 --batch_size 32 --lr 0.001
"""

# -----------------------------
# Import required libraries
# -----------------------------
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------
# Define class labels
# -----------------------------
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# -----------------------------
# Define improved CNN classifier
# -----------------------------
class ImprovedClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # Convolutional feature extractor with BatchNorm + ReLU + Pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # First conv layer
            nn.BatchNorm2d(16),                          # Normalize activations
            nn.ReLU(),                                   # Non-linearity
            nn.MaxPool2d(2, 2),                          # Downsample

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Second conv layer
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Third conv layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Adaptive pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected classifier with dropout
        self.fc = nn.Sequential(
            nn.Dropout(0.4),                             # Prevent overfitting
            nn.Linear(64, num_classes)                   # Final classification layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                             # Extract features
        x = self.pool(x)                                 # Pool to fixed size
        x = torch.flatten(x, 1)                          # Flatten for FC layer
        return self.fc(x)                                # Output logits


# -----------------------------
# Count trainable parameters
# -----------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Create dataloaders with augmentation
# -----------------------------
def get_dataloaders(data_dir: str, batch_size: int = 32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'. Run 'python download_data.py' first!"
        )

    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),               # Flip images randomly
        transforms.RandomRotation(10),                   # Rotate slightly
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust colors
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataloaders
    train_loader = DataLoader(
        datasets.ImageFolder(train_dir, transform=train_transform),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, transform=val_transform),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader


# -----------------------------
# Export trained model to ONNX
# -----------------------------
def export_to_onnx(model: nn.Module, output_path: str = "model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 96, 96)

    print(f"Exporting model to '{output_path}'...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported successfully! File size: {size_mb:.2f} MB (Limit: <= 1.50 MB)")


# -----------------------------
# Main training loop
# -----------------------------
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="EcoSort Improved Model Training Template")
    parser.add_argument("--data_dir", type=str, default="data/public", help="Path to public dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX filename")
    args = parser.parse_args()

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_loader, val_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"Loaded {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation images.")

    # Initialize model
    model = ImprovedClassifier(num_classes=len(CLASSES)).to(device)
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,} / 350,000 max allowed")

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Early stopping setup
    best_val_acc = 0.0
    patience, patience_counter = 3, 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = (correct / total) * 100.0

        # Validation loop
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        val_acc = (v_correct / v_total) * 100.0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        # Step scheduler
        scheduler.step()

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Export model to ONNX
    export_to_onnx(model.cpu(), output_path=args.output)
    print(f"\nNext step: Verify your submission before uploading:")
    print(f"  python verify_submission.py --model {args.output}")


# -----------------------------
# Run main if script is executed
# -----------------------------
if __name__ == "__main__":
    main()
