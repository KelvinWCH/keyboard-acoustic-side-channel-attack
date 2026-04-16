import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import time
import json


SPEC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "spectrograms")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
NUM_CLASSES = 36
IMG_SIZE = 224


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.10), ratio=(3.0, 10.0)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def build_model(num_classes, device):
    model = timm.create_model(
        "coatnet_0_rw_224",
        pretrained=True,
        num_classes=num_classes,
    )
    return model.to(device)


def set_parameter_requires_grad(model, freeze):
    for name, param in model.named_parameters():
        if 'head' not in name and 'fc' not in name:
            param.requires_grad = not freeze


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--train-dir", default=SPEC_DIR)
    parser.add_argument("--val-dir", default=None,
                        help="Held-out validation spectrograms dir. "
                             "If not set, uses a random split of the train set.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== CoAtNet Keystroke Classifier ===")
    print(f"Device:     {device}")
    print(f"Epochs:     {args.epochs} ({args.freeze_epochs} frozen)")
    print(f"LR:         {args.lr}")
    print(f"Batch size: {args.batch_size}")

    if not os.path.exists(args.train_dir):
        print(f"\nError: {args.train_dir} not found. Run spectrogram.py first.")
        return

    if args.val_dir:
        if not os.path.exists(args.val_dir):
            print(f"\nError: --val-dir {args.val_dir} not found.")
            return

        train_full = datasets.ImageFolder(args.train_dir, transform=get_transforms(True))
        val_full = datasets.ImageFolder(args.val_dir, transform=get_transforms(False))

        if train_full.class_to_idx != val_full.class_to_idx:
            print("\nError: train/val class sets differ.")
            return

        class_to_idx = train_full.class_to_idx
        train_subset = train_full
        val_subset = val_full
        split_mode = f"held-out ({os.path.basename(args.val_dir)})"
    else:
        full_dataset = datasets.ImageFolder(args.train_dir, transform=get_transforms(True))
        val_dataset = datasets.ImageFolder(args.train_dir, transform=get_transforms(False))
        class_to_idx = full_dataset.class_to_idx

        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_indices, val_indices = random_split(
            range(len(full_dataset)), [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        split_mode = f"random {args.val_split:.0%} (leakage-prone, use --val-dir for a real number)"

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    print(f"Classes:    {len(class_to_idx)}")
    print(f"Val split:  {split_mode}")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train:      {len(train_subset)}")
    print(f"Val:        {len(val_subset)}\n")

    model = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # phase 1: only train the classifier head
    print("Freezing backbone...")
    set_parameter_requires_grad(model, freeze=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 55)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # phase 2: unfreeze and fine-tune with a lower LR
        if epoch == args.freeze_epochs + 1:
            print("\nUnfreezing backbone...")
            set_parameter_requires_grad(model, freeze=False)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(args.epochs - args.freeze_epochs),
            )

        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start

        marker = "*" if val_acc > best_val_acc else " "
        print(f"{marker} {epoch:5d}  {train_loss:10.4f}  {train_acc:8.2f}%  "
              f"{val_loss:8.4f}  {val_acc:6.2f}%  {elapsed:5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, "coatnet_keystrokes.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
            }, model_path)

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to:    {os.path.abspath(MODEL_DIR)}/coatnet_keystrokes.pth")
    print("Next step: python ../4_infer/infer.py")


if __name__ == "__main__":
    main()
