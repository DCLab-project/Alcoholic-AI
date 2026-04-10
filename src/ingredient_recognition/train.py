import argparse
import gc
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir() and (candidate / "assets").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.ingredient_models import build_eval_transform, build_train_transform, create_model, is_image_file, load_labels
from src.common.project_paths import (
    INGREDIENT_CHECKPOINT_DIR,
    INGREDIENT_DATASET_DIR,
    INGREDIENT_LABELS_FILE,
    resolve_project_path,
    to_project_relative,
)


class IngredientDataset(Dataset):
    def __init__(self, data_root: Path, split: str, labels, transform=None):
        self.root = data_root / split
        self.labels = labels
        self.transform = transform
        self.samples = []
        self.class_to_idx = {label: i for i, label in enumerate(labels)}

        for label in labels:
            class_dir = self.root / label
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if is_image_file(img_path):
                    self.samples.append((img_path, self.class_to_idx[label]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in split={split} for labels={labels}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_dataset(data_root: Path, split: str, labels, image_size: int):
    tfm = {
        "train": build_train_transform(image_size=image_size),
        "val": build_eval_transform(image_size=image_size),
        "test": build_eval_transform(image_size=image_size),
    }[split]

    return IngredientDataset(data_root=data_root, split=split, labels=labels, transform=tfm)


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total


def evaluate(model, loader, criterion, device, amp_enabled):
    model.eval()
    running_loss = 0.0
    correct_sum = 0
    total_sum = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            correct, total = accuracy(outputs, targets)
            correct_sum += correct
            total_sum += total

    avg_loss = running_loss / max(total_sum, 1)
    avg_acc = correct_sum / max(total_sum, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_small")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--data-root", type=str, default=str(INGREDIENT_DATASET_DIR))
    parser.add_argument("--labels-file", type=str, default=str(INGREDIENT_LABELS_FILE))
    parser.add_argument("--checkpoint-dir", type=str, default=str(INGREDIENT_CHECKPOINT_DIR))
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    data_root = resolve_project_path(args.data_root)
    labels_file = resolve_project_path(args.labels_file)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)

    labels = load_labels(labels_file)
    num_classes = len(labels)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"{args.model_name}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    safe_run_name = run_name.replace("/", "_").replace(" ", "_")

    print("[INFO] run_name:", safe_run_name)
    print("[INFO] model_name:", args.model_name)
    print("[INFO] data_root:", data_root)
    print("[INFO] labels_file:", labels_file)
    print("[INFO] checkpoint_dir:", checkpoint_dir)
    print("[INFO] labels:", labels)
    print("[INFO] num_classes:", num_classes)
    print("[INFO] device:", device)
    print("[INFO] amp_enabled:", amp_enabled)

    train_ds = build_dataset(data_root, "train", labels, args.image_size)
    val_ds = build_dataset(data_root, "val", labels, args.image_size)
    test_ds = build_dataset(data_root, "test", labels, args.image_size)

    print(f"[INFO] train size: {len(train_ds)}")
    print(f"[INFO] val size:   {len(val_ds)}")
    print(f"[INFO] test size:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = create_model(args.model_name, num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_acc = 0.0
    best_epoch = 0
    best_path = checkpoint_dir / f"{safe_run_name}_best.pth"
    meta_path = checkpoint_dir / f"{safe_run_name}_meta.json"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        correct_sum = 0
        total_sum = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct, total = accuracy(outputs, targets)
            correct_sum += correct
            total_sum += total

        scheduler.step()

        train_loss = running_loss / max(total_sum, 1)
        train_acc = correct_sum / max(total_sum, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, amp_enabled)
        elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"[EPOCH {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            meta = {
                "run_name": safe_run_name,
                "labels": labels,
                "num_classes": num_classes,
                "model_name": args.model_name,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "image_size": args.image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "amp_enabled": amp_enabled,
                "data_root": to_project_relative(data_root),
                "labels_file": to_project_relative(labels_file),
                "best_checkpoint_path": to_project_relative(best_path),
                "meta_path": to_project_relative(meta_path),
                "history": history,
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[INFO] saved best checkpoint -> {best_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"[INFO] best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print("[INFO] loading best checkpoint for test evaluation")

    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, amp_enabled)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    final_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    final_meta["test_loss"] = test_loss
    final_meta["test_acc"] = test_acc
    meta_path.write_text(json.dumps(final_meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
