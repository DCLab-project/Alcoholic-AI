from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from src.common.project_paths import INGREDIENT_CHECKPOINT_DIR, PROJECT_ROOT, resolve_project_path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def create_model(model_name: str, num_classes: int, pretrained: bool = False):
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_webcam_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_meta(meta_path: str | Path):
    resolved_meta_path = resolve_project_path(meta_path)
    meta = json.loads(resolved_meta_path.read_text(encoding="utf-8"))
    return resolved_meta_path, meta


def _checkpoint_candidates(meta_path: Path, meta: dict):
    candidates = []

    for key in ("best_checkpoint_path", "checkpoint_path", "weight_path"):
        value = meta.get(key)
        if not value:
            continue
        value_path = Path(value).expanduser()
        if value_path.is_absolute():
            candidates.append(value_path)
        else:
            candidates.append(meta_path.parent / value_path)
            candidates.append(PROJECT_ROOT / value_path)

    run_name = meta.get("run_name")
    if run_name:
        candidates.append(meta_path.with_name(f"{run_name}_best.pth"))
        candidates.append(INGREDIENT_CHECKPOINT_DIR / f"{run_name}_best.pth")

    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_checkpoint_path(meta_path: Path, meta: dict) -> Path:
    candidates = _checkpoint_candidates(meta_path, meta)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates) or "- <no checkpoint candidates>"
    raise FileNotFoundError(
        f"Could not locate checkpoint for meta file: {meta_path}\n"
        f"Searched:\n{searched}"
    )


def load_model_from_meta(meta_path: str | Path, device):
    resolved_meta_path, meta = load_meta(meta_path)

    labels = meta["labels"]
    num_classes = int(meta.get("num_classes", len(labels)))
    image_size = int(meta.get("image_size", 224))
    model_name = meta["model_name"]

    model = create_model(model_name, num_classes, pretrained=False)
    checkpoint_path = resolve_checkpoint_path(resolved_meta_path, meta)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, labels, meta, resolved_meta_path, checkpoint_path, image_size
