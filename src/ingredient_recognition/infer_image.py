import argparse
import sys
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir() and (candidate / "assets").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.ingredient_models import build_eval_transform, load_model_from_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True, help="path to *_meta.json")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, labels, meta, meta_path, weight_path, image_size = load_model_from_meta(args.meta, device)
    tfm = build_eval_transform(image_size=image_size)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    topk = min(args.topk, len(labels))
    values, indices = torch.topk(probs, k=topk)

    print(f"[INFO] model_name: {meta['model_name']}")
    print(f"[INFO] run_name:   {meta['run_name']}")
    print(f"[INFO] meta:       {meta_path}")
    print(f"[INFO] weight:     {weight_path}")
    print(f"[INFO] image:      {image_path}")

    for rank, (v, i) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {labels[i]}  prob={v:.4f}")


if __name__ == "__main__":
    main()
