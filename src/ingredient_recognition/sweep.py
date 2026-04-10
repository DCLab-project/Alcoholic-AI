import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir() and (candidate / "assets").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.project_paths import INGREDIENT_CHECKPOINT_DIR, OUTPUTS_DIR, PROJECT_ROOT, to_project_relative


TRAIN_PY = PROJECT_ROOT / "src" / "ingredient_recognition" / "train.py"
CHECKPOINT_DIR = INGREDIENT_CHECKPOINT_DIR
RESULTS_DIR = OUTPUTS_DIR / "runs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "ingredient_sweep_results.csv"


STAGE1_EXPERIMENTS = [
    # memory-safe baseline
    {"model_name": "mobilenet_v3_small", "lr": 1e-3, "batch_size": 32, "epochs": 5, "num_workers": 0},
    {"model_name": "mobilenet_v3_small", "lr": 3e-4, "batch_size": 32, "epochs": 5, "num_workers": 0},

    # stronger MobileNet candidate
    {"model_name": "mobilenet_v3_large", "lr": 1e-3, "batch_size": 16, "epochs": 5, "num_workers": 0},
    {"model_name": "mobilenet_v3_large", "lr": 3e-4, "batch_size": 16, "epochs": 5, "num_workers": 0},

    # strong accuracy candidate at similar param scale to mnet-large
    {"model_name": "efficientnet_b0", "lr": 1e-3, "batch_size": 16, "epochs": 5, "num_workers": 0},
    {"model_name": "efficientnet_b0", "lr": 3e-4, "batch_size": 16, "epochs": 5, "num_workers": 0},

    # heavier comparator
    {"model_name": "resnet18", "lr": 1e-3, "batch_size": 8, "epochs": 5, "num_workers": 0},
    {"model_name": "resnet18", "lr": 3e-4, "batch_size": 8, "epochs": 5, "num_workers": 0},
]


def make_run_name(exp, stage):
    lr_str = str(exp["lr"]).replace(".", "p")
    return f"{stage}_{exp['model_name']}_lr{lr_str}_bs{exp['batch_size']}_ep{exp['epochs']}"


def append_result_row(row):
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage", "run_name", "model_name", "lr", "batch_size", "epochs",
                "num_workers", "status", "elapsed_sec", "best_epoch",
                "best_val_acc", "test_acc", "meta_path"
            ]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_meta(meta_path: Path):
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def run_experiment(exp, stage):
    run_name = make_run_name(exp, stage)
    meta_path = CHECKPOINT_DIR / f"{run_name}_meta.json"

    if meta_path.exists():
        meta = load_meta(meta_path)
        print(f"[INFO] skip existing: {run_name}")
        row = {
            "stage": stage,
            "run_name": run_name,
            "model_name": exp["model_name"],
            "lr": exp["lr"],
            "batch_size": exp["batch_size"],
            "epochs": exp["epochs"],
            "num_workers": exp["num_workers"],
            "status": "existing",
            "elapsed_sec": "",
            "best_epoch": meta.get("best_epoch", ""),
            "best_val_acc": meta.get("best_val_acc", ""),
            "test_acc": meta.get("test_acc", ""),
            "meta_path": to_project_relative(meta_path),
        }
        append_result_row(row)
        return row

    print("\n" + "=" * 100)
    print(f"[INFO] {stage}: {run_name}")
    print("=" * 100)

    cmd = [
        sys.executable, str(TRAIN_PY),
        "--model-name", exp["model_name"],
        "--lr", str(exp["lr"]),
        "--batch-size", str(exp["batch_size"]),
        "--epochs", str(exp["epochs"]),
        "--num-workers", str(exp["num_workers"]),
        "--device", "cuda",
        "--run-name", run_name,
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    t0 = time.time()
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    elapsed = time.time() - t0

    if completed.returncode != 0:
        row = {
            "stage": stage,
            "run_name": run_name,
            "model_name": exp["model_name"],
            "lr": exp["lr"],
            "batch_size": exp["batch_size"],
            "epochs": exp["epochs"],
            "num_workers": exp["num_workers"],
            "status": "failed",
            "elapsed_sec": f"{elapsed:.1f}",
            "best_epoch": "",
            "best_val_acc": "",
            "test_acc": "",
            "meta_path": to_project_relative(meta_path),
        }
        append_result_row(row)
        return row

    meta = load_meta(meta_path)
    if meta is None:
        row = {
            "stage": stage,
            "run_name": run_name,
            "model_name": exp["model_name"],
            "lr": exp["lr"],
            "batch_size": exp["batch_size"],
            "epochs": exp["epochs"],
            "num_workers": exp["num_workers"],
            "status": "missing_meta",
            "elapsed_sec": f"{elapsed:.1f}",
            "best_epoch": "",
            "best_val_acc": "",
            "test_acc": "",
            "meta_path": to_project_relative(meta_path),
        }
        append_result_row(row)
        return row

    row = {
        "stage": stage,
        "run_name": run_name,
        "model_name": exp["model_name"],
        "lr": exp["lr"],
        "batch_size": exp["batch_size"],
        "epochs": exp["epochs"],
        "num_workers": exp["num_workers"],
        "status": "ok",
        "elapsed_sec": f"{elapsed:.1f}",
        "best_epoch": meta.get("best_epoch", ""),
        "best_val_acc": meta.get("best_val_acc", ""),
        "test_acc": meta.get("test_acc", ""),
        "meta_path": to_project_relative(meta_path),
    }
    append_result_row(row)
    return row


def build_stage2_experiments(stage1_rows):
    ok_rows = [r for r in stage1_rows if r["status"] in ("ok", "existing") and r["best_val_acc"] != ""]
    ok_rows.sort(key=lambda r: float(r["best_val_acc"]), reverse=True)

    if len(ok_rows) == 0:
        return []

    stage2 = []

    # best stage1 config, longer run
    top1 = ok_rows[0]
    stage2.append({
        "model_name": top1["model_name"],
        "lr": float(top1["lr"]),
        "batch_size": int(top1["batch_size"]),
        "epochs": 10,
        "num_workers": int(top1["num_workers"]),
    })

    # best stage1 config, half lr, longer run
    stage2.append({
        "model_name": top1["model_name"],
        "lr": float(top1["lr"]) / 2.0,
        "batch_size": int(top1["batch_size"]),
        "epochs": 10,
        "num_workers": int(top1["num_workers"]),
    })

    # second-best stage1 config, longer run
    if len(ok_rows) > 1:
        top2 = ok_rows[1]
        stage2.append({
            "model_name": top2["model_name"],
            "lr": float(top2["lr"]),
            "batch_size": int(top2["batch_size"]),
            "epochs": 10,
            "num_workers": int(top2["num_workers"]),
        })

    return stage2


def main():
    print("[INFO] results csv:", RESULTS_CSV)

    stage1_rows = []
    for exp in STAGE1_EXPERIMENTS:
        row = run_experiment(exp, stage="stage1")
        stage1_rows.append(row)

    stage2_experiments = build_stage2_experiments(stage1_rows)
    if stage2_experiments:
        print("\n[INFO] stage2 experiments:")
        for exp in stage2_experiments:
            print(exp)

    for exp in stage2_experiments:
        run_experiment(exp, stage="stage2")

    print("\n[INFO] sweep complete")
    print(f"[INFO] results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
