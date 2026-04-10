import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
for candidate in THIS_FILE.parents:
    if (candidate / "src").is_dir() and (candidate / "assets").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from src.common.project_paths import EXTERNAL_DATA_DIR, INGREDIENT_DATASET_DIR


MC_ROOT = EXTERNAL_DATA_DIR / "multi_class_food_image_dataset"
GS_ROOT = EXTERNAL_DATA_DIR / "grocery_store_dataset" / "GroceryStoreDataset" / "dataset"
OUT_ROOT = INGREDIENT_DATASET_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABELS = [
    "bacon", "bread", "broccoli", "butter", "carrot", "cheese", "chicken",
    "cucumber", "egg", "fish", "lettuce", "milk", "onion", "pepper", "potato",
    "sausage", "spinach", "tomato", "yogurt", "eggplant", "cabbage", "garlic",
    "ginger", "leek", "mushroom", "zucchini"
]

MC_MAP = {
    "bacon": "bacon",
    "bread": "bread",
    "broccoli": "broccoli",
    "butter": "butter",
    "carrots": "carrot",
    "cheese": "cheese",
    "chicken": "chicken",
    "cucumber": "cucumber",
    "eggs": "egg",
    "fish": "fish",
    "lettuce": "lettuce",
    "milk": "milk",
    "onions": "onion",
    "peppers": "pepper",
    "potatoes": "potato",
    "sausages": "sausage",
    "spinach": "spinach",
    "tomato": "tomato",
    "yogurt": "yogurt",
}

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for label in LABELS:
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)

def clear_output():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    ensure_dirs()

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS

def safe_copy(src: Path, dst_dir: Path, prefix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem.replace(" ", "_").replace("/", "_")
    dst = dst_dir / f"{prefix}_{stem}{src.suffix.lower()}"
    idx = 1
    while dst.exists():
        dst = dst_dir / f"{prefix}_{stem}_{idx}{src.suffix.lower()}"
        idx += 1
    shutil.copy2(src, dst)

def import_multi_class_food():
    counts = defaultdict(int)

    for split in ["train", "val", "test"]:
        split_dir = MC_ROOT / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.lower()
            if class_name not in MC_MAP:
                continue

            target_label = MC_MAP[class_name]
            dst_dir = OUT_ROOT / split / target_label

            for img in class_dir.iterdir():
                if is_image_file(img):
                    safe_copy(img, dst_dir, prefix="mc")
                    counts[(split, target_label)] += 1

    return counts

def map_grocery_row(rel_path: str, fine_name: str, coarse_name: str):
    path = rel_path.lower()
    fine = fine_name.lower()
    coarse = coarse_name.lower()

    if "eggplant" in fine or "aubergine" in fine or "eggplant" in path:
        return "eggplant"
    if "cabbage" in fine or "cabbage" in path:
        return "cabbage"
    if "garlic" in fine or "garlic" in path:
        return "garlic"
    if "ginger" in fine or "ginger" in path:
        return "ginger"
    if "leek" in fine or "leek" in path:
        return "leek"
    if "zucchini" in fine or "zucchini" in path or "courgette" in fine:
        return "zucchini"
    if "mushroom" in fine or "mushroom" in coarse or "mushroom" in path:
        return "mushroom"

    if "tomato" in fine or "tomato" in coarse or "tomato" in path:
        return "tomato"
    if "carrot" in fine or "carrot" in coarse or "carrot" in path:
        return "carrot"
    if "onion" in fine or "onion" in coarse or "onion" in path:
        return "onion"
    if "milk" in fine or "milk" in coarse or "/milk/" in path:
        return "milk"
    if "yoghurt" in fine or "yoghurt" in coarse or "yogurt" in fine or "/yoghurt/" in path:
        return "yogurt"
    if "pepper" in fine or "pepper" in coarse or "pepper" in path:
        return "pepper"
    if "cucumber" in fine or "cucumber" in coarse or "cucumber" in path:
        return "cucumber"
    if "broccoli" in fine or "broccoli" in coarse or "broccoli" in path:
        return "broccoli"
    if "lettuce" in fine or "lettuce" in coarse or "lettuce" in path:
        return "lettuce"
    if "cheese" in fine or "cheese" in coarse or "cheese" in path:
        return "cheese"
    if "butter" in fine or "butter" in coarse or "butter" in path:
        return "butter"
    if "potato" in fine or "potato" in coarse or "potato" in path:
        return "potato"

    return None

def load_classes_csv():
    classes_csv = GS_ROOT / "classes.csv"
    id_to_names = {}

    with classes_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fine_id = int(row["Class ID (int)"])
            fine_name = row["Class Name (str)"]
            coarse_name = row["Coarse Class Name (str)"]
            id_to_names[fine_id] = (fine_name, coarse_name)

    return id_to_names

def import_grocery_store():
    counts = defaultdict(int)
    id_to_names = load_classes_csv()

    for split_txt_name, out_split in [("train.txt", "train"), ("val.txt", "val"), ("test.txt", "test")]:
        split_txt = GS_ROOT / split_txt_name
        if not split_txt.exists():
            continue

        lines = split_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

        for line in lines:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue

            rel_path = parts[0]
            fine_id = int(parts[1])

            fine_name, coarse_name = id_to_names[fine_id]
            target_label = map_grocery_row(rel_path, fine_name, coarse_name)

            if target_label is None:
                continue

            src = GS_ROOT / rel_path
            if not src.exists():
                continue

            dst_dir = OUT_ROOT / out_split / target_label
            safe_copy(src, dst_dir, prefix="gs")
            counts[(out_split, target_label)] += 1

    return counts

def summarize(mc_counts, gs_counts):
    print("\n=== BUILD SUMMARY ===")
    total_by_split = defaultdict(int)
    total_by_label = defaultdict(int)

    for (split, label), n in mc_counts.items():
        total_by_split[split] += n
        total_by_label[label] += n
    for (split, label), n in gs_counts.items():
        total_by_split[split] += n
        total_by_label[label] += n

    print("\n[total by split]")
    for split in ["train", "val", "test"]:
        print(f"{split}: {total_by_split[split]}")

    print("\n[total by label]")
    for label in LABELS:
        print(f"{label}: {total_by_label[label]}")

    print("\n[multi-class contribution]")
    for split in ["train", "val", "test"]:
        print(f"\n{split}")
        for label in LABELS:
            n = mc_counts.get((split, label), 0)
            if n > 0:
                print(f"  {label}: {n}")

    print("\n[grocery-store contribution]")
    for split in ["train", "val", "test"]:
        print(f"\n{split}")
        for label in LABELS:
            n = gs_counts.get((split, label), 0)
            if n > 0:
                print(f"  {label}: {n}")

def main():
    print("[INFO] clearing output directory")
    clear_output()

    print("[INFO] importing Multi-class Food dataset")
    mc_counts = import_multi_class_food()

    print("[INFO] importing Grocery Store dataset")
    gs_counts = import_grocery_store()

    summarize(mc_counts, gs_counts)
    print("\n[INFO] done")

if __name__ == "__main__":
    main()
