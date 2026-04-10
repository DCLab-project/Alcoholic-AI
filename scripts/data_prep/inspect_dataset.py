import argparse
import csv
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS

def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if is_image_file(p))

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def shallow_tree(root: Path, max_depth: int = 3):
    print_header(f"[TREE] shallow tree under: {root}")
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        kind = "[D]" if path.is_dir() else "[F]"
        print(f"{indent}{kind} {rel}")

def summarize_leaf_image_dirs(root: Path, limit: int = 50):
    print_header("[SUMMARY] directories that directly contain image files")
    rows = []
    for d in sorted([p for p in root.rglob("*") if p.is_dir()]):
        imgs = [x for x in d.iterdir() if is_image_file(x)]
        if imgs:
            rows.append((str(d.relative_to(root)), len(imgs)))

    if not rows:
        print("No directories with direct image files found.")
        return

    rows.sort(key=lambda x: (-x[1], x[0]))
    for rel, cnt in rows[:limit]:
        print(f"{cnt:5d}  {rel}")

def summarize_by_immediate_child(root: Path):
    print_header("[SUMMARY] image counts by immediate child directory")
    counter = Counter()

    for p in root.rglob("*"):
        if not is_image_file(p):
            continue
        rel = p.relative_to(root)
        key = rel.parts[0] if len(rel.parts) > 0 else "."
        counter[key] += 1

    if not counter:
        print("No images found.")
        return

    for name, cnt in counter.most_common():
        print(f"{cnt:5d}  {name}")

def inspect_split_file(path: Path, preview_n: int = 5):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"{path.name}: {len(lines)} lines")
    for line in lines[:preview_n]:
        print(f"  {line}")

def inspect_classes_csv(path: Path, preview_n: int = 10):
    print_header(f"[CSV] {path}")
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Empty CSV")
        return

    header = rows[0]
    data = rows[1:]
    print("header:", header)
    print("rows:", len(data))

    for row in data[:preview_n]:
        print(" ", row)

def inspect_grocery_store(root: Path):
    print_header("[DETECTED] GroceryStore-style dataset")

    candidate_csv = list(root.rglob("classes.csv"))
    candidate_train = list(root.rglob("train.txt"))
    candidate_val = list(root.rglob("val.txt"))
    candidate_test = list(root.rglob("test.txt"))

    print("classes.csv:", [str(p.relative_to(root)) for p in candidate_csv])
    print("train.txt  :", [str(p.relative_to(root)) for p in candidate_train])
    print("val.txt    :", [str(p.relative_to(root)) for p in candidate_val])
    print("test.txt   :", [str(p.relative_to(root)) for p in candidate_test])

    if candidate_csv:
        inspect_classes_csv(candidate_csv[0])

    print_header("[SPLITS]")
    if candidate_train:
        inspect_split_file(candidate_train[0])
    if candidate_val:
        inspect_split_file(candidate_val[0])
    if candidate_test:
        inspect_split_file(candidate_test[0])

def detect_dataset_type(root: Path) -> str:
    names = {p.name.lower() for p in root.rglob("*") if p.is_file()}
    if {"classes.csv", "train.txt", "val.txt", "test.txt"}.issubset(names):
        return "grocery_store"
    return "generic_image_dataset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="dataset root path")
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"[ERROR] path does not exist: {root}")

    print_header("[BASIC INFO]")
    print("root        :", root.resolve())
    print("exists      :", root.exists())
    print("is_dir      :", root.is_dir())
    print("total_images:", count_images(root))
    print("dataset_type:", detect_dataset_type(root))

    shallow_tree(root, max_depth=args.max_depth)
    summarize_by_immediate_child(root)
    summarize_leaf_image_dirs(root)

    dtype = detect_dataset_type(root)
    if dtype == "grocery_store":
        inspect_grocery_store(root)

if __name__ == "__main__":
    main()
