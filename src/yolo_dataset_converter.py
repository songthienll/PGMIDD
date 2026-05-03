"""
YOLO Dataset Converter — MVTec AD + VisA → YOLO format

Converts MVTec AD + VisA datasets to Ultralytics YOLO format.
Extracts bboxes from masks, splits 80/10/10 (train/val/test), generates data.yaml.

Usage:
    python src/yolo_dataset_converter.py --mvtec /path/to/mvtec_ad --visa /path/to/VisA --output data/yolo
"""
import os
import json
import random
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from data_processing import (
    load_mvtec_dataset,
    load_visa_dataset,
    process_mask_to_bbox_and_location,
)

# =============================================================================
# CLASS CONFIG — class-agnostic (all defects = class 0)
# =============================================================================
YOLO_CLASS_NAMES = ["defect"]       # class-agnostic
YOLO_CLASS_ID    = 0                 # all defects share same class id

# Train/val/test ratios
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# YOLO FORMAT HELPERS
# =============================================================================

def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """
    Convert normalized xyxy → normalized cx, cy, w, h (YOLO format).
    All values in [0, 1].
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    return cx, cy, w, h


def write_yolo_label(txt_path: str, bboxes: List[List[float]], class_id: int = YOLO_CLASS_ID):
    """
    Write YOLO label file.

    Args:
        txt_path: Path to .txt label file
        bboxes: List of [cx, cy, w, h] normalized bboxes
        class_id: Class id (default 0 = defect)
    """
    with open(txt_path, "w") as f:
        if not bboxes:
            # Good sample: write empty file (no objects)
            pass
        else:
            for cx, cy, w, h in bboxes:
                # Clamp to [0, 1] to handle edge cases
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get (width, height) from image file."""
    try:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    # Fallback: try PIL
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size  # (w, h)
    except Exception:
        return None, None


# =============================================================================
# DATASET LOADING + BBOX EXTRACTION
# =============================================================================

def load_all_samples(mvtec_root: str, visa_root: str) -> List[Dict]:
    """
    Load all MVTec + VisA samples with bbox extraction.
    Returns list of dicts with image_path, is_good, category, defect_type, bboxes.
    """
    all_samples = []

    # MVTec
    if mvtec_root and os.path.isdir(mvtec_root):
        print(f"[MVTec] Loading from {mvtec_root}...")
        mvtec_items = load_mvtec_dataset(mvtec_root, include_good=True, split="both")
        for item in mvtec_items:
            sample = {
                "source": "mvtec",
                "category": item["category"],
                "defect_type": item["defect_type"],
                "is_good": item["is_good"],
                "image_path": item["image_path"],
                "mask_path": item["mask_path"],
                "split": item.get("split", "test"),
            }

            # Extract bbox from mask for defect samples
            if not item["is_good"] and item["mask_path"] and os.path.exists(item["mask_path"]):
                w, h = get_image_size(item["image_path"])
                if w and h:
                    result = process_mask_to_bbox_and_location(item["mask_path"], image_size=(w, h))
                    sample["bboxes"] = result.get("normalized_bboxes", [])
                    sample["combined_bbox"] = result.get("normalized_combined_bbox", None)
                else:
                    sample["bboxes"] = []
                    sample["combined_bbox"] = None
            else:
                sample["bboxes"] = []
                sample["combined_bbox"] = None

            all_samples.append(sample)

        print(f"  → {len(mvtec_items)} samples (good={sum(1 for s in all_samples if s['source']=='mvtec' and s['is_good'])}, defect={sum(1 for s in all_samples if s['source']=='mvtec' and not s['is_good'])})")

    # VisA
    if visa_root and os.path.isdir(visa_root):
        print(f"[VisA] Loading from {visa_root}...")
        visa_items = load_visa_dataset(visa_root)
        for item in visa_items:
            sample = {
                "source": "visa",
                "category": item["category"],
                "defect_type": item["defect_type"],
                "is_good": item["is_good"],
                "image_path": item["image_path"],
                "mask_path": item.get("mask_path"),
                "split": "test",  # VisA has no explicit train/test split
            }

            if not item["is_good"] and item.get("mask_path") and os.path.exists(item.get("mask_path", "")):
                w, h = get_image_size(item["image_path"])
                if w and h:
                    result = process_mask_to_bbox_and_location(item["mask_path"], image_size=(w, h))
                    sample["bboxes"] = result.get("normalized_bboxes", [])
                    sample["combined_bbox"] = result.get("normalized_combined_bbox", None)
                else:
                    sample["bboxes"] = []
                    sample["combined_bbox"] = None
            else:
                sample["bboxes"] = []
                sample["combined_bbox"] = None

            all_samples.append(sample)

        mvtec_count = sum(1 for s in all_samples if s["source"] == "mvtec")
        visa_count  = len(all_samples) - mvtec_count
        print(f"  → {visa_count} VisA samples (good={sum(1 for s in all_samples if s['source']=='visa' and s['is_good'])}, defect={sum(1 for s in all_samples if s['source']=='visa' and not s['is_good'])})")

    print(f"[Total] {len(all_samples)} samples loaded")
    return all_samples


# =============================================================================
# STRATIFIED SPLIT — 80/10/10
# =============================================================================

def stratified_split_80_10_10(samples: List[Dict], seed: int = RANDOM_SEED) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split by (source, category, is_good) groups.
    80% train / 10% val / 10% test.
    """
    random.seed(seed)

    # Group by (source, category, is_good)
    groups = defaultdict(list)
    for sample in samples:
        key = (sample["source"], sample["category"], sample["is_good"])
        groups[key].append(sample)

    train_samples = []
    val_samples   = []
    test_samples  = []

    for key, group in groups.items():
        shuffled = group.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train:n_train + n_val])
        test_samples.extend(shuffled[n_train + n_val:])

    # Assign split label
    for s in train_samples: s["new_split"] = "train"
    for s in val_samples:   s["new_split"] = "val"
    for s in test_samples:  s["new_split"] = "test"

    # Shuffle each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    print(f"[Split] train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    return train_samples, val_samples, test_samples


# =============================================================================
# EXPORT TO YOLO FORMAT
# =============================================================================

def export_to_yolo(
    samples: List[Dict],
    output_dir: str,
    copy_images: bool = True,
    symlink_images: bool = False,
) -> Dict:
    """
    Export samples to YOLO folder structure.

    output_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

    Args:
        samples: List of sample dicts with new_split, image_path, bboxes
        output_dir: Root output directory
        copy_images: Copy images to output dir (slow but portable)
        symlink_images: Use symlinks instead of copy

    Returns:
        stats dict with counts per split
    """
    output_dir = Path(output_dir)

    # Create dirs
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0}
    errors = []

    for sample in samples:
        split = sample.get("new_split", "test")
        src_image = sample["image_path"]

        # Destination paths
        rel_path = Path(src_image).name
        dst_image = output_dir / "images" / split / rel_path
        dst_label = output_dir / "labels" / split / (Path(src_image).stem + ".txt")

        # Copy/symlink image
        if copy_images:
            try:
                shutil.copy2(src_image, dst_image)
            except Exception as e:
                errors.append(f"Failed to copy {src_image}: {e}")
                continue
        elif symlink_images:
            try:
                if not dst_image.exists():
                    os.symlink(src_image, dst_image)
            except Exception as e:
                errors.append(f"Failed to symlink {src_image}: {e}")

        # Convert bboxes to YOLO format and write label
        yolo_bboxes = []
        for bbox in sample.get("bboxes", []):
            # bbox is [x1, y1, x2, y2] normalized
            cx, cy, w, h = xyxy_to_yolo(bbox[0], bbox[1], bbox[2], bbox[3])
            yolo_bboxes.append([cx, cy, w, h])

        write_yolo_label(str(dst_label), yolo_bboxes)
        stats[split] += 1

    # Write data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_content = f"""# YOLO Dataset — MVTec AD + VisA
# Class-agnostic: all defect types = class 0 ("defect")
# Split: 80% train / 10% val / 10% test

path: {output_dir.absolute()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: defect
"""
    data_yaml.write_text(yaml_content, encoding="utf-8")

    if errors:
        print(f"[Warning] {len(errors)} file errors during export")
        for e in errors[:5]:
            print(f"  - {e}")

    return stats


# =============================================================================
# VERIFICATION — visualize random sample bboxes
# =============================================================================

def verify_sample_overlay(
    sample: Dict,
    output_path: str = None,
    max_samples: int = 5,
    counter: dict = None,
) -> int:
    """
    Overlay bboxes on image and save for verification.
    Returns how many images were saved.
    """
    if counter is None:
        counter = {"count": 0}

    if counter["count"] >= max_samples:
        return 0

    if sample["is_good"]:
        return 0

    image_path = sample["image_path"]
    bboxes = sample.get("bboxes", [])

    if not bboxes:
        return 0

    if not os.path.exists(image_path):
        return 0

    img = cv2.imread(image_path)
    if img is None:
        return 0

    h, w = img.shape[:2]
    overlay = img.copy()

    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, overlay)
        counter["count"] += 1

    return 1


def verify_export(output_dir: str, sample_count: int = 20) -> Dict:
    """
    Spot-check exported labels — load a few, verify format.
    """
    output_dir = Path(output_dir)
    issues = []

    for split in ["train", "val", "test"]:
        label_dir = output_dir / "labels" / split
        image_dir = output_dir / "images" / split

        if not label_dir.exists():
            continue

        txt_files = list(label_dir.glob("*.txt"))[:sample_count]
        for txt in txt_files:
            img_file = image_dir / (txt.stem + ".png")
            if not img_file.exists() and not list(image_dir.glob(txt.stem + ".*")):
                issues.append(f"Missing image for label: {txt.name}")
                continue

            with open(txt, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"Malformed line in {txt.name}: {line.strip()}")
                    continue
                try:
                    cls = float(parts[0])
                    cx, cy, w, h = [float(x) for x in parts[1:]]
                    if not (0 <= cls <= 1 and all(0 <= v <= 1 for v in [cx, cy, w, h])):
                        issues.append(f"Out-of-range values in {txt.name}: {line.strip()}")
                except ValueError:
                    issues.append(f"Non-numeric values in {txt.name}: {line.strip()}")

    return {
        "total_checked": sample_count * 3,
        "issues": issues,
        "ok": len(issues) == 0,
    }


# =============================================================================
# MAIN — build YOLO dataset from MVTec + VisA
# =============================================================================

def build_yolo_dataset(
    mvtec_root: str,
    visa_root: str,
    output_dir: str,
    copy_images: bool = True,
    verify: bool = True,
    max_verify_samples: int = 5,
) -> Dict:
    """
    End-to-end: load → extract bboxes → split → export → verify.
    """
    print("=" * 60)
    print("YOLO Dataset Converter — MVTec AD + VisA")
    print("=" * 60)

    # Step 1: Load all samples
    print("\n[Step 1] Loading datasets...")
    samples = load_all_samples(mvtec_root, visa_root)

    if not samples:
        raise RuntimeError("No samples loaded. Check MVTec/VisA paths.")

    # Step 2: Stratified split 80/10/10
    print("\n[Step 2] Splitting 80/10/10 (stratified by source/category/is_good)...")
    train_samples, val_samples, test_samples = stratified_split_80_10_10(samples)
    all_split = train_samples + val_samples + test_samples

    # Step 3: Export to YOLO format
    print("\n[Step 3] Exporting to YOLO format...")
    print(f"  Copy images: {copy_images}")
    print(f"  Output: {output_dir}")
    stats = export_to_yolo(all_split, output_dir, copy_images=copy_images)

    # Step 4: Verification
    if verify:
        print("\n[Step 4] Verifying export...")
        verify_result = verify_export(output_dir)

        # Generate verification overlays for random defect samples
        verify_dir = Path(output_dir) / "verify_overlays"
        verify_dir.mkdir(exist_ok=True)
        defect_samples = [s for s in all_split if not s["is_good"]]
        random.seed(RANDOM_SEED)
        sampled = random.sample(defect_samples, min(max_verify_samples, len(defect_samples)))
        counter = {"count": 0}
        for i, s in enumerate(sampled):
            out_path = verify_dir / f"verify_{i}_{Path(s['image_path']).name}"
            verify_sample_overlay(s, str(out_path), max_verify_samples, counter)

        print(f"  Format verification: {'PASS' if verify_result['ok'] else 'ISSUES FOUND'}")
        if not verify_result["ok"]:
            for issue in verify_result["issues"][:5]:
                print(f"    - {issue}")

        if counter["count"] > 0:
            print(f"  Verification overlays saved to: {verify_dir}/")

    # Step 5: Summary
    total = stats["train"] + stats["val"] + stats["test"]
    print("\n" + "=" * 60)
    print(f"DONE — {total} samples exported to {output_dir}")
    print(f"  train: {stats['train']} ({stats['train']/total*100:.1f}%)")
    print(f"  val:   {stats['val']} ({stats['val']/total*100:.1f}%)")
    print(f"  test:  {stats['test']} ({stats['test']/total*100:.1f}%)")
    print(f"  data.yaml: {(Path(output_dir)/'data.yaml').absolute()}")
    print("=" * 60)

    return {
        "total": total,
        "stats": stats,
        "output_dir": output_dir,
        "data_yaml": str(Path(output_dir) / "data.yaml"),
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MVTec AD + VisA to YOLO format")
    parser.add_argument("--mvtec",  type=str, default="",  help="Path to MVTec AD root")
    parser.add_argument("--visa",   type=str, default="",  help="Path to VisA root")
    parser.add_argument("--output", type=str, default="data/yolo", help="Output directory")
    parser.add_argument("--no-copy", action="store_true",  help="Don't copy images (use symlinks or skip)")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--seed",  type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--verify-samples", type=int, default=5,
        help="Number of verification overlay images to generate",
    )
    args = parser.parse_args()

    RANDOM_SEED = args.seed

    result = build_yolo_dataset(
        mvtec_root=args.mvtec,
        visa_root=args.visa,
        output_dir=args.output,
        copy_images=not args.no_copy,
        verify=not args.no_verify,
        max_verify_samples=args.verify_samples,
    )
    print(json.dumps(result, indent=2, default=str))