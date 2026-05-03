"""
Verify Mask → Bbox Extraction Quality

Loads random defect samples from MVTec + VisA, overlays extracted bboxes
on images, and saves to output folder for visual inspection.

Run BEFORE running yolo_dataset_converter.py to verify bbox quality.

Usage:
    python src/verify_mask_bbox_overlay.py \
        --mvtec /path/to/MVTec_AD \
        --visa  /path/to/VisA \
        --output data/verify_overlays \
        --count 20
"""
import os
import cv2
import random
import argparse
from pathlib import Path

from data_processing import (
    load_mvtec_dataset,
    load_visa_dataset,
    process_mask_to_bbox_and_location,
    get_combined_bbox,
    normalize_bbox,
)
from yolo_dataset_converter import xyxy_to_yolo


def get_image_size(image_path: str):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return img.shape[1], img.shape[0]  # (w, h)
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None, None


def overlay_and_save(sample: dict, output_path: str) -> bool:
    """
    Draw bbox overlay on image and save.
    Returns True if saved successfully.
    """
    image_path = sample["image_path"]
    mask_path  = sample.get("mask_path") or sample.get("mask")

    if not image_path or not os.path.exists(image_path):
        return False

    w, h = get_image_size(image_path)
    if w is None:
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    overlay = img.copy()

    # Get bbox from mask
    if mask_path and os.path.exists(mask_path):
        result = process_mask_to_bbox_and_location(mask_path, image_size=(w, h))
        bboxes = result.get("normalized_bboxes", [])
    else:
        bboxes = []

    n_drawn = 0
    if bboxes:
        # Draw each individual bbox (from contours)
        for bbox in bboxes:
            x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n_drawn += 1

        # Draw combined bbox (red, thicker)
        combined = get_combined_bbox([(int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)) for b in bboxes])
        if combined:
            cx1, cy1, cx2, cy2 = combined
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

            # Add label
            label = f"defect  n={n_drawn}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (cx1, max(0, cy1 - th - 8)), (cx1 + tw, cy1), (0, 0, 255), -1)
            cv2.putText(overlay, label, (cx1, max(0, cy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # No bbox found — just mark as "no detection"
        label = "NO BBOX FROM MASK"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(overlay, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add info text
    info = f"{sample.get('source','?')} | {sample['category']} | {sample['defect_type']}"
    cv2.putText(overlay, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    success = cv2.imwrite(output_path, overlay)
    return success


def main():
    parser = argparse.ArgumentParser(description="Verify mask → bbox extraction quality")
    parser.add_argument("--mvtec",  type=str, default="",  help="Path to MVTec AD root")
    parser.add_argument("--visa",   type=str, default="",  help="Path to VisA root")
    parser.add_argument("--output", type=str, default="data/verify_overlays", help="Output folder")
    parser.add_argument("--count",  type=int, default=20,  help="Total overlay images to generate")
    parser.add_argument("--seed",   type=int, default=42,   help="Random seed")
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Sample equally from each category instead of random"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect defect samples
    defect_samples = []

    if args.mvtec and os.path.isdir(args.mvtec):
        print(f"[MVTec] Loading from {args.mvtec}...")
        items = load_mvtec_dataset(args.mvtec, include_good=False, split="test")
        for item in items:
            if not item["is_good"] and item["mask_path"]:
                defect_samples.append({**item, "source": "mvtec"})

    if args.visa and os.path.isdir(args.visa):
        print(f"[VisA] Loading from {args.visa}...")
        items = load_visa_dataset(args.visa)
        for item in items:
            if not item["is_good"] and item.get("mask_path"):
                defect_samples.append({**item, "source": "visa"})

    print(f"[Total] {len(defect_samples)} defect samples found")
    if not defect_samples:
        print("No defect samples found. Check paths.")
        return

    # Sample
    if args.per_category:
        # Sample equally from each (source, category)
        from collections import defaultdict
        groups = defaultdict(list)
        for s in defect_samples:
            key = (s["source"], s["category"])
            groups[key].append(s)
        sampled = []
        per_cat = max(1, args.count // len(groups))
        for key, group in groups.items():
            sampled.extend(random.sample(group, min(per_cat, len(group))))
    else:
        sampled = random.sample(defect_samples, min(args.count, len(defect_samples)))

    # Generate overlays
    print(f"[Generate] {len(sampled)} overlays → {output_dir}")
    saved = 0
    failed = []

    for i, sample in enumerate(sampled):
        fname = f"{i:03d}_{sample['source']}_{sample['category']}_{sample['defect_type']}_{sample['image_name'][:20]}.jpg"
        out_path = output_dir / fname
        ok = overlay_and_save(sample, str(out_path))
        if ok:
            saved += 1
        else:
            failed.append(sample["image_path"])

    print(f"\n{'='*50}")
    print(f"Done: {saved}/{len(sampled)} overlays saved to:")
    print(f"  {output_dir.absolute()}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed[:3]:
            print(f"  - {f}")
    print(f"\nCheck overlays:")
    print(f"  - GREEN boxes  = individual contour bboxes")
    print(f"  - RED box     = combined (encompassing) bbox")
    print(f"  - NO BOX      = mask has no contours above threshold")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
