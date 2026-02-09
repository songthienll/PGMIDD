"""
Data Processing Module for MVTec AD Dataset
Handles dataset loading, bbox extraction, normalization, and location mapping.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
# =============================================================================
# MVTEC AD DATASET LOADING
# =============================================================================
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper'
]
def load_mvtec_dataset(root_path: str, 
                       include_good: bool = False,
                       categories: List[str] = None,
                       split: str = "both") -> List[Dict]:
    """
    Load MVTec AD dataset structure.
    
    Args:
        root_path: Path to MVTec AD root directory
        include_good: Whether to include 'good' (non-defective) samples
        categories: List of categories to load (None = all)
        split: Which split to load - 'train', 'test', or 'both' (default: 'both')
        
    Returns:
        List of dicts with image info
        
    Note:
        - Train split: ~3629 good samples (no defects)
        - Test split: ~1725 samples (good + defective)
        - Both: ~5354 total samples
    """
    root = Path(root_path)
    dataset = []
    
    if categories is None:
        categories = MVTEC_CATEGORIES
    
    for category in categories:
        cat_path = root / category
        if not cat_path.exists():
            continue
        
        # Train images (only good samples, no masks)
        if split in ["train", "both"]:
            train_path = cat_path / "train" / "good"
            if train_path.exists():
                for img_file in train_path.iterdir():
                    if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                        continue
                    
                    if include_good:
                        dataset.append({
                            'image_path': str(img_file),
                            'mask_path': None,
                            'category': category,
                            'defect_type': 'good',
                            'image_name': img_file.stem,
                            'is_good': True,
                            'split': 'train'
                        })
        
        # Test images (good + defective samples)
        if split in ["test", "both"]:
            test_path = cat_path / "test"
            gt_path = cat_path / "ground_truth"
            
            if test_path.exists():
                for defect_type in os.listdir(test_path):
                    defect_path = test_path / defect_type
                    if not defect_path.is_dir():
                        continue
                    
                    is_good = defect_type == "good"
                    if is_good and not include_good:
                        continue
                    
                    for img_file in defect_path.iterdir():
                        if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                            continue
                        
                        # Get mask path (for defective samples)
                        mask_path = None
                        if not is_good:
                            mask_name = img_file.stem + "_mask.png"
                            mask_file = gt_path / defect_type / mask_name
                            if mask_file.exists():
                                mask_path = str(mask_file)
                        
                        dataset.append({
                            'image_path': str(img_file),
                            'mask_path': mask_path,
                            'category': category,
                            'defect_type': defect_type,
                            'image_name': img_file.stem,
                            'is_good': is_good,
                            'split': 'test'
                        })
    
    return dataset
# =============================================================================
# BBOX EXTRACTION AND PROCESSING
# =============================================================================
def mask_to_bbox(mask: np.ndarray, 
                 min_area_ratio: float = 0.0001) -> List[Tuple[int, int, int, int]]:
    """
    Convert binary mask to bounding boxes.
    
    Args:
        mask: Grayscale mask image (0=background, 255=defect)
        min_area_ratio: Minimum area as fraction of image (filters noise)
        
    Returns:
        List of bboxes as (x1, y1, x2, y2) in absolute pixel coordinates
    """
    if mask is None:
        return []
    
    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    total_area = mask.shape[0] * mask.shape[1]
    min_area = total_area * min_area_ratio
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area >= min_area:
            bboxes.append((x, y, x + w, y + h))
    
    return bboxes
def get_combined_bbox(bboxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Combine multiple bboxes into one encompassing bbox."""
    if not bboxes:
        return None
    
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    
    return (x1, y1, x2, y2)
def normalize_bbox(bbox: Tuple[int, int, int, int], 
                   image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Convert absolute bbox to normalized [0,1] range.
    
    Args:
        bbox: (x1, y1, x2, y2) in absolute pixels
        image_size: (width, height) of image
        
    Returns:
        (x1, y1, x2, y2) normalized to [0, 1]
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox
    return (x1 / w, y1 / h, x2 / w, y2 / h)
def denormalize_bbox(bbox: Tuple[float, float, float, float],
                     image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Convert normalized bbox back to absolute pixels."""
    w, h = image_size
    x1, y1, x2, y2 = bbox
    return (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
def bbox_to_coco_format(bbox: Tuple[int, int, int, int]) -> List[int]:
    """Convert (x1, y1, x2, y2) to COCO format [x, y, width, height]."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]
def coco_to_bbox(coco_bbox: List[int]) -> Tuple[int, int, int, int]:
    """Convert COCO [x, y, w, h] to (x1, y1, x2, y2)."""
    x, y, w, h = coco_bbox
    return (x, y, x + w, y + h)
# =============================================================================
# DEFECT SIZE AND RATIO COMPUTATION
# =============================================================================
def compute_defect_ratio(mask: np.ndarray) -> float:
    """
    Compute defect area as fraction of total image area.
    
    Args:
        mask: Grayscale mask image
        
    Returns:
        Ratio in range [0, 1]
    """
    if mask is None:
        return 0.0
    
    binary = (mask > 127).astype(np.uint8)
    defect_area = np.sum(binary)
    total_area = mask.shape[0] * mask.shape[1]
    
    return defect_area / total_area if total_area > 0 else 0.0
def categorize_defect_size(ratio: float) -> str:
    """
    Categorize defect size based on area ratio.
    
    Args:
        ratio: Defect area / total area
        
    Returns:
        'small' (<1%), 'medium' (1-5%), 'large' (>5%)
    """
    if ratio < 0.01:
        return "small"
    elif ratio < 0.05:
        return "medium"
    else:
        return "large"
# =============================================================================
# 3x3 GRID LOCATION MAPPING
# =============================================================================
LOCATION_GRID = {
    (0, 0): "Top Left",
    (0, 1): "Top",
    (0, 2): "Top Right",
    (1, 0): "Left",
    (1, 1): "Center",
    (1, 2): "Right",
    (2, 0): "Bottom Left",
    (2, 1): "Bottom",
    (2, 2): "Bottom Right"
}
def bbox_to_location(bbox: Tuple[int, int, int, int],
                     image_size: Tuple[int, int],
                     grid_size: int = 3) -> str:
    """
    Map bbox center to location in grid.
    
    Args:
        bbox: (x1, y1, x2, y2) in absolute pixels
        image_size: (width, height) of image
        grid_size: Size of grid (default 3x3)
        
    Returns:
        Location name (e.g., "Top Left", "Center")
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    w, h = image_size
    
    col = int(center_x / w * grid_size)
    row = int(center_y / h * grid_size)
    
    # Clamp to valid range
    col = min(col, grid_size - 1)
    row = min(row, grid_size - 1)
    
    return LOCATION_GRID.get((row, col), "Center")
# =============================================================================
# COMPLETE PROCESSING FUNCTION
# =============================================================================
def process_mask_to_bbox_and_location(mask_path: str,
                                       image_size: Optional[Tuple[int, int]] = None,
                                       normalize: bool = True,
                                       grid_size: int = 3,
                                       min_area_ratio: float = 0.0001) -> Dict:
    """
    Complete processing: bbox + location + size + normalization.
    
    Args:
        mask_path: Path to mask image
        image_size: (width, height), auto-detected if None
        normalize: Whether to compute normalized coordinates
        grid_size: Grid size for location (default 3x3)
        min_area_ratio: Minimum contour area ratio (filters noise)
        
    Returns:
        Dict with all bbox and location information
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        return {
            'bboxes': [],
            'normalized_bboxes': [],
            'combined_bbox': None,
            'normalized_combined_bbox': None,
            'location': None,
            'defect_ratio': 0.0,
            'defect_size': None,
            'image_size': image_size
        }
    
    # Auto-detect size if not provided
    if image_size is None:
        h, w = mask.shape[:2]
        image_size = (w, h)
    
    # Extract bboxes
    bboxes = mask_to_bbox(mask, min_area_ratio)
    combined = get_combined_bbox(bboxes)
    
    # Normalize
    normalized_bboxes = []
    normalized_combined = None
    
    if normalize and bboxes:
        normalized_bboxes = [normalize_bbox(b, image_size) for b in bboxes]
    if normalize and combined:
        normalized_combined = normalize_bbox(combined, image_size)
    
    # Location
    location = bbox_to_location(combined, image_size, grid_size) if combined else None
    
    # Size
    ratio = compute_defect_ratio(mask)
    size = categorize_defect_size(ratio)
    
    return {
        'bboxes': bboxes,
        'normalized_bboxes': normalized_bboxes,
        'combined_bbox': combined,
        'normalized_combined_bbox': normalized_combined,
        'location': location,
        'defect_ratio': round(ratio, 6),
        'defect_size': size,
        'image_size': image_size
    }
def process_dataset_item(item: Dict, 
                         output_dir: str = None,
                         normalize: bool = True) -> Dict:
    """
    Process a single dataset item with full bbox and location info.
    
    Args:
        item: Dataset item dict from load_mvtec_dataset
        output_dir: Directory for annotated images (optional)
        normalize: Whether to compute normalized coordinates
        
    Returns:
        Enhanced item dict with bbox info
    """
    result = item.copy()
    
    if item.get('is_good') or not item.get('mask_path'):
        # Good sample - no bbox info
        result.update({
            'bboxes': [],
            'normalized_bboxes': [],
            'combined_bbox': None,
            'normalized_combined_bbox': None,
            'location': None,
            'defect_ratio': 0.0,
            'defect_size': None,
            'image_size': None
        })
        
        # Get image size
        img = cv2.imread(item['image_path'])
        if img is not None:
            h, w = img.shape[:2]
            result['image_size'] = (w, h)
    else:
        # Defective sample - get bbox info
        img = cv2.imread(item['image_path'])
        image_size = None
        if img is not None:
            h, w = img.shape[:2]
            image_size = (w, h)
        
        bbox_info = process_mask_to_bbox_and_location(
            item['mask_path'],
            image_size=image_size,
            normalize=normalize
        )
        result.update(bbox_info)
    
    return result
if __name__ == "__main__":
    print("Testing Data Processing Module\n" + "="*50)
    
    # Test normalize_bbox
    print("\n1. Normalize Bbox Test:")
    bbox = (100, 200, 300, 400)
    size = (1024, 768)
    norm = normalize_bbox(bbox, size)
    print(f"   Original: {bbox}")
    print(f"   Image size: {size}")
    print(f"   Normalized: {norm}")
    
    # Test defect size
    print("\n2. Defect Size Test:")
    for ratio in [0.005, 0.03, 0.08]:
        size = categorize_defect_size(ratio)
        print(f"   Ratio {ratio:.3f} -> {size}")
    
    # Test location
    print("\n3. Location Test:")
    for bbox in [(50, 50, 150, 150), (400, 300, 600, 500), (800, 600, 950, 750)]:
        loc = bbox_to_location(bbox, (1024, 768))
        print(f"   Bbox {bbox} -> {loc}")
