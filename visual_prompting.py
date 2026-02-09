"""
Visual Prompting Module for Defect Detection
Draws visual annotations (red contours, overlays) on defect regions.
"""
import os
import cv2
import numpy as np
from typing import Tuple, Optional
def draw_defect_contour(image: np.ndarray,
                         mask: np.ndarray,
                         color: Tuple[int, int, int] = (0, 0, 255),
                         thickness: int = 2,
                         fill_alpha: float = 0.0) -> np.ndarray:
    """
    Draw red contour around defect region.
    
    Args:
        image: BGR image (H, W, 3)
        mask: Grayscale mask (H, W), white = defect
        color: Contour color in BGR (default: red)
        thickness: Contour line thickness
        fill_alpha: If > 0, fill contour with semi-transparent color
        
    Returns:
        Annotated image with red contour
    """
    annotated = image.copy()
    
    # Threshold mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if fill_alpha > 0:
        # Create overlay with filled contours
        overlay = annotated.copy()
        cv2.drawContours(overlay, contours, -1, color, -1)
        annotated = cv2.addWeighted(overlay, fill_alpha, annotated, 1 - fill_alpha, 0)
    
    # Draw contour outline
    cv2.drawContours(annotated, contours, -1, color, thickness)
    
    return annotated
def draw_bbox(image: np.ndarray,
              bbox: Tuple[int, int, int, int],
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2,
              label: str = None) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: BGR image
        bbox: (x1, y1, x2, y2) in absolute pixels
        color: Box color in BGR
        thickness: Line thickness
        label: Optional text label
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    x1, y1, x2, y2 = [int(c) for c in bbox]
    
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return annotated
def create_annotated_image(image_path: str,
                            mask_path: str,
                            output_path: str = None,
                            contour_color: Tuple[int, int, int] = (0, 0, 255),
                            contour_thickness: int = 2,
                            fill_alpha: float = 0.0) -> Optional[np.ndarray]:
    """
    Create annotated image from image and mask paths.
    
    Args:
        image_path: Path to original image
        mask_path: Path to defect mask
        output_path: Path to save annotated image (optional)
        contour_color: Contour color in BGR
        contour_thickness: Line thickness
        fill_alpha: Fill transparency (0 = no fill)
        
    Returns:
        Annotated image array
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        return None
    
    # Resize mask if sizes don't match
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    annotated = draw_defect_contour(
        image, mask,
        color=contour_color,
        thickness=contour_thickness,
        fill_alpha=fill_alpha
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, annotated)
    
    return annotated
def process_visual_prompts(dataset: list,
                            output_dir: str,
                            contour_color: Tuple[int, int, int] = (0, 0, 255),
                            contour_thickness: int = 2,
                            fill_alpha: float = 0.0,
                            show_progress: bool = True) -> list:
    """
    Create visual prompts for entire dataset.
    
    Args:
        dataset: List of dataset items from load_mvtec_dataset
        output_dir: Directory to save annotated images
        contour_color: Contour color in BGR
        contour_thickness: Line thickness
        fill_alpha: Fill transparency
        show_progress: Show progress bar
        
    Returns:
        List of items with annotated_path added
    """
    results = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(dataset, desc="Creating visual prompts")
        except ImportError:
            iterator = dataset
    else:
        iterator = dataset
    
    for item in iterator:
        result = item.copy()
        
        if item.get('is_good') or not item.get('mask_path'):
            # Good samples - no annotation needed
            result['annotated_path'] = item['image_path']
        else:
            # Defective samples - create annotated version
            category = item.get('category', 'unknown')
            defect_type = item.get('defect_type', 'unknown')
            image_name = item.get('image_name', 'image')
            
            output_subdir = os.path.join(output_dir, category, defect_type)
            output_path = os.path.join(output_subdir, f"{image_name}_annotated.png")
            
            annotated = create_annotated_image(
                item['image_path'],
                item['mask_path'],
                output_path,
                contour_color=contour_color,
                contour_thickness=contour_thickness,
                fill_alpha=fill_alpha
            )
            
            if annotated is not None:
                result['annotated_path'] = output_path
            else:
                result['annotated_path'] = item['image_path']
        
        results.append(result)
    
    return results
if __name__ == "__main__":
    print("Testing Visual Prompting Module\n" + "="*50)
    
    # Create test image and mask
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    test_image[:] = (200, 200, 200)  # Gray background
    
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_mask, (128, 128), 50, 255, -1)  # White circle as defect
    
    # Draw contour
    annotated = draw_defect_contour(test_image, test_mask, fill_alpha=0.2)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Mask shape: {test_mask.shape}")
    print(f"Output shape: {annotated.shape}")
    print("Visual prompting test complete!")