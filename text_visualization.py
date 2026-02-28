"""
Text Visualization and Saving Module
Provides functions to save and visualize generated text descriptions with images.
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import textwrap

def save_generated_text(results: List[Dict], 
                        output_dir: str,
                        text_folder: str = "generated_text"):
    """
    Save generated text descriptions to files organized by category and defect type.
    
    Args:
        results: List of result dictionaries with 'generated_text', 'category', 
                'defect_type', 'image_name'
        output_dir: Base output directory
        text_folder: Subfolder name for text files (default: 'generated_text')
        
    Returns:
        Dictionary with statistics about saved files
    """
    text_dir = os.path.join(output_dir, text_folder)
    os.makedirs(text_dir, exist_ok=True)
    
    saved_count = 0
    saved_files = []
    
    for item in results:
        if 'generated_text' not in item:
            continue
            
        category = item.get('category', 'unknown')
        defect_type = item.get('defect_type', 'unknown')
        image_name = item.get('image_name', 'unknown')
        
        # Create category/defect_type directory
        cat_dir = os.path.join(text_dir, category, defect_type)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Save text file
        text_file = os.path.join(cat_dir, f"{image_name}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(item['generated_text'])
        
        saved_count += 1
        saved_files.append(text_file)
    
    # Save metadata
    metadata = {
        'total_saved': saved_count,
        'output_dir': text_dir,
        'files': saved_files
    }
    
    metadata_file = os.path.join(text_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n Saved {saved_count} text files to {text_dir}")
    print(f"  Metadata saved to: {metadata_file}")
    
    return metadata


def load_generated_text(output_dir: str,
                        text_folder: str = "generated_text") -> Dict:
    """
    Load previously saved text descriptions.
    
    Args:
        output_dir: Base output directory
        text_folder: Subfolder name for text files
        
    Returns:
        Dictionary mapping image paths to their text descriptions
    """
    text_dir = os.path.join(output_dir, text_folder)
    metadata_file = os.path.join(text_dir, 'metadata.json')
    
    if not os.path.exists(metadata_file):
        print(f"No metadata found at {metadata_file}")
        return {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    text_dict = {}
    for text_file in metadata.get('files', []):
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                text_dict[text_file] = f.read()
    
    print(f"\n Loaded {len(text_dict)} text descriptions from {text_dir}")
    return text_dict


def visualize_results(results: List[Dict],
                     num_samples: int = 6,
                     cols: int = 2,
                     figsize: tuple = (15, 12),
                     show_metadata: bool = True,
                     save_path: str = None):
    """
    Visualize images with their generated text descriptions.
    
    Args:
        results: List of result dictionaries
        num_samples: Number of samples to display
        cols: Number of columns in grid
        figsize: Figure size (width, height)
        show_metadata: Whether to show category/defect_type/location
        save_path: Optional path to save the visualization
    """
    # Filter results with generated text
    valid_results = [r for r in results if 'generated_text' in r]
    
    if len(valid_results) == 0:
        print("No results with generated text found!")
        return
    
    # Sample results
    import random
    if len(valid_results) > num_samples:
        samples = random.sample(valid_results, num_samples)
    else:
        samples = valid_results[:num_samples]
    
    # Calculate rows
    rows = (len(samples) + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load image
        image_path = result.get('annotated_path') or result.get('image_path')
        if not os.path.exists(image_path):
            ax.axis('off')
            continue
            
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Display image (use gray colormap for grayscale images)
        cmap = 'gray' if img_array.ndim == 2 else None
        ax.imshow(img_array, cmap=cmap)
        ax.axis('off')
        
        # Color border based on defect type
        is_defect = result.get('defect_type', 'unknown') != 'good'
        border_color = 'red' if is_defect else 'green'
        border_width = 3
        
        # Add colored border
        rect = Rectangle((0, 0), img_array.shape[1], img_array.shape[0],
                        linewidth=border_width, edgecolor=border_color,
                        facecolor='none')
        ax.add_patch(rect)
        
        # Prepare title text
        title_parts = []
        if show_metadata:
            category = result.get('category', 'Unknown')
            defect_type = result.get('defect_type', 'Unknown')
            location = result.get('location', '')
            
            if location:
                title_parts.append(f"{category} | {defect_type} | {location}")
            else:
                title_parts.append(f"{category} | {defect_type}")
        
        # Add generated text
        generated_text = result.get('generated_text', 'No text generated')
        
        title_parts.append(f"\n\"{generated_text}\"")
        
        title = '\n'.join(title_parts)
        ax.set_title(title, fontsize=9, pad=10, color=border_color, weight='bold')
    
    # Hide empty subplots
    for idx in range(len(samples), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Visualization saved to {save_path}")
    
    plt.show()


def visualize_by_category(results: List[Dict],
                         category: str,
                         num_samples: int = 4,
                         cols: int = 2,
                         figsize: tuple = (12, 10)):
    """
    Visualize results for a specific category.
    
    Args:
        results: List of result dictionaries
        category: Category to filter by
        num_samples: Number of samples to display
        cols: Number of columns in grid
        figsize: Figure size
    """
    # Filter by category
    category_results = [r for r in results if r.get('category') == category and 'generated_text' in r]
    
    if len(category_results) == 0:
        print(f"No results found for category: {category}")
        return
    
    print(f"\nVisualizing {category} ({len(category_results)} samples available)")
    visualize_results(category_results, num_samples, cols, figsize)


def compare_defect_vs_normal(results: List[Dict],
                             category: str = None,
                             defect_samples: int = 3,
                             normal_samples: int = 3,
                             figsize: tuple = (15, 10)):
    """
    Compare defect and normal images side by side.
    
    Args:
        results: List of result dictionaries
        category: Optional category filter
        defect_samples: Number of defect samples
        normal_samples: Number of normal samples
        figsize: Figure size
    """
    # Filter results
    if category:
        results = [r for r in results if r.get('category') == category]
    
    defect_results = [r for r in results if r.get('defect_type') != 'good' and 'generated_text' in r]
    normal_results = [r for r in results if r.get('defect_type') == 'good' and 'generated_text' in r]
    
    # Sample
    import random
    defect_samples = random.sample(defect_results, min(defect_samples, len(defect_results)))
    normal_samples = random.sample(normal_results, min(normal_samples, len(normal_results)))
    
    # Create figure
    fig, axes = plt.subplots(2, max(defect_samples.__len__(), normal_samples.__len__()), 
                            figsize=figsize)
    
    # Top row: Defects
    for idx, result in enumerate(defect_samples):
        ax = axes[0, idx] if len(defect_samples) > 1 else axes[0]
        _display_single_result(ax, result, 'red')
    
    # Bottom row: Normal
    for idx, result in enumerate(normal_samples):
        ax = axes[1, idx] if len(normal_samples) > 1 else axes[1]
        _display_single_result(ax, result, 'green')
    
    # Labels
    axes[0, 0].set_ylabel('DEFECTIVE', fontsize=14, weight='bold', rotation=0, 
                         labelpad=50, va='center')
    axes[1, 0].set_ylabel('NORMAL', fontsize=14, weight='bold', rotation=0,
                         labelpad=50, va='center')
    
    category_str = f" - {category}" if category else ""
    plt.suptitle(f'Defect vs Normal Comparison{category_str}', 
                fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.show()


def _display_single_result(ax, result, border_color):
    """Helper function to display a single result."""
    image_path = result.get('annotated_path') or result.get('image_path')
    
    if not os.path.exists(image_path):
        ax.axis('off')
        return
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    ax.imshow(img_array, cmap='gray' if img_array.ndim == 2 else None)
    ax.axis('off')
    
    # Border
    rect = Rectangle((0, 0), img_array.shape[1], img_array.shape[0],
                    linewidth=3, edgecolor=border_color, facecolor='none')
    ax.add_patch(rect)
    
    # Text
    generated_text = result.get('generated_text', 'No text')
    wrapped_text = textwrap.fill(generated_text, width=40)
    
    ax.set_title(f'"{wrapped_text}"', fontsize=9, color=border_color, weight='bold')


def visualize_results_with_metrics(results,
                                    num_samples: int = 6,
                                    cols: int = 2,
                                    figsize: tuple = (16, 20),
                                    save_path: str = None,
                                    dataset_type: str = "mvtec"):
    """
    Visualize images with generated text, ground truth, and per-sample metrics.
    
    Args:
        results: List of result dictionaries with 'generated_text', 'category', etc.
        num_samples: Number of samples to display
        cols: Number of columns in grid
        figsize: Figure size (width, height)
        save_path: Optional path to save the visualization
        dataset_type: 'mvtec' or 'visa'
    """
    from evaluation import (create_ground_truth, compute_bleu_simplified,
                            compute_rouge_1_simplified, compute_rouge_l_simplified,
                            compute_meteor_simplified, compute_spice_simplified)
    import random

    valid = [r for r in results if 'generated_text' in r]
    if not valid:
        print("No results with generated text found!")
        return

    samples = random.sample(valid, min(num_samples, len(valid)))
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(samples):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        # Load & display image
        image_path = result.get('annotated_path') or result.get('image_path')
        if not image_path or not os.path.exists(image_path):
            ax.axis('off')
            continue

        img = Image.open(image_path)
        img_array = np.array(img)
        ax.imshow(img_array, cmap='gray' if img_array.ndim == 2 else None)
        ax.axis('off')

        # Border color
        is_defect = result.get('defect_type', 'good') != 'good'
        border_color = 'red' if is_defect else 'green'
        rect = Rectangle((0, 0), img_array.shape[1], img_array.shape[0],
                         linewidth=3, edgecolor=border_color, facecolor='none')
        ax.add_patch(rect)

        # Compute per-sample metrics
        category = result.get('category', 'product')
        defect_type = result.get('defect_type', 'good')
        location = result.get('location')
        gen_text = result.get('generated_text', '')
        ref_text = create_ground_truth(category, defect_type, location,
                                       dataset_type=dataset_type)

        bleu = compute_bleu_simplified(gen_text, ref_text)
        rouge1 = compute_rouge_1_simplified(gen_text, ref_text)
        rougel = compute_rouge_l_simplified(gen_text, ref_text)
        meteor = compute_meteor_simplified(gen_text, ref_text)
        spice = compute_spice_simplified(gen_text, ref_text)

        # Build title
        meta_line = f"{category} | {defect_type}"
        if location and is_defect:
            meta_line += f" | {location}"

        gen_wrapped = textwrap.fill(gen_text[:120], width=50)
        ref_wrapped = textwrap.fill(ref_text[:120], width=50)

        metrics_line = (f"BLEU-4={bleu:.3f}  ROUGE-1={rouge1:.3f}  "
                       f"ROUGE-L={rougel:.3f}  METEOR={meteor:.3f}  SPICE={spice:.3f}")

        title = (f"{meta_line}\n"
                f"Gen: \"{gen_wrapped}\"\n"
                f"Ref: \"{ref_wrapped}\"\n"
                f"{metrics_line}")

        ax.set_title(title, fontsize=8, pad=10, color=border_color,
                    weight='bold', ha='center')

    # Hide empty subplots
    for idx in range(len(samples), rows * cols):
        axes[idx // cols, idx % cols].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    print("Text Visualization Module")
    print("="*50)
    print("\nAvailable functions:")
    print("  - save_generated_text()")
    print("  - load_generated_text()")
    print("  - visualize_results()")
    print("  - visualize_by_category()")
    print("  - compare_defect_vs_normal()")
    print("  - visualize_results_with_metrics()")
