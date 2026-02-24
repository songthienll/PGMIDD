"""
Evaluation Module for Defect Detection
Two-tier evaluation system:
- Tier 1: Attribute-level factual accuracy (automatic)
- Tier 2: Text quality metrics (official pycocoevalcap)
"""
import re
import math
import warnings
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter

# =============================================================================
# PRE-DEFINED GROUND TRUTH DESCRIPTIONS
# =============================================================================
MVTEC_GROUND_TRUTH = {
    "bottle": {
        "good": "A clean glass bottle with no visible defects.",
        "broken_large": "A glass bottle with a large fracture or significant structural damage visible on the body.",
        "broken_small": "A glass bottle with a small crack or chip visible on the surface.",
        "contamination": "A glass bottle with contamination defect showing foreign particles or debris on the surface."
    },
    "cable": {
        "good": "A normal industrial cable with no visible defects.",
        "bent_wire": "An industrial cable with bent wire showing deformation in the conductor.",
        "cable_swap": "An industrial cable with incorrect wire positioning or swapped conductors.",
        "combined": "An industrial cable with multiple combined defects visible.",
        "cut_inner_insulation": "An industrial cable with cut inner insulation exposing the wire.",
        "cut_outer_insulation": "An industrial cable with cut outer insulation showing damage.",
        "missing_cable": "An industrial cable with a missing conductor or component.",
        "missing_wire": "An industrial cable with missing wire in the assembly.",
        "poke_insulation": "An industrial cable with poked insulation showing puncture damage."
    },
    "capsule": {
        "good": "A normal pharmaceutical capsule with no visible defects.",
        "crack": "A pharmaceutical capsule with crack defect showing structural damage.",
        "faulty_imprint": "A pharmaceutical capsule with faulty or missing imprint.",
        "poke": "A pharmaceutical capsule with poke damage or puncture.",
        "scratch": "A pharmaceutical capsule with scratch marks on the surface.",
        "squeeze": "A pharmaceutical capsule with squeeze deformation."
    },
    "carpet": {
        "good": "A normal carpet with no visible defects.",
        "color": "A carpet with color defect showing incorrect or uneven coloring.",
        "cut": "A carpet with cut damage showing torn fibers.",
        "hole": "A carpet with hole defect showing missing material.",
        "metal_contamination": "A carpet with metal contamination embedded in fibers.",
        "thread": "A carpet with thread defect showing loose or pulled threads."
    },
    "grid": {
        "good": "A normal metal grid with no visible defects.",
        "bent": "A metal grid with bent structure showing deformation.",
        "broken": "A metal grid with broken wires or damaged structure.",
        "glue": "A metal grid with glue residue or contamination.",
        "metal_contamination": "A metal grid with metal contamination or foreign particles.",
        "thread": "A metal grid with thread or fiber contamination."
    },
    "hazelnut": {
        "good": "A normal hazelnut with no visible defects.",
        "crack": "A hazelnut with crack defect showing shell damage.",
        "cut": "A hazelnut with cut marks on the shell.",
        "hole": "A hazelnut with hole defect indicating pest damage.",
        "print": "A hazelnut with print or marking defect."
    },
    "leather": {
        "good": "A normal leather surface with no visible defects.",
        "color": "A leather surface with color defect showing discoloration.",
        "cut": "A leather surface with cut damage showing torn material.",
        "fold": "A leather surface with fold marks or creases.",
        "glue": "A leather surface with visible glue residue.",
        "poke": "A leather surface with poke marks or punctures."
    },
    "metal_nut": {
        "good": "A normal metal nut with no visible defects.",
        "bent": "A metal nut with bent structure showing deformation.",
        "color": "A metal nut with color defect or discoloration.",
        "flip": "A metal nut incorrectly flipped or oriented.",
        "scratch": "A metal nut with scratch marks on the surface."
    },
    "pill": {
        "good": "A normal pharmaceutical pill with no visible defects.",
        "color": "A pharmaceutical pill with color defect or discoloration.",
        "combined": "A pharmaceutical pill with multiple combined defects.",
        "contamination": "A pharmaceutical pill with contamination on surface.",
        "crack": "A pharmaceutical pill with crack in the coating.",
        "faulty_imprint": "A pharmaceutical pill with faulty or missing imprint.",
        "pill_type": "A pharmaceutical pill with incorrect pill type.",
        "scratch": "A pharmaceutical pill with scratch marks."
    },
    "screw": {
        "good": "A normal screw with no visible defects.",
        "manipulated_front": "A screw with manipulated front showing damage to head.",
        "scratch_head": "A screw with scratched head surface.",
        "scratch_neck": "A screw with scratched neck area.",
        "thread_side": "A screw with damaged thread on side.",
        "thread_top": "A screw with damaged thread at top."
    },
    "tile": {
        "good": "A normal tile with no visible defects.",
        "crack": "A tile with crack defect showing surface damage.",
        "glue_strip": "A tile with visible glue strip residue.",
        "gray_stroke": "A tile with gray stroke mark or discoloration.",
        "oil": "A tile with oil stain or contamination.",
        "rough": "A tile with rough surface texture defect."
    },
    "toothbrush": {
        "good": "A normal toothbrush with no visible defects.",
        "defective": "A toothbrush with manufacturing defect in bristles or handle."
    },
    "transistor": {
        "good": "A normal transistor with no visible defects.",
        "bent_lead": "A transistor with bent lead pins.",
        "cut_lead": "A transistor with cut or missing lead pins.",
        "damaged_case": "A transistor with damaged plastic case.",
        "misplaced": "A transistor with misplaced or incorrectly positioned component."
    },
    "wood": {
        "good": "A normal wood surface with no visible defects.",
        "color": "A wood surface with color defect or staining.",
        "combined": "A wood surface with multiple combined defects.",
        "hole": "A wood surface with hole or knot defect.",
        "liquid": "A wood surface with liquid stain or water damage.",
        "scratch": "A wood surface with scratch marks."
    },
    "zipper": {
        "good": "A normal zipper with no visible defects.",
        "broken_teeth": "A zipper with broken teeth showing damage.",
        "combined": "A zipper with multiple combined defects.",
        "fabric_border": "A zipper with fabric border defect.",
        "fabric_interior": "A zipper with fabric interior damage.",
        "rough": "A zipper with rough texture or irregular teeth.",
        "split_teeth": "A zipper with split or separated teeth.",
        "squeezed_teeth": "A zipper with squeezed or compressed teeth."
    }
}

VISA_GROUND_TRUTH = {
    "candle": {
        "good": "A normal decorative candle with no visible defects.",
        "anomaly": "A decorative candle with manufacturing defect visible on surface."
    },
    "capsules": {
        "good": "A normal pharmaceutical capsule with no visible defects.",
        "anomaly": "A pharmaceutical capsule with visible defect or damage."
    },
    "cashew": {
        "good": "A normal cashew nut with no visible defects.",
        "anomaly": "A cashew nut with visible defect such as burn or damage."
    },
    "chewinggum": {
        "good": "A normal piece of chewing gum with no visible defects.",
        "anomaly": "A piece of chewing gum with manufacturing defect."
    },
    "fryum": {
        "good": "A normal fryum snack with no visible defects.",
        "anomaly": "A fryum snack with visible defect or damage."
    },
    "macaroni1": {
        "good": "A normal macaroni pasta piece with no visible defects.",
        "anomaly": "A macaroni pasta piece with visible defect."
    },
    "macaroni2": {
        "good": "A normal macaroni pasta piece with no visible defects.",
        "anomaly": "A macaroni pasta piece with visible defect."
    },
    "pcb1": {
        "good": "A normal PCB board with no visible defects.",
        "anomaly": "A PCB board with visible defect such as missing component or solder issue."
    },
    "pcb2": {
        "good": "A normal PCB board with no visible defects.",
        "anomaly": "A PCB board with visible defect such as missing component or solder issue."
    },
    "pcb3": {
        "good": "A normal PCB board with no visible defects.",
        "anomaly": "A PCB board with visible defect such as missing component or solder issue."
    },
    "pcb4": {
        "good": "A normal PCB board with no visible defects.",
        "anomaly": "A PCB board with visible defect such as missing component or solder issue."
    },
    "pipe_fryum": {
        "good": "A normal pipe fryum snack with no visible defects.",
        "anomaly": "A pipe fryum snack with visible defect or damage."
    }
}

def evaluate_text_quality(generated_texts: Dict[str, str],
                          reference_texts: Dict[str, str],
                          use_standard: bool = True) -> Dict:
    """
    Evaluate generated text quality using official metrics.
    
    Args:
        generated_texts: Dict[image_id, generated_text]
        reference_texts: Dict[image_id, reference_text]
        use_standard: Use pycocoevalcap (True) or simplified (False)
    
    Returns:
        Dict with metrics and comparability flag
    """
    if use_standard:
        try:
            return _compute_standard_metrics(generated_texts, reference_texts)
        except ImportError as e:
            warnings.warn(f"pycocoevalcap not available: {e}. Using simplified metrics.")
            use_standard = False
    
    if not use_standard:
        results = _compute_simplified_metrics(generated_texts, reference_texts)
        return {
            "metrics": results,
            "warning": "Install pycocoevalcap for accurate comparison with published results: pip install pycocoevalcap"
        }

def _compute_standard_metrics(generated_texts: Dict[str, str],
                               reference_texts: Dict[str, str]) -> Dict:
    """Compute metrics using official pycocoevalcap."""
    from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.meteor.meteor import Meteor
    
    # Format data for pycocoevalcap
    gts = {id: [text] for id, text in reference_texts.items()}
    res = {id: [text] for id, text in generated_texts.items()}
    
    results = {}
    
    try:
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
        results["METEOR"] = meteor_score
    except Exception as e:
        warnings.warn(f"Error computing METEOR: {e}")
        results["METEOR"] = None
    
    try:
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(gts, res)
        results["SPICE"] = spice_score
    except Exception as e:
        warnings.warn(f"Error computing SPICE: {e}")
        results["SPICE"] = None
    
    return {
        "method": "pycocoevalcap (official)",
        "metrics": results,
    }

def _compute_simplified_metrics(generated_texts: Dict[str, str],
                                 reference_texts: Dict[str, str]) -> Dict:
    """Compute all simplified metrics."""
    from evaluation import compute_meteor_simplified, compute_spice_simplified
    
    meteor_scores = []
    spice_scores = []
    
    for id, gen_text in generated_texts.items():
        ref_text = reference_texts.get(id, '')
        if ref_text:
            meteor_scores.append(compute_meteor_simplified(gen_text, ref_text))
            spice_scores.append(compute_spice_simplified(gen_text, ref_text))
    
    return {
        "METEOR": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        "SPICE": sum(spice_scores) / len(spice_scores) if spice_scores else 0.0
    }

def create_ground_truth(category: str,
                        defect_type: str,
                        location: str = None,
                        defect_size: str = None,
                        dataset_type: str = "mvtec") -> str:
    """Create ground truth text using hybrid approach."""
    ground_truth_dict = MVTEC_GROUND_TRUTH if dataset_type == "mvtec" else VISA_GROUND_TRUTH
    
    if category in ground_truth_dict and defect_type in ground_truth_dict[category]:
        base_text = ground_truth_dict[category][defect_type]
        
        if location and defect_type != 'good':
            location_clean = location.lower().replace(' region', '').strip()
            base_text += f" The defect is located at the {location_clean} region."
        
        return base_text
    
    defect_clean = defect_type.replace('_', ' ')
    parts = [f"A {category} with {defect_clean} defect"]
    
    if location:
        parts.append(f"at the {location} region")
    
    return ' '.join(parts)

# =============================================================================
# COMBINED EVALUATION
# =============================================================================
def evaluate_all(results: List[Dict],
                 use_location: bool = True,
                 use_standard_metrics: bool = True,
                 verbose: bool = False) -> Dict:
    """
    Complete evaluation with both attribute and text quality metrics.
    
    Args:
        results: List of dicts with generated_text, category, defect_type, location
        use_location: Include location in ground truth
        use_standard_metrics: Use pycocoevalcap (True) or simplified (False)
        verbose: Print detailed mismatch information
        
    Returns:
        Dict with all evaluation results
    """
    # Prepare texts for quality evaluation
    generated_texts = {}
    reference_texts = {}
    
    for idx, item in enumerate(results):
        if item.get('defect_type') == 'good':
            continue
            
        image_id = str(idx)
        generated_texts[image_id] = item.get('generated_text', '')
        
        location = item.get('location') if use_location else None
        reference_texts[image_id] = create_ground_truth(
            item.get('category', 'product'),
            item.get('defect_type', 'defect'),
            location,
        )
        
    # Text quality evaluation
    quality_results = evaluate_text_quality(generated_texts, reference_texts, use_standard_metrics)
    
    return {
        "attribute_accuracy": attribute_results,
        "text_quality": quality_results,
        "summary": {"num_samples": attribute_results['total_samples']}
    }
def compute_meteor_simplified(gen_text: str, ref_text: str) -> float:
    return 0.0 

def compute_spice_simplified(gen_text: str, ref_text: str) -> float:
    return 0.0
if __name__ == "__main__":
    print("Testing Evaluation Module\n" + "="*50)
    
    # Test metrics
    print("\n Metrics Test:")
    gen = {"0": "A scratch on the wood at the center"}
    ref = {"0": "A wood with scratch defect at the Center region"}
    
    result = evaluate_text_quality(gen, ref, use_standard=False)
    print(f"   Method: {result['method']}")
    print(f"   Metrics: {result['metrics']}")




