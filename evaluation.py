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

# =============================================================================
# IMPROVED DEFECT TYPE MAPPING
# =============================================================================
# Map GT defect types to their canonical forms for matching
DEFECT_TYPE_MAPPING = {
    # Exact matches
    'scratch': ['scratch', 'scratched', 'scratches'],
    'crack': ['crack', 'cracked', 'cracks', 'fracture'],
    'cut': ['cut', 'cuts', 'cutting'],
    'poke': ['poke', 'poked', 'puncture', 'punctured'],
    'contamination': ['contamination', 'contaminated', 'debris', 'particles'],
    'broken': ['broken', 'break', 'damage', 'damaged'],
    'bent': ['bent', 'bend', 'deformation', 'deformed'],
    'hole': ['hole', 'holes'],
    'color': ['color', 'discoloration', 'discolored', 'stain', 'stained'],
    'faulty_imprint': ['faulty imprint', 'missing imprint', 'imprint'],
    'squeeze': ['squeeze', 'squeezed', 'compressed'],
    'glue': ['glue', 'adhesive', 'residue'],
    'thread': ['thread', 'fiber', 'loose thread'],
    'gray_stroke': ['gray stroke', 'grey stroke', 'stroke', 'mark'],
    'oil': ['oil', 'grease'],
    'rough': ['rough', 'uneven'],
    
    # Compound defects
    'cut_inner_insulation': ['cut inner insulation', 'inner insulation cut', 'insulation exposed'],
    'cut_outer_insulation': ['cut outer insulation', 'outer insulation cut'],
    'broken_teeth': ['broken teeth', 'damaged teeth'],
    'split_teeth': ['split teeth', 'separated teeth'],
    'squeezed_teeth': ['squeezed teeth', 'compressed teeth'],
}

def normalize_defect_type(defect_type: str) -> str:
    """Normalize defect type for comparison."""
    return defect_type.lower().replace('_', ' ').strip()

def match_defect_types(predicted: str, ground_truth: str) -> bool:
    """
    Strict defect type matching with synonym support.
    
    Args:
        predicted: Extracted defect type from generated text
        ground_truth: Ground truth defect type from annotation
        
    Returns:
        True if match, False otherwise
    """
    pred_norm = normalize_defect_type(predicted)
    gt_norm = normalize_defect_type(ground_truth)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # Check if GT has a mapping
    if gt_norm in DEFECT_TYPE_MAPPING:
        return pred_norm in DEFECT_TYPE_MAPPING[gt_norm]
    
    # Fuzzy match: check if pred is a key word in gt
    gt_words = set(gt_norm.split())
    pred_words = set(pred_norm.split())
    
    # Require at least 2 words to match for compound defects
    if len(gt_words) > 1:
        return len(gt_words & pred_words) >= 2
    else:
        return pred_norm in gt_words

# =============================================================================
# LOCATION KEYWORDS WITH SYNONYMS
# =============================================================================
LOCATION_SYNONYMS = {
    'top left': {'top left', 'upper left', 'top-left', 'upper-left'},
    'top': {'top', 'upper', 'above'},
    'top right': {'top right', 'upper right', 'top-right', 'upper-right'},
    'left': {'left', 'left side', 'left edge'},
    'center': {'center', 'middle', 'central', 'centre'},
    'right': {'right', 'right side', 'right edge'},
    'bottom left': {'bottom left', 'lower left', 'bottom-left', 'lower-left'},
    'bottom': {'bottom', 'lower', 'below'},
    'bottom right': {'bottom right', 'lower right', 'bottom-right', 'lower-right'}
}

def normalize_location(location: str) -> Optional[str]:
    """Normalize location to canonical form."""
    if not location:
        return None
    
    loc_lower = location.lower().strip()
    
    # Find canonical form
    for canonical, synonyms in LOCATION_SYNONYMS.items():
        if loc_lower in synonyms:
            return canonical
    
    return loc_lower

def match_locations(predicted: str, ground_truth: str) -> bool:
    """Match locations with synonym support."""
    pred_norm = normalize_location(predicted)
    gt_norm = normalize_location(ground_truth)
    
    if not pred_norm or not gt_norm:
        return False
    
    return pred_norm == gt_norm

# =============================================================================
# ATTRIBUTE EXTRACTION
# =============================================================================
def extract_defect_type(text: str) -> Optional[str]:
    """Extract defect type from generated text with improved pattern matching."""
    text_lower = text.lower()
    
    # Priority 1: Check for compound defects first (longer patterns)
    compound_defects = [
        'cut inner insulation', 'cut outer insulation',
        'broken teeth', 'split teeth', 'squeezed teeth',
        'faulty imprint', 'gray stroke', 'grey stroke'
    ]
    
    for defect in compound_defects:
        if defect in text_lower:
            return defect
    
    # Priority 2: Check for single-word defects
    single_defects = [
        'scratch', 'crack', 'poke', 'contamination', 'cut',
        'broken', 'bent', 'hole', 'squeeze', 'glue', 'thread',
        'oil', 'rough', 'color', 'imprint'
    ]
    
    for defect in single_defects:
        # Use word boundary matching to avoid partial matches
        if re.search(r'\b' + defect + r'\b', text_lower):
            return defect
    
    return None

def extract_location(text: str) -> Optional[str]:
    """Extract location from generated text."""
    text_lower = text.lower()
    
    # Check each location synonym set
    for canonical, synonyms in LOCATION_SYNONYMS.items():
        for syn in synonyms:
            if syn in text_lower:
                return canonical
    
    return None

def extract_attributes(text: str) -> Dict[str, Optional[str]]:
    """Extract all attributes from generated text."""
    return {
        'defect_type': extract_defect_type(text),
        'location': extract_location(text)
    }

# =============================================================================
# TIER 1: ATTRIBUTE-LEVEL EVALUATION (IMPROVED)
# =============================================================================
def evaluate_attributes(results: List[Dict], 
                        use_ground_truth_location: bool = True,
                        verbose: bool = False) -> Dict:
    """
    Evaluate attribute-level accuracy with strict matching.
    
    Args:
        results: List of result dicts with generated_text, category, defect_type, location
        use_ground_truth_location: Use location from bbox analysis as ground truth
        verbose: Print detailed matching results
        
    Returns:
        Dict with accuracy metrics for each attribute
    """
    defect_correct = 0
    location_correct = 0
    total = 0
    
    mismatches = []
    
    for item in results:
        if item.get('defect_type') == 'good':
            continue
            
        total += 1
        generated_text = item.get('generated_text', '')
        extracted = extract_attributes(generated_text)
        
        gt_defect = item.get('defect_type', '')
        gt_location = item.get('location', '')
        
        # Defect type accuracy (STRICT matching)
        defect_match = False
        if extracted['defect_type'] and gt_defect:
            defect_match = match_defect_types(extracted['defect_type'], gt_defect)
            if defect_match:
                defect_correct += 1
        
        # Location accuracy
        location_match = False
        if use_ground_truth_location and extracted['location'] and gt_location:
            location_match = match_locations(extracted['location'], gt_location)
            if location_match:
                location_correct += 1
        
        # Track mismatches
        if not defect_match or not location_match:
            mismatches.append({
                'category': item.get('category'),
                'gt_defect': gt_defect,
                'pred_defect': extracted['defect_type'],
                'defect_match': defect_match,
                'gt_location': gt_location,
                'pred_location': extracted['location'],
                'location_match': location_match,
                'text': generated_text[:100]
            })
    
    if verbose and mismatches:
        print("\n=== ATTRIBUTE MISMATCHES (first 10) ===")
        for i, m in enumerate(mismatches[:10]):
            print(f"\n{i+1}. Category: {m['category']}")
            print(f"   Defect - GT: '{m['gt_defect']}' | Pred: '{m['pred_defect']}' | Match: {m['defect_match']}")
            print(f"   Location - GT: '{m['gt_location']}' | Pred: '{m['pred_location']}' | Match: {m['location_match']}")
            print(f"   Text: {m['text']}...")
    
    return {
        'defect_type_accuracy': defect_correct / total if total > 0 else 0.0,
        'location_accuracy': location_correct / total if total > 0 else 0.0,
        'total_samples': total,
        'defect_type_correct': defect_correct,
        'location_correct': location_correct,
        'mismatches': mismatches if verbose else []
    }

# =============================================================================
# TIER 2: TEXT QUALITY METRICS (UNCHANGED)
# =============================================================================
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
            "method": "simplified (NOT COMPARABLE WITH PAPERS)",
            "metrics": results,
            "comparable_with_papers": False,
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
        "comparable_with_papers": True
    }

# ... (Keep all simplified metric functions unchanged)

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

# =============================================================================
# GROUND TRUTH GENERATION (UNCHANGED)
# =============================================================================
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
            None  # No size
        )
    
    # Tier 1: Attribute evaluation
    attribute_results = evaluate_attributes(results, use_location, verbose)
    
    # Tier 2: Text quality evaluation
    quality_results = evaluate_text_quality(
        generated_texts, reference_texts, use_standard_metrics
    )
    
    return {
        "attribute_accuracy": attribute_results,
        "text_quality": quality_results,
        "summary": {
            "defect_type_accuracy": attribute_results['defect_type_accuracy'],
            "location_accuracy": attribute_results['location_accuracy'],
            "comparable_with_papers": quality_results.get('comparable_with_papers', False),
            "num_samples": attribute_results['total_samples']
        }
    }
if __name__ == "__main__":
    print("Testing Evaluation Module\n" + "="*50)
    
    # Test attribute extraction
    print("\n1. Attribute Extraction Test:")
    test_text = "A large scratch defect on the top left area of the wood surface."
    attrs = extract_attributes(test_text)
    print(f"   Text: {test_text}")
    print(f"   Extracted: {attrs}")
    
    # Test metrics
    print("\n2. Metrics Test:")
    gen = {"0": "A scratch on the wood at the center"}
    ref = {"0": "A wood with scratch defect at the Center region"}
    
    result = evaluate_text_quality(gen, ref, use_standard=False)
    print(f"   Method: {result['method']}")
    print(f"   Metrics: {result['metrics']}")
    print(f"   Comparable: {result['comparable_with_papers']}")


