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
# DEFECT KEYWORDS AND ATTRIBUTE EXTRACTION
# =============================================================================
DEFECT_TYPES = {
    'scratch', 'crack', 'dent', 'hole', 'stain', 'discoloration',
    'contamination', 'broken', 'missing', 'bent', 'cut', 'tear',
    'defect', 'damage', 'anomaly', 'flaw', 'imperfection', 'fault',
    'breakage', 'chip', 'peel', 'leak', 'spot', 'mark', 'dirt',
    'void', 'bubble', 'scratch', 'pitted', 'corroded', 'worn'
}
SIZE_KEYWORDS = {
    'small': ['small', 'tiny', 'minor', 'slight', 'subtle', 'faint'],
    'medium': ['medium', 'moderate', 'visible', 'noticeable', 'clear'],
    'large': ['large', 'big', 'major', 'significant', 'obvious', 'prominent', 'severe']
}
LOCATION_KEYWORDS = {
    'top left': ['top left', 'upper left', 'top-left', 'upper-left'],
    'top': ['top', 'upper', 'above'],
    'top right': ['top right', 'upper right', 'top-right', 'upper-right'],
    'left': ['left', 'left side', 'left edge'],
    'center': ['center', 'middle', 'central', 'centre'],
    'right': ['right', 'right side', 'right edge'],
    'bottom left': ['bottom left', 'lower left', 'bottom-left', 'lower-left'],
    'bottom': ['bottom', 'lower', 'below'],
    'bottom right': ['bottom right', 'lower right', 'bottom-right', 'lower-right']
}
def extract_defect_type(text: str) -> Optional[str]:
    """Extract defect type from generated text."""
    text_lower = text.lower()
    for defect in DEFECT_TYPES:
        if defect in text_lower:
            return defect
    return None
def extract_size(text: str) -> Optional[str]:
    """Extract size category from generated text."""
    text_lower = text.lower()
    for size, keywords in SIZE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return size
    return None
def extract_location(text: str) -> Optional[str]:
    """Extract location from generated text."""
    text_lower = text.lower()
    for location, keywords in LOCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return location
    return None
def extract_attributes(text: str) -> Dict[str, Optional[str]]:
    """Extract all attributes from generated text."""
    return {
        'defect_type': extract_defect_type(text),
        'size': extract_size(text),
        'location': extract_location(text)
    }
# =============================================================================
# TIER 1: ATTRIBUTE-LEVEL EVALUATION
# =============================================================================
def evaluate_attributes(results: List[Dict], 
                        use_ground_truth_location: bool = True) -> Dict:
    """
    Evaluate attribute-level accuracy.
    
    Compares extracted attributes from generated_text against ground truth
    from mask annotations (category, defect_type, location).
    
    Args:
        results: List of result dicts with generated_text, category, defect_type, location
        use_ground_truth_location: Use location from bbox analysis as ground truth
        
    Returns:
        Dict with accuracy metrics for each attribute
    """
    defect_correct = 0
    size_correct = 0
    location_correct = 0
    total = 0
    
    for item in results:
        if item.get('defect_type') == 'good':
            continue  # Skip good samples for attribute evaluation
            
        total += 1
        generated_text = item.get('generated_text', '')
        extracted = extract_attributes(generated_text)
        
        # Defect type accuracy (check if ground truth defect type is mentioned)
        gt_defect = item.get('defect_type', '').replace('_', ' ')
        if extracted['defect_type'] and gt_defect:
            # Fuzzy match: check if any keyword from gt_defect is in generated
            gt_words = set(gt_defect.split())
            if gt_words & DEFECT_TYPES or extracted['defect_type'] in gt_defect:
                defect_correct += 1
        
        # Size accuracy
        gt_size = item.get('defect_size')
        if extracted['size'] and gt_size:
            if extracted['size'] == gt_size:
                size_correct += 1
        
        # Location accuracy
        if use_ground_truth_location:
            gt_location = item.get('location', '').lower()
            if extracted['location'] and gt_location:
                if extracted['location'] == gt_location or extracted['location'] in gt_location:
                    location_correct += 1
    
    return {
        'defect_type_accuracy': defect_correct / total if total > 0 else 0.0,
        'size_accuracy': size_correct / total if total > 0 else 0.0,
        'location_accuracy': location_correct / total if total > 0 else 0.0,
        'total_samples': total
    }
# =============================================================================
# TIER 2: TEXT QUALITY METRICS (OFFICIAL PYCOCOEVALCAP)
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
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    
    # Format data for pycocoevalcap
    gts = {id: [text] for id, text in reference_texts.items()}
    res = {id: [text] for id, text in generated_texts.items()}
    
    results = {}
    scorers = [
        (Bleu(4), "BLEU"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Spice(), "SPICE")
    ]
    
    for scorer, name in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
            if name == "BLEU":
                results["BLEU-1"] = score[0]
                results["BLEU-2"] = score[1]
                results["BLEU-3"] = score[2]
                results["BLEU-4"] = score[3]
            else:
                results[name] = score
        except Exception as e:
            warnings.warn(f"Error computing {name}: {e}")
            results[name] = None
    
    return {
        "method": "pycocoevalcap (official)",
        "metrics": results,
        "comparable_with_papers": True
    }
# =============================================================================
# SIMPLIFIED METRICS (FALLBACK)
# =============================================================================
def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase and split on non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())
def compute_meteor_simplified(candidate: str, reference: str,
                               alpha: float = 0.9,
                               beta: float = 3.0,
                               gamma: float = 0.5) -> float:
    """
    Simplified METEOR score (exact + stem matching only).
    
    WARNING: This does NOT include WordNet synonyms or paraphrase matching.
    Results are NOT directly comparable with published papers.
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    
    if not cand_tokens or not ref_tokens:
        return 0.0
    
    # Simple exact matching
    cand_set = set(cand_tokens)
    ref_set = set(ref_tokens)
    matches = len(cand_set & ref_set)
    
    if matches == 0:
        return 0.0
    
    precision = matches / len(cand_tokens)
    recall = matches / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Simple fragmentation penalty
    chunks = 1
    penalty = gamma * (chunks / matches) ** beta if matches > 0 else 0
    
    return max(0.0, min(1.0, f_mean * (1 - penalty)))
def compute_spice_simplified(candidate: str, reference: str) -> float:
    """
    Simplified SPICE-like score using keyword matching.
    
    WARNING: This is NOT true SPICE which uses dependency parsing and scene graphs.
    Results are NOT directly comparable with published papers.
    """
    def extract_keywords(text):
        text_lower = text.lower()
        keywords = set()
        
        for defect in DEFECT_TYPES:
            if defect in text_lower:
                keywords.add(f"defect:{defect}")
        
        for size, kws in SIZE_KEYWORDS.items():
            if any(kw in text_lower for kw in kws):
                keywords.add(f"size:{size}")
        
        for loc, kws in LOCATION_KEYWORDS.items():
            if any(kw in text_lower for kw in kws):
                keywords.add(f"location:{loc}")
        
        return keywords
    
    cand_kw = extract_keywords(candidate)
    ref_kw = extract_keywords(reference)
    
    if not ref_kw:
        return 1.0 if not cand_kw else 0.0
    if not cand_kw:
        return 0.0
    
    matches = cand_kw & ref_kw
    precision = len(matches) / len(cand_kw)
    recall = len(matches) / len(ref_kw)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)
def compute_bleu_simplified(candidate: str, reference: str, n: int = 4) -> float:
    """Simplified BLEU-n score."""
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return 0.0
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    precisions = []
    for i in range(1, n + 1):
        cand_ngrams = Counter(get_ngrams(cand_tokens, i))
        ref_ngrams = Counter(get_ngrams(ref_tokens, i))
        
        matches = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(matches / total)
    
    if 0 in precisions:
        return 0.0
    
    # Geometric mean
    log_sum = sum(math.log(p) for p in precisions) / n
    
    # Brevity penalty
    bp = 1 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(cand_tokens))
    
    return bp * math.exp(log_sum)
def _compute_simplified_metrics(generated_texts: Dict[str, str],
                                 reference_texts: Dict[str, str]) -> Dict:
    """Compute all simplified metrics."""
    meteor_scores = []
    spice_scores = []
    bleu_scores = []
    
    for id, gen_text in generated_texts.items():
        ref_text = reference_texts.get(id, '')
        if ref_text:
            meteor_scores.append(compute_meteor_simplified(gen_text, ref_text))
            spice_scores.append(compute_spice_simplified(gen_text, ref_text))
            bleu_scores.append(compute_bleu_simplified(gen_text, ref_text))
    
    return {
        "METEOR": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        "SPICE": sum(spice_scores) / len(spice_scores) if spice_scores else 0.0,
        "BLEU-4": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    }
# =============================================================================
# GROUND TRUTH GENERATION
# =============================================================================
def create_ground_truth(category: str,
                        defect_type: str,
                        location: str = None,
                        defect_size: str = None,
                        dataset_type: str = "mvtec") -> str:
    """
    Create ground truth text using hybrid approach:
    1. Try pre-defined descriptions first (high quality, professional)
    2. Enhance with location info if available
    3. Fallback to template if not in pre-defined dictionary
    
    Args:
        category: Object category (e.g., 'bottle')
        defect_type: Type of defect (e.g., 'broken_large')
        location: Location from 3x3 grid (optional)
        defect_size: Size category (optional, for fallback)
        dataset_type: 'mvtec' or 'visa' (default: 'mvtec')
        
    Returns:
        Ground truth text
    """
    # Select appropriate dictionary
    ground_truth_dict = MVTEC_GROUND_TRUTH if dataset_type == "mvtec" else VISA_GROUND_TRUTH
    
    # Try pre-defined description first
    if category in ground_truth_dict and defect_type in ground_truth_dict[category]:
        base_text = ground_truth_dict[category][defect_type]
        
        # Enhance with location info if available and not a 'good' sample
        if location and defect_type != 'good':
            # Normalize location (remove 'region' suffix if already present)
            location_clean = location.lower().replace(' region', '').strip()
            base_text += f" The defect is located at the {location_clean} region."
        
        return base_text
    
    # Fallback to template-based generation for unknown cases
    defect_clean = defect_type.replace('_', ' ')
    
    parts = [f"A {category} with"]
    
    if defect_size:
        parts.append(f"a {defect_size}")
    
    parts.append(f"{defect_clean} defect")
    
    if location:
        parts.append(f"at the {location} region")
    
    return ' '.join(parts)

# =============================================================================
# COMBINED EVALUATION
# =============================================================================
def evaluate_all(results: List[Dict],
                 use_location: bool = True,
                 use_standard_metrics: bool = True) -> Dict:
    """
    Complete evaluation with both attribute and text quality metrics.
    
    Args:
        results: List of dicts with generated_text, category, defect_type, location
        use_location: Include location in ground truth
        use_standard_metrics: Use pycocoevalcap (True) or simplified (False)
        
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
        defect_size = item.get('defect_size')
        reference_texts[image_id] = create_ground_truth(
            item.get('category', 'product'),
            item.get('defect_type', 'defect'),
            location,
            defect_size
        )
    
    # Tier 1: Attribute evaluation
    attribute_results = evaluate_attributes(results, use_location)
    
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
