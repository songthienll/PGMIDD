import torch
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMAIN KNOWLEDGE BASE  (Knowledge Guide module)
#    Defect vocabulary per MVTec-AD category + defect descriptions
# ─────────────────────────────────────────────────────────────────────────────
MVTEC_DEFECT_KNOWLEDGE = {
    "zipper": {
        "defects": ["fabric_border", "fabric_interior", "broken_teeth", "combined",
                    "rough", "split", "squeezed_teeth"],
        "descriptions": {
            "fabric_border":    "damage or fraying along the outer fabric edge of the zipper",
            "fabric_interior":  "damage, holes, or stains on the inner fabric area of the zipper",
            "broken_teeth":     "missing, broken, or misaligned zipper teeth",
            "combined":         "multiple defect types present simultaneously",
            "rough":            "rough or uneven texture on zipper surface",
            "split":            "the zipper tape is split or separated",
            "squeezed_teeth":   "zipper teeth that are squeezed, deformed, or compressed",
        },
        "normal_description": "a properly aligned zipper with intact teeth and clean fabric",
    },
    "metal_nut": {
        "defects": ["bent", "color", "flip", "scratch"],
        "descriptions": {
            "bent":    "the metal nut is bent or deformed from its original shape",
            "color":   "abnormal discoloration, rust, or staining on the nut surface",
            "flip":    "the metal nut is flipped upside-down or oriented incorrectly",
            "scratch": "surface scratches or abrasion marks on the nut",
        },
        "normal_description": "a flat, correctly oriented metal nut with uniform color and no surface damage",
    },
    "bottle": {
        "defects": ["broken_large", "broken_small", "contamination"],
        "descriptions": {
            "broken_large":   "large crack or break in the bottle",
            "broken_small":   "small chip or crack in the bottle",
            "contamination":  "foreign substance or contamination inside/on the bottle",
        },
        "normal_description": "an intact, clean bottle with no cracks or contamination",
    },
    "cable": {
        "defects": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation",
                    "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"],
        "descriptions": {
            "bent_wire":             "wire is bent or kinked",
            "cable_swap":            "cables are in wrong positions or swapped",
            "combined":              "multiple cable defects present",
            "cut_inner_insulation":  "inner insulation is cut or damaged",
            "cut_outer_insulation":  "outer insulation is cut or damaged",
            "missing_cable":         "one or more cables are missing",
            "missing_wire":          "one or more wires are missing",
            "poke_insulation":       "insulation has been poked or pierced",
        },
        "normal_description": "properly arranged cables with intact insulation and correct color order",
    },
    "capsule": {
        "defects": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
        "descriptions": {
            "crack":          "crack on the capsule surface",
            "faulty_imprint": "incorrect, missing, or blurry imprint on the capsule",
            "poke":           "small hole poked through the capsule",
            "scratch":        "surface scratch on the capsule",
            "squeeze":        "capsule is squeezed or deformed",
        },
        "normal_description": "a smooth, correctly imprinted capsule with no damage",
    },
    "carpet": {
        "defects": ["color", "cut", "hole", "metal_contamination", "thread"],
        "descriptions": {
            "color":               "abnormal color patch or stain on the carpet",
            "cut":                 "cut or tear in the carpet fibers",
            "hole":                "hole in the carpet",
            "metal_contamination": "metal object contamination embedded in carpet",
            "thread":              "loose or pulled thread on the carpet surface",
        },
        "normal_description": "uniform carpet texture with consistent color and no damage",
    },
    "grid": {
        "defects": ["bent", "broken", "glue", "metal_contamination", "thread"],
        "descriptions": {
            "bent":                "grid wires are bent out of shape",
            "broken":              "one or more grid wires are broken",
            "glue":                "glue residue on the grid",
            "metal_contamination": "metal contamination on the grid surface",
            "thread":              "thread caught in the grid",
        },
        "normal_description": "a regular grid pattern with straight, intact wires",
    },
    "hazelnut": {
        "defects": ["crack", "cut", "hole", "print"],
        "descriptions": {
            "crack": "crack on the hazelnut surface",
            "cut":   "cut mark on the hazelnut",
            "hole":  "hole on the hazelnut surface",
            "print": "abnormal print or marking on the hazelnut",
        },
        "normal_description": "a smooth hazelnut with uniform surface texture",
    },
    "leather": {
        "defects": ["color", "cut", "fold", "glue", "poke"],
        "descriptions": {
            "color": "abnormal color patch or stain on the leather",
            "cut":   "cut mark on the leather surface",
            "fold":  "leather is folded or creased",
            "glue":  "glue residue on the leather",
            "poke":  "small hole poked through the leather",
        },
        "normal_description": "smooth leather with uniform color and texture and no damage",
    },
    "pill": {
        "defects": ["color", "combined", "contamination", "crack", "faulty_imprint",
                    "pill_type", "scratch"],
        "descriptions": {
            "color":          "pill has wrong or uneven color",
            "combined":       "multiple defects present on the pill",
            "contamination":  "foreign substance on the pill",
            "crack":          "crack on the pill surface",
            "faulty_imprint": "incorrect or missing imprint on the pill",
            "pill_type":      "wrong pill type present",
            "scratch":        "scratch on the pill surface",
        },
        "normal_description": "a properly colored pill with correct imprint and no damage",
    },
    "screw": {
        "defects": ["manipulated_front", "scratch_head", "scratch_neck",
                    "thread_side", "thread_top"],
        "descriptions": {
            "manipulated_front": "screw head has been manipulated or damaged",
            "scratch_head":      "scratch on the screw head",
            "scratch_neck":      "scratch on the screw neck/shaft",
            "thread_side":       "damaged thread visible from the side",
            "thread_top":        "damaged thread visible from the top",
        },
        "normal_description": "a properly formed screw with intact threads and undamaged head",
    },
    "tile": {
        "defects": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
        "descriptions": {
            "crack":       "crack on the tile surface",
            "glue_strip":  "glue strip residue on the tile",
            "gray_stroke": "gray stroke or mark on the tile",
            "oil":         "oil stain on the tile surface",
            "rough":       "rough patch on otherwise smooth tile",
        },
        "normal_description": "a smooth, uniformly colored tile with no cracks or stains",
    },
    "toothbrush": {
        "defects": ["defective"],
        "descriptions": {
            "defective": "deformed, missing, or irregular bristles on the toothbrush",
        },
        "normal_description": "a toothbrush with evenly distributed, undamaged bristles",
    },
    "transistor": {
        "defects": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
        "descriptions": {
            "bent_lead":    "transistor lead is bent",
            "cut_lead":     "transistor lead is cut or broken",
            "damaged_case": "transistor case is cracked or damaged",
            "misplaced":    "transistor is incorrectly placed or oriented",
        },
        "normal_description": "a transistor with straight leads and undamaged case in correct position",
    },
    "wood": {
        "defects": ["color", "combined", "hole", "liquid", "scratch"],
        "descriptions": {
            "color":    "abnormal color patch or discoloration on the wood",
            "combined": "multiple defects present on the wood",
            "hole":     "hole in the wood surface",
            "liquid":   "liquid stain or spill on the wood",
            "scratch":  "scratch on the wood surface",
        },
        "normal_description": "wood with uniform grain pattern and no damage or stains",
    },
}

LOCATION_LABELS = [
    "top left", "top center", "top right",
    "center left", "center", "center right",
    "bottom left", "bottom center", "bottom right",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_knowledge_prompt(category: str) -> str:
    """Build a domain knowledge snippet for the given MVTec category."""
    info = MVTEC_DEFECT_KNOWLEDGE.get(category, {})
    if not info:
        return ""
    defect_list = ", ".join(info.get("defects", []))
    desc_lines = "\n".join(
        f"  - {k}: {v}" for k, v in info.get("descriptions", {}).items()
    )
    normal = info.get("normal_description", "")
    return (
        f"DOMAIN KNOWLEDGE for '{category}':\n"
        f"  Normal appearance: {normal}\n"
        f"  Possible defect types: {defect_list}\n"
        f"  Defect definitions:\n{desc_lines}\n"
    )


def build_cot_prompt(category: str, location_hint: str | None, valid_defects: list[str]) -> str:
    """
    Build a Chain-of-Thought prompt that:
      - Injects domain knowledge (Knowledge Guide)
      - Anchors location from mask bbox (removes location hallucination)
      - Constrains defect vocabulary (reduces type hallucination)
    """
    knowledge = get_knowledge_prompt(category)
    valid_str = ", ".join(valid_defects) if valid_defects else "unknown"

    location_instruction = ""
    if location_hint:
        location_instruction = (
            f"The defective region has been highlighted with a red contour in the image. "
            f"Based on the mask analysis, the defect is located at: {location_hint}.\n"
        )
    else:
        location_instruction = (
            "The defective region has been highlighted with a red contour in the image. "
            "Identify its location from: " + ", ".join(LOCATION_LABELS) + ".\n"
        )

    prompt = (
        f"{knowledge}\n"
        f"You are an industrial quality control inspector analyzing a '{category}' component.\n"
        f"{location_instruction}\n"
        f"Step 1 - Examine the highlighted region carefully.\n"
        f"Step 2 - Compare it with normal appearance: {MVTEC_DEFECT_KNOWLEDGE.get(category, {}).get('normal_description', 'a normal component')}.\n"
        f"Step 3 - Identify the defect type. Choose ONLY from this list: [{valid_str}].\n"
        f"Step 4 - Output your answer in this exact format:\n"
        f"  Defect: <defect_type>\n"
        f"  Location: <location>\n"
        f"  Description: <one sentence describing the defect>\n\n"
        f"Answer:"
    )
    return prompt


def parse_model_output(text: str, valid_defects: list[str], location_hint: str | None) -> dict:
    """
    Parse structured output from the model.
    Falls back gracefully if format is not followed.
    """
    result = {
        "defect_type": None,
        "location": location_hint,
        "description": text.strip(),
        "raw_output": text.strip(),
    }

    # Try to extract Defect:
    defect_match = re.search(r"Defect:\s*([^\n]+)", text, re.IGNORECASE)
    if defect_match:
        raw_defect = defect_match.group(1).strip().lower().replace(" ", "_")
        # Find closest valid defect (exact or substring match)
        matched = None
        for vd in valid_defects:
            if vd == raw_defect or vd in raw_defect or raw_defect in vd:
                matched = vd
                break
        result["defect_type"] = matched or raw_defect

    # Try to extract Location: — but only override if we don't have a mask-based hint
    if not location_hint:
        loc_match = re.search(r"Location:\s*([^\n]+)", text, re.IGNORECASE)
        if loc_match:
            raw_loc = loc_match.group(1).strip().lower()
            # Validate against known locations
            matched_loc = None
            for loc in LOCATION_LABELS:
                if loc in raw_loc:
                    matched_loc = loc
                    break
            result["location"] = matched_loc or raw_loc

    # Try to extract Description:
    desc_match = re.search(r"Description:\s*([^\n]+)", text, re.IGNORECASE)
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen2_vl_model(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Load Qwen2-VL model and processor."""
    print(f"Loading model: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded successfully!")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# 4. SINGLE-SAMPLE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_single_inference(
    item: dict,
    model,
    processor,
    max_tokens: int = 150,
    temperature: float = 0.1,
) -> dict:
    """
    Run Echo-inspired inference for a single image item.
    
    Key improvements over vanilla inference:
      - Domain knowledge injected (Knowledge Guide)
      - Location anchored from mask bbox (no hallucination)
      - Constrained defect vocabulary per category (no wrong type)
      - Chain-of-Thought structured prompt (Reasoning Expert)
    """
    category = item.get("category", "unknown")
    is_good = item.get("defect_type") == "good"
    image_path = item.get("annotated_path") or item.get("image_path")
    location_hint = item.get("location")  # Pre-computed from mask bbox

    # ── Good image: simple check ──────────────────────────────────────────
    if is_good:
        valid_defects = MVTEC_DEFECT_KNOWLEDGE.get(category, {}).get("defects", [])
        normal_desc = MVTEC_DEFECT_KNOWLEDGE.get(category, {}).get(
            "normal_description", "a normal component"
        )
        prompt_text = (
            f"You are an industrial quality inspector analyzing a '{category}' component.\n"
            f"Normal appearance: {normal_desc}\n"
            f"Examine this image carefully. Does it have any defects?\n"
            f"Possible defect types would be: {', '.join(valid_defects)}\n"
            f"Answer in this exact format:\n"
            f"  Defect: good\n"
            f"  Location: none\n"
            f"  Description: <one sentence describing what you see>\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]}]
    else:
        # ── Defective image: full Echo-style prompt ───────────────────────
        valid_defects = MVTEC_DEFECT_KNOWLEDGE.get(category, {}).get("defects", [])
        prompt_text = build_cot_prompt(category, location_hint, valid_defects)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]}]

    # ── Tokenize and generate ──────────────────────────────────────────────
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # ── Parse structured output ───────────────────────────────────────────
    parsed = parse_model_output(output_text, valid_defects, location_hint if not is_good else None)

    # Update item with predictions
    item = item.copy()
    item["generated_text"] = output_text
    item["predicted_defect_type"] = parsed["defect_type"] or ("good" if is_good else "unknown")
    item["predicted_location"] = parsed["location"]
    item["predicted_description"] = parsed["description"]

    # For good images, force defect = good
    if is_good:
        item["predicted_defect_type"] = "good"
        item["predicted_location"] = None

    return item


# ─────────────────────────────────────────────────────────────────────────────
# 5. BATCH INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def batch_inference(
    annotated_images: list[dict],
    model,
    processor,
    batch_size: int = 4,
    use_location: bool = True,
    max_tokens: int = 150,
    temperature: float = 0.1,
    show_progress: bool = True,
) -> list[dict]:
    """
    Run improved batch inference over all images.
    Processes one at a time (Qwen2-VL works best this way for quality).
    batch_size param kept for API compatibility but inference is sequential.
    """
    results = []
    iterable = tqdm(annotated_images, desc="Running Echo-style inference") if show_progress else annotated_images

    for item in iterable:
        try:
            result = run_single_inference(
                item=item,
                model=model,
                processor=processor,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)
        except Exception as e:
            print(f"\nError processing {item.get('image_name', '?')}: {e}")
            item = item.copy()
            item["generated_text"] = ""
            item["predicted_defect_type"] = "error"
            item["predicted_location"] = None
            item["predicted_description"] = ""
            results.append(item)

    return results
