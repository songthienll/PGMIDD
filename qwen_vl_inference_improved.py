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

def build_prompt(category: str, location_hint: str | None, is_good: bool) -> str:
    """
    Compact prompt — keeps accuracy gains but is much shorter than the CoT version.
    Shorter prompt = fewer tokens to process per image = faster batching.
    """
    info = MVTEC_DEFECT_KNOWLEDGE.get(category, {})
    valid_defects = info.get("defects", [])
    normal_desc = info.get("normal_description", f"a normal {category}")
    valid_str = ", ".join(valid_defects)

    if is_good:
        return (
            f"Inspect this {category}. Normal: {normal_desc}. "
            f"Possible defects: {valid_str}. "
            f"Reply ONLY:\nDefect: good\nLocation: none\nDescription: <one sentence>"
        )

    loc_str = f"Defect region is at {location_hint}." if location_hint else "Defect region is highlighted in red."
    return (
        f"Inspect this {category}. {loc_str} "
        f"Normal: {normal_desc}. "
        f"Choose defect type from ONLY: [{valid_str}]. "
        f"Reply ONLY:\nDefect: <type>\nLocation: {location_hint or '<location>'}\nDescription: <one sentence>"
    )


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
    # Disable padding-side issues for batch inference
    processor.tokenizer.padding_side = "left"
    print("Model loaded successfully!")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# 4. BATCH INFERENCE  (true GPU batching — main speedup)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_batch(items: list[dict], processor) -> tuple[dict, list]:
    """
    Prepare a batch of items into a single padded tensor batch.
    Returns (model_inputs, list_of_meta) where meta holds per-item info for parsing.
    """
    texts = []
    all_image_inputs = []
    meta = []

    for item in items:
        is_good = item.get("defect_type") == "good"
        category = item.get("category", "unknown")
        location_hint = item.get("location")
        image_path = item.get("annotated_path") or item.get("image_path")

        prompt_text = build_prompt(category, location_hint, is_good)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        texts.append(text)
        all_image_inputs.extend(image_inputs)
        meta.append({
            "is_good": is_good,
            "category": category,
            "location_hint": location_hint,
            "valid_defects": MVTEC_DEFECT_KNOWLEDGE.get(category, {}).get("defects", []),
        })

    inputs = processor(
        text=texts,
        images=all_image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs, meta


def batch_inference(
    annotated_images: list[dict],
    model,
    processor,
    batch_size: int = 8,          
    use_location: bool = True,
    max_tokens: int = 60,         
    temperature: float = 0.0,
    show_progress: bool = True,
) -> list[dict]:
    """
    Fast batched inference with true GPU parallelism.

    Speedup vs original:
      - Real batch processing (N images in one forward pass)
      - Greedy decoding (temperature=0, no sampling overhead)
      - Compact prompts (fewer tokens to encode)
      - Estimated: ~5-10x faster than sequential on T4
    """
    results = []
    device = next(model.parameters()).device

    # Split into batches
    batches = [annotated_images[i:i+batch_size] for i in range(0, len(annotated_images), batch_size)]
    iterable = tqdm(batches, desc=f"Inference") if show_progress else batches

    for batch_items in iterable:
        try:
            inputs, meta = _prepare_batch(batch_items, processor)
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,          # greedy — fastest
                    use_cache=True,
                )

            # Decode only the newly generated tokens (strip input)
            input_len = inputs["input_ids"].shape[1]
            new_ids = generated_ids[:, input_len:]
            decoded = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Parse each output and attach back to item
            for item, output_text, m in zip(batch_items, decoded, meta):
                parsed = parse_model_output(
                    output_text,
                    m["valid_defects"],
                    m["location_hint"] if use_location else None,
                )
                out = item.copy()
                out["generated_text"] = output_text
                out["predicted_defect_type"] = "good" if m["is_good"] else (parsed["defect_type"] or "unknown")
                out["predicted_location"] = None if m["is_good"] else parsed["location"]
                out["predicted_description"] = parsed["description"]
                results.append(out)

        except torch.cuda.OutOfMemoryError:
            # Fallback: halve batch and retry
            torch.cuda.empty_cache()
            print(f"\nOOM with batch_size={batch_size}, retrying with batch_size={max(1, batch_size//2)}")
            for item in batch_items:
                try:
                    inputs, meta = _prepare_batch([item], processor)
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                    input_len = inputs["input_ids"].shape[1]
                    output_text = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
                    m = meta[0]
                    parsed = parse_model_output(output_text, m["valid_defects"], m["location_hint"])
                    out = item.copy()
                    out["generated_text"] = output_text
                    out["predicted_defect_type"] = "good" if m["is_good"] else (parsed["defect_type"] or "unknown")
                    out["predicted_location"] = None if m["is_good"] else parsed["location"]
                    out["predicted_description"] = parsed["description"]
                    results.append(out)
                except Exception as e2:
                    print(f"Error on single item {item.get('image_name','?')}: {e2}")
                    out = item.copy()
                    out.update({"generated_text": "", "predicted_defect_type": "error",
                                "predicted_location": None, "predicted_description": ""})
                    results.append(out)

        except Exception as e:
            print(f"\nBatch error: {e}")
            for item in batch_items:
                out = item.copy()
                out.update({"generated_text": "", "predicted_defect_type": "error",
                            "predicted_location": None, "predicted_description": ""})
                results.append(out)

    return results
