import os
import re
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_qwen2_vl_model(model_name: str, device: str = "auto") -> Tuple:
    """
    Load Qwen2-VL model and processor.

    Args:
        model_name: Full HuggingFace model name or local path
        device: Device to load model ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Loading {model_name}...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    ).eval()

    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    print(f"Model loaded successfully! Device: {next(model.parameters()).device}")
    return model, processor
                            

# =============================================================================
# PROMPTS
# =============================================================================
def build_prompt(is_good: bool, location_hint: str = None, category: str = "product") -> str:
    """
    Build a compact, structured prompt for defect description.

    Args:
        is_good: Whether the image is non-defective
        location_hint: Location hint from 3x3 grid (e.g., "top left")
        category: Product category string

    Returns:
        Formatted prompt string
    """
    normal_desc = f"a normal {category} with no visible defects"
    valid_str = "scratch, crack, hole, stain, deformation, contamination, broken, missing_part, other"

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


def get_prompt(is_good: bool, location: str = "Center", category: str = "product") -> str:
    """
    Get appropriate prompt based on image type.
    Delegates to build_prompt for structured output.

    Args:
        is_good: Whether image is good (non-defective)
        location: Location from 3x3 grid
        category: Product category

    Returns:
        Formatted prompt string
    """
    return build_prompt(is_good=is_good, location_hint=location, category=category)


# =============================================================================
# OUTPUT PARSING
# =============================================================================
def parse_model_output(text: str, location_hint: str = None) -> dict:
    """
    Parse structured output from the model.
    Falls back gracefully if format is not followed.

    Args:
        text: Raw model output text
        location_hint: Original location hint passed to the prompt

    Returns:
        Dict with keys: defect_type, location, description, raw_output
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
        result["defect_type"] = defect_match.group(1).strip().lower().replace(" ", "_")

    # Try to extract Location: — only override if we don't have a mask-based hint
    if not location_hint:
        loc_match = re.search(r"Location:\s*([^\n]+)", text, re.IGNORECASE)
        if loc_match:
            raw_loc = loc_match.group(1).strip().lower()
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


# =============================================================================
# SINGLE IMAGE INFERENCE
# =============================================================================
def describe_defect(image_path: str,
                    model,
                    processor,
                    prompt: str = None,
                    max_tokens: int = 80,
                    temperature: float = 0.0) -> str:
    """
    Generate defect description for a single image.

    Args:
        image_path: Path to image
        model: Qwen2.5-VL model
        processor: Model processor
        prompt: Text prompt (uses default if None)
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature

    Returns:
        Generated description text
    """
    from qwen_vl_utils import process_vision_info

    if prompt is None:
        prompt = build_prompt(is_good=False)

    image = Image.open(image_path).convert('RGB')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response.strip()


def describe_defect_with_location(image_path: str,
                                   model,
                                   processor,
                                   location: str,
                                   category: str = "product",
                                   max_tokens: int = 100,
                                   temperature: float = 0.1) -> str:
    """Generate defect description with location context."""
    prompt = build_prompt(is_good=False, location_hint=location, category=category)
    return describe_defect(image_path, model, processor, prompt, max_tokens, temperature)


def describe_product_for_defects(image_path: str,
                                  model,
                                  processor,
                                  category: str = "product",
                                  max_tokens: int = 100,
                                  temperature: float = 0.1) -> str:
    """Check if a product image has any defects (for 'good' images)."""
    prompt = build_prompt(is_good=True, category=category)
    return describe_defect(image_path, model, processor, prompt, max_tokens, temperature)


def batch_inference(annotated_images: List[Dict],
                    model,
                    processor,
                    batch_size: int = 4,
                    use_location: bool = True,
                    max_tokens: int = 80,
                    temperature: float = 0.0,
                    show_progress: bool = True) -> List[Dict]:
    """
    Process images in true batches for GPU efficiency.

    Args:
        annotated_images: List of image dictionaries
        model: Qwen2.5-VL model
        processor: Model processor
        batch_size: Number of images per batch (default 4)
        use_location: Include location in prompt for defective images
        max_tokens: Maximum tokens per response
        temperature: Generation temperature (0 = greedy, fastest)
        show_progress: Show progress bar

    Returns:
        List of results with generated_text, predicted_defect_type,
        predicted_location, and predicted_description
    """
    from qwen_vl_utils import process_vision_info

    results = []

    # Setup progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(annotated_images), desc="Generating descriptions")
        except ImportError:
            pbar = None
    else:
        pbar = None

    # Process in batches
    for i in range(0, len(annotated_images), batch_size):
        batch = annotated_images[i:i + batch_size]

        try:
            batch_results = _process_batch(
                batch, model, processor,
                use_location, max_tokens, temperature
            )
            results.extend(batch_results)
        except Exception as e:
            print(f"Batch error at index {i}: {e}")
            # Fallback to sequential processing
            for item in batch:
                result = _process_single_fallback(
                    item, model, processor,
                    use_location, max_tokens, temperature
                )
                results.append(result)

        if pbar:
            pbar.update(len(batch))

    if pbar:
        pbar.close()

    return results


def _process_batch(batch: List[Dict],
                   model,
                   processor,
                   use_location: bool,
                   max_tokens: int,
                   temperature: float) -> List[Dict]:
    """Process a single batch of images."""
    from qwen_vl_utils import process_vision_info

    all_messages = []
    all_images = []
    all_meta = []

    for idx, item in enumerate(batch):
        try:
            is_good = item.get('defect_type') == 'good'
            category = item.get('category', 'product')
            location_hint = item.get('location') if use_location else None
            image_path = item.get('annotated_path') or item.get('image_path')

            if not image_path or not os.path.exists(image_path):
                print(f"  Skipping invalid path at index {idx}")
                continue

            prompt = build_prompt(is_good=is_good, location_hint=location_hint, category=category)

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"  Failed to open {image_path}: {e}")
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            all_messages.append(messages)

            img_inputs, _ = process_vision_info(messages)
            all_images.append(img_inputs[0] if img_inputs else image)
            all_meta.append({
                "is_good": is_good,
                "location_hint": location_hint,
            })

        except Exception as e:
            print(f"  Error processing item {idx}: {e}")
            continue

    if not all_messages:
        return []

    # Batch tokenization
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    inputs = processor(
        text=texts,
        images=all_images,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Batch generation
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    # Decode responses (strip input tokens)
    input_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, input_len:]
    responses = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Build results
    batch_results = []
    for item, response, meta in zip(batch, responses, all_meta):
        parsed = parse_model_output(response, location_hint=meta["location_hint"])
        out = item.copy()
        out["generated_text"] = response
        out["predicted_defect_type"] = "good" if meta["is_good"] else (parsed["defect_type"] or "unknown")
        out["predicted_location"] = None if meta["is_good"] else parsed["location"]
        out["predicted_description"] = parsed["description"]
        batch_results.append(out)

    return batch_results


def _process_single_fallback(item: Dict,
                              model,
                              processor,
                              use_location: bool,
                              max_tokens: int,
                              temperature: float) -> Dict:
    """Fallback: process single image when batch fails."""
    result = item.copy()

    try:
        is_good = item.get('defect_type') == 'good'
        category = item.get('category', 'product')
        location_hint = item.get('location') if use_location else None
        image_path = item.get('annotated_path') or item.get('image_path')

        if not image_path:
            result['generated_text'] = "Error: No image path found"
            result['predicted_defect_type'] = "error"
            result['predicted_location'] = None
            result['predicted_description'] = ""
            return result

        if is_good:
            description = describe_product_for_defects(image_path, model, processor, category, max_tokens, temperature)
        else:
            if use_location:
                description = describe_defect_with_location(image_path, model, processor, location_hint or "Center", category, max_tokens, temperature)
            else:
                description = describe_defect(image_path, model, processor, None, max_tokens, temperature)

        parsed = parse_model_output(description, location_hint=location_hint)
        result['generated_text'] = description
        result['predicted_defect_type'] = "good" if is_good else (parsed["defect_type"] or "unknown")
        result['predicted_location'] = None if is_good else parsed["location"]
        result['predicted_description'] = parsed["description"]

    except Exception as e:
        result['generated_text'] = f"Error: {str(e)}"
        result['predicted_defect_type'] = "error"
        result['predicted_location'] = None
        result['predicted_description'] = ""

    return result

