"""
Qwen2-VL Integration Module for Defect Detection
Optimized for Google Colab T4 GPU with true batch processing.
"""
import os
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path
# =============================================================================
# MODEL REGISTRY
# =============================================================================
QWEN_VL_MODELS = {
    'qwen2-vl-2b': {
        'name': 'Qwen/Qwen2-VL-2B-Instruct',
        'memory_gb': 4,
        'recommended_batch_size': 4
    },
    'qwen2-vl-7b': {
        'name': 'Qwen/Qwen2-VL-7B-Instruct',
        'memory_gb': 14,
        'recommended_batch_size': 2
    },
    'qwen2.5-vl-3b': {
        'name': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'memory_gb': 6,
        'recommended_batch_size': 4
    },
    'qwen2.5-vl-7b': {
        'name': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'memory_gb': 14,
        'recommended_batch_size': 2
    }
}
DEFAULT_MODEL = 'qwen2-vl-2b'
def get_available_models() -> Dict:
    """Get list of available models with their specifications."""
    return QWEN_VL_MODELS.copy()
def select_model_for_gpu(gpu_memory_gb: float = None) -> str:
    """
    Select appropriate model based on available GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory (auto-detected if None)
        
    Returns:
        Model key from QWEN_VL_MODELS
    """
    if gpu_memory_gb is None:
        info = get_gpu_memory_info()
        gpu_memory_gb = info.get('free_memory_gb', 4)
    
    # Select largest model that fits
    suitable = []
    for key, spec in QWEN_VL_MODELS.items():
        if spec['memory_gb'] <= gpu_memory_gb * 0.8:  # 80% safety margin
            suitable.append((key, spec['memory_gb']))
    
    if suitable:
        return max(suitable, key=lambda x: x[1])[0]
    
    return DEFAULT_MODEL
# =============================================================================
# MODEL LOADING
# =============================================================================
def load_qwen2_vl_model(model_name: str = None,
                        model_key: str = None,
                        device: str = "auto") -> Tuple:
    """
    Load Qwen2-VL model optimized for T4 GPU.
    
    Args:
        model_name: Full HuggingFace model name (e.g., 'Qwen/Qwen2-VL-2B-Instruct')
        model_key: Short key from QWEN_VL_MODELS (e.g., 'qwen2-vl-2b')
        device: Device to load model ('auto', 'cuda', 'cpu')
        
    Returns:
        Tuple of (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    # Resolve model name
    if model_name is None:
        if model_key and model_key in QWEN_VL_MODELS:
            model_name = QWEN_VL_MODELS[model_key]['name']
        else:
            model_key = select_model_for_gpu()
            model_name = QWEN_VL_MODELS[model_key]['name']
    
    print(f"Loading {model_name}...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    print(f"Model loaded successfully! Device: {next(model.parameters()).device}")
    return model, processor
# Backward compatibility alias
load_qwen_model = load_qwen2_vl_model
# =============================================================================
# PROMPTS
# =============================================================================
DEFECT_PROMPT = """Provide a concise description of the industrial defect in the red-outlined area. 
Include the object type, defect type, location and visual details. 
Example: Metal surface has a scratch with a linear mark at the bottom right."""

NORMAL_PROMPT = """Inspect this image for defects.
Describe it in one sentence following this format: "A normal [product type] with no visible defects."
Example: A normal toothbrush with no visible defects."""
def get_prompt(is_good: bool, location: str = "Center", category: str = "product") -> str:
    """
    Get appropriate prompt based on image type.
    
    Args:
        is_good: Whether image is good (non-defective)
        location: Location from 3x3 grid (kept for compatibility, not used in new prompts)
        category: Product category (kept for compatibility, not used in new prompts)
        
    Returns:
        Formatted prompt string
    """
    
    if is_good:
        return NORMAL_PROMPT
    else:
        return DEFECT_PROMPT
# =============================================================================
# SINGLE IMAGE INFERENCE
# =============================================================================
def describe_defect(image_path: str,
                    model,
                    processor,
                    prompt: str = None,
                    max_tokens: int = 100,
                    temperature: float = 0.1) -> str:
    """
    Generate defect description for a single image.
    
    Args:
        image_path: Path to image
        model: Qwen2-VL model
        processor: Model processor
        prompt: Text prompt (uses default if None)
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        
    Returns:
        Generated description text
    """
    from qwen_vl_utils import process_vision_info
    
    if prompt is None:
        prompt = "Describe any defects visible in this image. Be concise."
    
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
    prompt = get_prompt(is_good=False, location=location, category=category)
    return describe_defect(image_path, model, processor, prompt, max_tokens, temperature)
def describe_product_for_defects(image_path: str,
                                  model,
                                  processor,
                                  category: str = "product",
                                  max_tokens: int = 100,
                                  temperature: float = 0.1) -> str:
    """Check if a product image has any defects (for 'good' images)."""
    prompt = get_prompt(is_good=True, category=category)
    return describe_defect(image_path, model, processor, prompt, max_tokens, temperature)
# =============================================================================
# TRUE BATCH PROCESSING
# =============================================================================
def batch_inference(annotated_images: List[Dict],
                               model,
                               processor,
                               batch_size: int = 4,
                               use_location: bool = True,
                               max_tokens: int = 100,
                               temperature: float = 0.1,
                               show_progress: bool = True) -> List[Dict]:
    """
    Process images in true batches for GPU efficiency.
    
    Args:
        annotated_images: List of image dictionaries
        model: Qwen2-VL model
        processor: Model processor
        batch_size: Number of images per batch (default 4)
        use_location: Include location in prompt for defective images
        max_tokens: Maximum tokens per response
        temperature: Generation temperature
        show_progress: Show progress bar
        
    Returns:
        List of results with generated_text
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
        batch = annotated_images[i:i+batch_size]
        
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
    
    # Prepare batch inputs
    all_messages = []
    all_images = []
    
    for item in batch:
        is_good = item.get('defect_type') == 'good'
        category = item.get('category', 'product')
        
        if is_good:
            image_path = item.get('image_path')
            prompt = get_prompt(is_good=True, category=category)
        else:
            image_path = item.get('annotated_path', item.get('image_path'))
            location = item.get('location', 'Center') if use_location else 'Center'
            prompt = get_prompt(is_good=False, location=location, category=category)
        
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
        
        all_messages.append(messages)
        
        # Process vision info
        img_inputs, _ = process_vision_info(messages)
        all_images.append(img_inputs[0] if img_inputs else image)
    
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
    
    # Decode responses
    input_lengths = [len(ids) for ids in inputs.input_ids]
    responses = []
    
    for idx, (gen_ids, in_len) in enumerate(zip(generated_ids, input_lengths)):
        trimmed = gen_ids[in_len:]
        response = processor.decode(trimmed, skip_special_tokens=True)
        responses.append(response.strip())
    
    # Build results
    batch_results = []
    for item, response in zip(batch, responses):
        result = item.copy()
        result['generated_text'] = response
        batch_results.append(result)
    
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
        
        if is_good:
            image_path = item.get('image_path')
            description = describe_product_for_defects(
                image_path, model, processor, category, max_tokens, temperature
            )
        else:
            image_path = item.get('annotated_path', item.get('image_path'))
            location = item.get('location', 'Center')
            
            if use_location:
                description = describe_defect_with_location(
                    image_path, model, processor, location, category, max_tokens, temperature
                )
            else:
                description = describe_defect(
                    image_path, model, processor, None, max_tokens, temperature
                )
        
        result['generated_text'] = description
        
    except Exception as e:
        result['generated_text'] = f"Error: {str(e)}"
    
    return result
# Legacy function for backward compatibility
def batch_inference_b(annotated_images: List[Dict],
                    model,
                    processor,
                    use_location: bool = True,
                    max_tokens: int = 100,
                    temperature: float = 0.1,
                    show_progress: bool = True) -> List[Dict]:
    """
    Process images sequentially (legacy, for compatibility).
    Use batch_inference_optimized for better performance.
    """
    return batch_inference(
        annotated_images, model, processor,
        batch_size=1,  # Sequential
        use_location=use_location,
        max_tokens=max_tokens,
        temperature=temperature,
        show_progress=show_progress
    )
if __name__ == "__main__":
    print("Qwen2-VL Inference Module\n" + "="*50)
    
    print("\n1. Available Models:")
    for key, spec in QWEN_VL_MODELS.items():
        print(f"   {key}: {spec['name']} ({spec['memory_gb']}GB)")
    
    print("\n2. GPU Info:")
    info = get_gpu_memory_info()
    print(f"   GPU: {info['gpu_name']}")
    print(f"   Memory: {info.get('total_memory_gb', 'N/A')} GB")
    
    print("\n3. Recommended Model:")
    recommended = select_model_for_gpu()

    print(f"   {recommended}")




