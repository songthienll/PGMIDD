"""
PGMIDD Gradio Demo

Pipeline:
  Upload image → User selects model
  → Detection (generic prompt or class-agnostic)
  → Annotate → If defect found → Qwen describe + Q&A
  → If no defect → Green banner "GOOD"

Detection Models:
  - GroundingDINO Base: pretrained, generic defect prompt
  - GroundingDINO LoRA: fine-tuned rank=8 14-modules, generic defect prompt
  - OWL-ViT Base: pretrained OWL-ViT v2, no prompt needed
  - OWL-ViT LoRA: OWL-ViT v2 fine-tuned with LoRA from My Drive/epoch_5

VLM: Qwen2.5-VL-7B fine-tuned (describe + Q&A)

Run locally:
  pip install -r deploy/requirements_runpod.txt
  python notebooks/gradio_demo.py --port 7860 --share
"""

import os, re, warnings, argparse, json, uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import gradio as gr

# Config
BOX_THRESHOLD  = 0.25   # lowered from 0.35 for better recall (detect more defects)
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD  = 0.40
MAX_BOXES      = 10   # maximum bounding boxes to display

# Generic defect prompt for GDINO (matches training caption format: "object . defect . location .")
GDINO_GENERIC_PROMPT = "object . defect damage scratch crack hole ."

# OWL-ViT v2 text-prompted detection (transformers)
_owlvit_model = None
_owlvit_processor = None
_owlvit_load_error = None

# OWL-ViT LoRA (fine-tuned adapter at My Drive/epoch_5)
_owlvit_lora_model = None
_owlvit_lora_processor = None
_owlvit_lora_load_error = None
OWL_LORA_DIR = "/content/drive/MyDrive/epoch_5"  # adapter_model.safetensors + adapter_config.json

def load_owlvit():
    global _owlvit_model, _owlvit_processor, _owlvit_load_error
    if _owlvit_model is not None:
        return _owlvit_model, _owlvit_processor
    if _owlvit_load_error:
        raise RuntimeError(_owlvit_load_error)
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("[OWL-ViT] Loading OWL-ViT v2...")
        _owlvit_processor = AutoProcessor.from_pretrained(
            "google/owlvitv2-base-patch32-ensemble",
            token=hf_token, trust_remote_code=True,
        )
        _owlvit_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "google/owlvitv2-base-patch32-ensemble",
            token=hf_token, trust_remote_code=True,
        )
        _owlvit_model.to(DEVICE).eval()
        print("[OWL-ViT] Ready.")
        return _owlvit_model, _owlvit_processor
    except Exception as e:
        _owlvit_load_error = str(e)
        print(f"[OWL-ViT] Load failed: {e} — OWL-ViT disabled.")
        raise RuntimeError(f"OWL-ViT unavailable: {e}")

def load_owlvit_lora():
    """Load OWL-ViT base + LoRA adapter from OWL_LORA_DIR."""
    global _owlvit_lora_model, _owlvit_lora_processor, _owlvit_lora_load_error
    if _owlvit_lora_model is not None:
        return _owlvit_lora_model, _owlvit_lora_processor
    if _owlvit_lora_load_error:
        raise RuntimeError(_owlvit_lora_load_error)
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from peft import PeftModel
        print("[OWL-LoRA] Loading OWL-ViT base + LoRA adapter...")
        _owlvit_lora_processor = AutoProcessor.from_pretrained(
            "google/owlvitv2-base-patch32-ensemble",
            token=hf_token, trust_remote_code=True,
        )
        _owlvit_lora_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "google/owlvitv2-base-patch32-ensemble",
            token=hf_token, trust_remote_code=True,
        )
        print(f"[OWL-LoRA] Applying LoRA adapter from: {OWL_LORA_DIR}")
        _owlvit_lora_model = PeftModel.from_pretrained(
            _owlvit_lora_model, OWL_LORA_DIR, adapter_name="default",
        )
        print("[OWL-LoRA] Merging LoRA into base model...")
        _owlvit_lora_model = _owlvit_lora_model.merge_and_unload()
        _owlvit_lora_model.to(DEVICE).eval()
        print("[OWL-LoRA] Ready.")
        return _owlvit_lora_model, _owlvit_lora_processor
    except Exception as e:
        _owlvit_lora_load_error = str(e)
        print(f"[OWL-LoRA] Load failed: {e} — OWL-ViT LoRA disabled.")
        raise RuntimeError(f"OWL-ViT LoRA unavailable: {e}")


def run_owlvit(model, processor, image_path: str, box_thresh: float, query: str = "defect object"):
    """OWL-ViT v2 text-prompted detection."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=query, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # OWL-ViT v2 output fields:
    #   logits:   (batch, num_queries, seq_len) — no seq_lens field in output
    #   pred_boxes: (batch, num_queries, 4) cxcywh normalized
    logits = outputs.logits            # (batch, num_queries, seq_len)
    boxes  = outputs.pred_boxes        # (batch, num_queries, 4) cxcywh norm
    batch_logits = logits[0]          # (num_queries, seq_len)
    batch_boxes  = boxes[0]           # (num_queries, 4)

    # Text-prompted: max logit across ALL tokens per query → score
    # Higher logit = stronger match between image region and text query
    scores = batch_logits.max(dim=1).values  # (num_queries,)
    scores = scores.sigmoid()

    # Filter by threshold
    mask = scores > box_thresh
    if mask.sum() == 0:
        return [], []

    sel_scores = scores[mask]
    sel_boxes  = batch_boxes[mask]      # tensor (k, 4)

    # Sort by score descending, convert to numpy for safe indexing
    order = sel_scores.argsort(descending=True)
    sel_scores_np = sel_scores[order].cpu().tolist()
    sel_boxes_np  = sel_boxes[order].cpu().numpy()   # (k, 4) numpy array

    # Convert cxcywh normalized → xyxy pixel with clipping
    w, h = image.size
    xyxy = []
    for i in range(len(sel_boxes_np)):
        cx, cy, bw, bh = sel_boxes_np[i]   # numpy floats — safe to unpack
        xyxy.append([
            float(max(0.0, min(w, (cx - bw/2) * w))),
            float(max(0.0, min(h, (cy - bh/2) * h))),
            float(max(0.0, min(w, (cx + bw/2) * w))),
            float(max(0.0, min(h, (cy + bh/2) * h))),
        ])
    return xyxy, sel_scores_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_DISABLED = os.environ.get("QWEN_DISABLED", "0") == "1"
warnings.filterwarnings("ignore")

# Apply GroundingDINO patches BEFORE model loading 
# These patches must run BEFORE load_gdino() is called.
# Patch 1: MultiScaleDeformableAttention → CPU fallback (no CUDA kernel on Colab)
# Patch 2: bertwarper → fix empty cate_to_token_mask crash
def _apply_gdino_patches():
    import sys as _sys
    _sys.path.insert(0, str(GDINO_ROOT))
    import groundingdino.models.GroundingDINO.ms_deform_attn as _msda
    import torch

    # Patch 1: bypass CUDA _C, redirect forward to CPU path
    _msda._C = None
    _orig_fwd = _msda.MultiScaleDeformableAttention.forward

    def _cpu_fwd(self, query, key=None, value=None, query_pos=None,
                  key_padding_mask=None, reference_points=None,
                  spatial_shapes=None, level_start_index=None, **kwargs):
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1,
            ).to(device=reference_points.device, dtype=reference_points.dtype)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4, "
                              "got {}.".format(reference_points.shape[-1]))
        output = _msda.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output

    _msda.MultiScaleDeformableAttention.forward = _cpu_fwd

    # Patch 2: bertwarper empty cate_to_token_mask crash
    import groundingdino.models.GroundingDINO.bertwarper as _bw
    _orig_fn = _bw.generate_masks_with_special_tokens_and_transfer_map

    def _safe_masks(token_ids, special_tokens_len=2):
        result = _orig_fn(token_ids, special_tokens_len)
        text_mask, position_ids, cate_list = result
        fixed = []
        for inner in cate_list:
            if len(inner) == 0:
                dummy = text_mask.new_zeros(1, text_mask.shape[1])
                fixed.append(dummy)
            else:
                fixed.append(torch.stack(inner, dim=0))
        return text_mask, position_ids, fixed

    _bw.generate_masks_with_special_tokens_and_transfer_map = _safe_masks
    print("[Patch] GroundingDINO patches applied — CUDA bypass + bertwarper fix")


# Paths
WEIGHTS_DIR   = Path(os.environ.get("WEIGHTS_DIR", "/workspace/weights"))
GDINO_ROOT    = Path(os.environ.get("GDINO_ROOT",  "/workspace/GroundingDINO"))
GDINO_CONFIG = str(GDINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
BASE_WEIGHTS = str(WEIGHTS_DIR / "groundingdino_swint_ogc.pth")
# LoRA adapter: contains adapter_model.safetensors + adapter_config.json (NOT merged .pth)
LORA_ADAPTER_DIR = str(WEIGHTS_DIR / "gd_lora/model")
GDINO_LORA_DIR = "/content/drive/MyDrive/AIP491/results/gd_lora/model"  # Colab absolute path

_LOCAL_ROOT = Path(__file__).parent.resolve()
if (_LOCAL_ROOT / "groundingdino_swint_ogc.pth").exists():
    BASE_WEIGHTS = str(_LOCAL_ROOT / "groundingdino_swint_ogc.pth")
    GDINO_CONFIG = str(_LOCAL_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
if (_LOCAL_ROOT / "groundingdino_swint_ogc.pth").exists():
    BASE_WEIGHTS = str(_LOCAL_ROOT / "groundingdino_swint_ogc.pth")
    GDINO_CONFIG = str(_LOCAL_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# LoRA adapter — check both paths (local relative vs absolute)
_lora_local = _LOCAL_ROOT / "gd_lora" / "model"
if _lora_local.exists():
    LORA_ADAPTER_DIR = str(_lora_local)

# Model Cache 
_gdino_model   = None
_gdino_lora_model = None
_qwen_proc     = None
_qwen_model    = None

#  Session Store (in-memory) 
# Format: {session_id: {"image": PIL, "description": str, "bboxes": list, "scores": list, "defect": str, "location": str, "messages": []}}
_sessions = {}


def get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "image": None,
            "description": "",
            "bboxes": [],
            "scores": [],
            "defect": "",
            "location": "",
            "category": "",
            "messages": [],
        }
    return _sessions[session_id]


def load_gdino():
    """Load GroundingDINO using build_model + torch.load (avoids groundingdino.util.inference import)."""
    global _gdino_model
    if _gdino_model is not None:
        return _gdino_model

    import sys
    sys.path.insert(0, str(GDINO_ROOT))
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util import box_ops

    print("[GD] Loading GroundingDINO Base...")
    args = SLConfig.fromfile(GDINO_CONFIG)
    _gdino_model = build_model(args)

    ckpt = torch.load(BASE_WEIGHTS, map_location=DEVICE)
    _gdino_model.load_state_dict(ckpt["model"], strict=False)
    _gdino_model.to(DEVICE).eval()

    # Load tokenizer from model config (required for caption encoding)
    text_encoder_type = getattr(args, "text_encoder_type", "bert-base-uncased")
    from transformers import AutoTokenizer
    _gdino_model.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

    print("[GD] Ready.")
    return _gdino_model


def load_gdino_lora():
    """
    Load GroundingDINO base + LoRA adapter (rank=8, 14 modules).
    Loads base model, then applies LoRA adapter from LORA_ADAPTER_DIR.
    """
    global _gdino_lora_model
    if _gdino_lora_model is not None:
        return _gdino_lora_model

    import sys
    sys.path.insert(0, str(GDINO_ROOT))
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from peft import PeftModel, LoraConfig

    print("[GD-LoRA] Loading base model + LoRA adapter...")
    args = SLConfig.fromfile(GDINO_CONFIG)
    _gdino_lora_model = build_model(args)

    # Load base weights
    base_ckpt = torch.load(BASE_WEIGHTS, map_location=DEVICE)
    _gdino_lora_model.load_state_dict(base_ckpt["model"], strict=False)

    # Load LoRA adapter (adapter_model.safetensors + adapter_config.json)
    print(f"[GD-LoRA] Applying LoRA adapter from: {GDINO_LORA_DIR}")
    _gdino_lora_model = PeftModel.from_pretrained(
        _gdino_lora_model,
        GDINO_LORA_DIR,
        adapter_name="default",
    )
    # Merge LoRA into base weights for single forward pass
    print("[GD-LoRA] Merging LoRA into base model...")
    _gdino_lora_model = _gdino_lora_model.merge_and_unload()
    _gdino_lora_model.to(DEVICE).eval()

    # Tokenizer
    text_encoder_type = getattr(args, "text_encoder_type", "bert-base-uncased")
    from transformers import AutoTokenizer
    _gdino_lora_model.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

    print("[GD-LoRA] Ready.")
    return _gdino_lora_model


def load_qwen_vl():
    global _qwen_proc, _qwen_model
    if _qwen_model is not None:
        return _qwen_proc, _qwen_model
    if QWEN_DISABLED:
        return None, None
    hf_token = os.environ.get("HF_TOKEN", None)

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForConditionalGeneration as AutoModelForVision2Seq

    MODEL_NAME = "songthienll/qwen2.5-vl-7b-mvtec-visa"
    print("[Qwen] Loading Qwen2.5-VL-7B fine-tuned...")
    _qwen_proc  = AutoProcessor.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
    _qwen_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        token=hf_token, trust_remote_code=True,
    )
    _qwen_model.eval()
    print("[Qwen] Ready.")
    return _qwen_proc, _qwen_model


# Qwen: Generate Defect Description 
def describe_defect(image: Image.Image) -> str:
    proc, model = load_qwen_vl()
    if model is None:
        return "Defect: defect\nLocation: unknown\nDescription: Qwen VL unavailable."

    prompt = (
        "You are an industrial defect inspector. "
        "Describe the defect concisely using this exact format:\n"
        "Defect: <defect_type>\n"
        "Location: <location>\n"
        "Description: <1-sentence description>"
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the defect in this image."},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(DEVICE, torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = proc.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text.strip()


# Qwen: Follow-up Q&A
def qwen_qa(session_id: str, user_message: str) -> str:
    """
    Context-aware Q&A using Qwen VL.
    Injects detection context + conversation history into the prompt.
    """
    proc, model = load_qwen_vl()
    if model is None:
        return "Qwen VL is unavailable. Please run detection first."

    session = get_or_create_session(session_id)
    if session["description"] == "":
        return "Please run detection first before asking questions."

    # Build context block
    n_dets = len(session["bboxes"])
    if n_dets == 0:
        context_block = (
            f"- Ảnh đã phân tích: sản phẩm trong ảnh không có defect rõ ràng.\n"
            f"- Mô tả: {session['description']}\n"
        )
    else:
        bbox_str = ", ".join(
            f"(x1={int(b[0])}, y1={int(b[1])}, x2={int(b[2])}, y2={int(b[3])}, conf={s:.2f})"
            for b, s in zip(session["bboxes"], session["scores"])
        )
        context_block = (
            f"- Defect đã detect: {session['defect']}\n"
            f"- Vị trí: {session['location'] or 'unknown'}\n"
            f"- Số lượng bbox: {n_dets}\n"
            f"- Chi tiết bboxes: {bbox_str}\n"
            f"- Mô tả từ Qwen: {session['description']}\n"
        )

    # Conversation history (last 3 turns)
    history = session["messages"][-6:]  # max 3 user + 3 assistant
    history_block = ""
    if history:
        history_lines = []
        for m in history:
            role = "Inspector" if m["role"] == "assistant" else "User"
            history_lines.append(f"- {role}: {m['content']}")
        history_block = "\nLịch sử hội thoại:\n" + "\n".join(history_lines) + "\n"

    system_prompt = (
        "Bạn là một inspector AI chuyên về kiểm tra lỗi công nghiệp. "
        "Dựa trên thông tin phân tích ảnh dưới đây, trả lời câu hỏi của user một cách chi tiết và hữu ích.\n\n"
        f"Thông tin phân tích ảnh:\n{context_block}\n"
        f"{history_block}"
        "Trả lời bằng tiếng Việt, ngắn gọn, có tính thực tế."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": session["image"]},
            {"type": "text", "text": user_message},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[session["image"]], return_tensors="pt").to(DEVICE, torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = proc.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    # Save to history
    session["messages"].append({"role": "user", "content": user_message})
    session["messages"].append({"role": "assistant", "content": answer})

    return answer


# Parse + Build Query
def parse_description(text: str) -> dict:
    defect   = re.search(r"Defect:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    location = re.search(r"Location:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    raw_defect = defect.group(1).strip() if defect else ""
    # Only skip if defect field is genuinely empty or explicitly "good"/"none"
    skip_keywords = {"", "none", "good", "normal", "null", "n/a", "na"}
    skip_guard = raw_defect.lower() in skip_keywords
    return {
        "defect": raw_defect if raw_defect else "defect",
        "location": location.group(1).strip() if location else "",
        "skip_guard": skip_guard,
    }


def build_query(category: str, defect: str, location: str = "") -> str:
    """
    Build grounded query for GroundingDINO.
    Direct defect type — no article prefix, no dot separator.
    """
    defect = defect.replace("_", " ").strip().lower()
    if defect in ("", "defect", "unknown", "none"):
        defect = "defect"
    if category and category.strip().lower() not in ("", "object", "none"):
        return f"{category.strip().lower()} {defect}"
    return defect


# NMS + top-K
def apply_nms(bboxes, scores, iou_threshold=NMS_THRESHOLD, max_boxes=MAX_BOXES):
    if not bboxes:
        return [], []
    keep = torchvision.ops.nms(
        torch.tensor(bboxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_threshold,
    )
    bboxes = [bboxes[i] for i in keep.tolist()]
    scores = [scores[i] for i in keep.tolist()]
    # Top-K: keep only highest-confidence boxes
    if len(bboxes) > max_boxes:
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bboxes = [bboxes[i] for i in sorted_idx[:max_boxes]]
        scores = [scores[i] for i in sorted_idx[:max_boxes]]
    return bboxes, scores


# GroundingDINO Inference


# GroundingDINO Inference
def run_gdino(model, image_path: str, query: str, box_thresh: float, text_thresh: float):
    """Run GroundingDINO using groundingdino.util.inference.predict()."""
    from groundingdino.util.inference import load_image, predict as gd_predict
    from groundingdino.util import box_ops

    image_src, image_t = load_image(image_path)
    h, w = image_src.shape[:2]

    boxes, logits, phrases = gd_predict(
        model=model, image=image_t, caption=query,
        box_threshold=box_thresh, text_threshold=text_thresh, device=DEVICE,
    )
    if len(boxes) == 0:
        return [], []

    boxes_abs = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
        [w, h, w, h], dtype=torch.float32)
    bboxes = boxes_abs.cpu().numpy().tolist()
    scores = logits.cpu().numpy().tolist()
    return apply_nms(bboxes, scores)


# Annotation
def annotate(img, bboxes, scores, label):
    if isinstance(img, Image.Image):
        img = np.array(img)[:, :, ::-1].copy()  # .copy() makes it contiguous for OpenCV 4.13
    h, w = img.shape[:2]
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = [int(max(0, min(v, d))) for v, d in zip(bbox, [w, h, w, h])]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1-th-8)), (x1+tw, y1), (0, 0, 255), -1)
        cv2.putText(img, txt, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Main Pipeline (Flow 2 — Detection First)
def run_pipeline(image: Image.Image, model_name: str,
                box_thresh: float, text_thresh: float,
                session_id: str, state: dict):
    """
    Flow: Detection FIRST, then Qwen describe only if defect found.

    model_name: "GroundingDINO Base" | "GroundingDINO LoRA" | "OWL-ViT Base" | "OWL-ViT LoRA"
    """
    tmp = "/tmp/pgmidd.jpg"
    image.save(tmp)

    # Step 1: Detect based on selected model
    if model_name == "OWL-ViT Base":
        try:
            model, processor = load_owlvit()
            bboxes, scores = run_owlvit(model, processor, tmp, box_thresh,
                                        query=GDINO_GENERIC_PROMPT)
        except RuntimeError:
            print("[OWL-ViT] Unavailable — falling back to GroundingDINO Base.")
            model = load_gdino()
            bboxes, scores = run_gdino(model, tmp, GDINO_GENERIC_PROMPT, box_thresh, text_thresh)
        defect_label = "defect"

    elif model_name == "OWL-ViT LoRA":
        try:
            model, processor = load_owlvit_lora()
            bboxes, scores = run_owlvit(model, processor, tmp, box_thresh,
                                        query=GDINO_GENERIC_PROMPT)
        except RuntimeError:
            print("[OWL-LoRA] Unavailable — falling back to GroundingDINO Base.")
            model = load_gdino()
            bboxes, scores = run_gdino(model, tmp, GDINO_GENERIC_PROMPT, box_thresh, text_thresh)
        defect_label = "defect"

    elif model_name == "GroundingDINO LoRA":
        model = load_gdino_lora()
        bboxes, scores = run_gdino(model, tmp, GDINO_GENERIC_PROMPT, box_thresh, text_thresh)
        defect_label = "defect"

    else:  # "GroundingDINO Base"
        model = load_gdino()
        bboxes, scores = run_gdino(model, tmp, GDINO_GENERIC_PROMPT, box_thresh, text_thresh)
        defect_label = "defect"

    n = len(bboxes)
    max_s = max(scores) if scores else 0.0

    # Step 2: Annotate or green banner 
    if n > 0:
        annotated = annotate(image, bboxes, scores, defect_label)
        # Qwen describe only on positive detection
        description = describe_defect(image)
        parsed = parse_description(description)
        # Override defect label if Qwen gives meaningful info
        if not parsed.get("skip_guard", False):
            defect_label = parsed.get("defect", "defect")
    else:
        # No defect — green banner
        img_rgb = np.array(image.convert("RGB"))
        h, w = img_rgb.shape[:2]
        banner_h = 60
        from PIL import Image as PILImage, ImageDraw as PILDraw
        pil_img = PILImage.fromarray(img_rgb)
        draw = PILDraw.Draw(pil_img)
        draw.rectangle([(0, h - banner_h), (w, h)], fill=(0, 200, 0))
        draw.text((20, h - banner_h + 14), "GOOD — No defect detected",
                  fill=(255, 255, 255))
        annotated = np.array(pil_img)
        description = (
            "Defect: none\n"
            "Location: none\n"
            "Description: No defect found — product is in good condition."
        )
        defect_label = "good"

    # Step 3: Save to session 
    if session_id == "" or session_id is None:
        session_id = str(uuid.uuid4())[:8]

    session = get_or_create_session(session_id)
    session.update({
        "image": image,
        "description": description,
        "bboxes": bboxes,
        "scores": scores,
        "defect": defect_label,
        "location": "",
        "category": "",
        "model": model_name,
    })

    summary = (
        f"**Session ID:** `{session_id}`\n"
        f"**Model:** `{model_name}`\n"
        f"**Detections:** {n} | Max conf: {max_s:.3f}\n\n"
        f"**Qwen description:**\n{description}"
    )

    return annotated, description, summary, session_id, state


# Chat Response
def chat_reply(session_id: str, user_msg: str, history: list):
    if session_id in _sessions and _sessions[session_id]["image"] is None:
        return history + [[user_msg, "Vui lòng chạy detection trước bằng cách upload ảnh và nhấn 'Run Pipeline'."]]

    answer = qwen_qa(session_id, user_msg)
    return history + [[user_msg, answer]]


# Export Session
def export_session(session_id: str) -> dict:
    if session_id not in _sessions:
        return {"error": "Session not found"}
    s = _sessions[session_id]
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "description": s["description"],
        "defect": s["defect"],
        "location": s["location"],
        "category": s["category"],
        "n_detections": len(s["bboxes"]),
        "bboxes": [b.tolist() if isinstance(b, (np.ndarray, torch.Tensor)) else b for b in s["bboxes"]],
        "scores": s["scores"],
        "messages": s["messages"],
    }


# Gradio UI
def build_demo():
    demo = gr.Blocks(
        title="PGMIDD — Industrial Defect Detection",
        theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate"),
        css="""
        .gradio-container { max-width: 1400px !important; }
        """,
    )

    with demo:
        # Header 
        gr.Markdown("""
        # PGMIDD — Industrial Defect Detection
        **Prompt-Guided Multimodal Industrial Defect Detection**
        """)

        session_id = gr.State("")

        # 3-column layout: Controls | Chat | Results
        with gr.Row(equal_height=False):
            # LEFT: Controls
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Controls")

                image_input = gr.Image(
                    label="Upload industrial product image",
                    type="pil", height=220,
                )

                model_select = gr.Radio(
                    choices=["GroundingDINO Base", "GroundingDINO LoRA", "OWL-ViT Base", "OWL-ViT LoRA"],
                    value="GroundingDINO Base",
                    label="Detection Model",
                )

                gr.Markdown("#### Thresholds")
                box_thresh = gr.Slider(
                    label="Box Threshold", minimum=0.05, maximum=0.95,
                    value=BOX_THRESHOLD, step=0.05,
                )
                text_thresh = gr.Slider(
                    label="Text Threshold", minimum=0.05, maximum=0.95,
                    value=TEXT_THRESHOLD, step=0.05,
                    info="Only used for GDINO models",
                )

                run_btn = gr.Button(" Run Pipeline", variant="primary", size="lg")

            # CENTER: Chat
            with gr.Column(scale=2):
                gr.Markdown("### Q&A")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=420,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about the defect...",
                        lines=1,
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Markdown("""
                *Upload an image + Run Pipeline first, then ask questions here.*
                """)

            # RIGHT: Results
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Detection Result")

                annotated_output = gr.Image(
                    label="Annotated (red = defect)",
                    type="numpy", height=260,
                )

                description_output = gr.Textbox(
                    label="Qwen VL Description",
                    lines=3,
                    show_label=True,
                )

                summary_output = gr.Textbox(
                    label="Summary",
                    lines=3,
                    show_label=True,
                )

                gr.Markdown("#### Session")
                session_display = gr.JSON(label=None, show_label=False)



        # Event handlers 
        run_btn.click(
            fn=run_pipeline,
            inputs=[image_input, model_select, box_thresh, text_thresh, session_id, gr.State({})],
            outputs=[annotated_output, description_output, summary_output, session_id, gr.State({})],
        )

        def on_send(msg, history, session_id):
            if not msg.strip():
                return "", history
            new_history = chat_reply(session_id, msg, history or [])
            return "", new_history

        send_btn.click(
            fn=on_send,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot],
        )
        msg.submit(
            fn=on_send,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot],
        )

        def on_export(session_id):
            return export_session(session_id)

        export_btn = gr.Button("Export", variant="secondary", size="sm")
        export_btn.click(
            fn=on_export,
            inputs=[session_id],
            outputs=[session_display],
        )

    return demo


if __name__ == "__main__":
    # Apply GroundingDINO patches BEFORE loading any model
    _apply_gdino_patches()

    # Lazy load all detectors — only pre-load Qwen (always needed for describe + Q&A)
    # This ensures max 2 models in VRAM at any time:
    #   Option 1: GDINO Base + Qwen (~9-12GB)
    #   Option 2: GDINO LoRA + Qwen (~10-15GB)  — LoRA is MERGED weights, not additive
    #   Option 3: OWL-ViT Base + Qwen (~10-15GB)
    #   Option 4: OWL-ViT LoRA + Qwen (~10-15GB) — LoRA MERGED, same footprint as Base
    print("[Pre-load] Downloading and loading models...")
    if not QWEN_DISABLED:
        load_qwen_vl()  # Qwen2.5-VL-7B fine-tuned — only model pre-loaded

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)