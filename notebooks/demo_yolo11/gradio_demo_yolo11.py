"""
PGMIDD Gradio Demo — YOLOv11 Detection + Qwen VL Description

Pipeline:
  Upload image -> YOLOv11 (detect potential defect regions, class-agnostic)
  -> Annotate -> Qwen describe + Q&A
  -> No defect -> Green banner "GOOD"

Detection: YOLOv11 (Ultralytics) — real-time object detection, pretrained on COCO.
We use it class-agnostically: detect all objects above threshold,
then let Qwen VL classify whether each is a defect.

NOTE: This demo is for visual quick-testing only.
      Evaluation metrics (AUROC, PR AUC, mAP@0.5) are computed separately
      on the full MVTec+VISA dataset.

Run locally:
  pip install ultralytics gradio torch torchvision
  python notebooks/demo_yolo11/gradio_demo_yolo11.py --port 7860 --share
"""

import os, re, warnings, argparse, uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr

# CONFIG
BOX_THRESHOLD  = 0.30
NMS_THRESHOLD  = 0.45
MAX_BOXES      = 10
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_DISABLED  = os.environ.get("QWEN_DISABLED", "0") == "1"
MODEL_VARIANT  = os.environ.get("YOLO_VARIANT", "yolo11m")
warnings.filterwarnings("ignore")

# MODEL CACHE
_yolo_model  = None
_qwen_proc   = None
_qwen_model  = None

# SESSION STORE
_sessions = {}

def get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "image": None,
            "description": "",
            "bboxes": [],
            "scores": [],
            "defect": "",
            "messages": [],
            "model": "",
        }
    return _sessions[session_id]


# YOLOv11 (Ultralytics)
def load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    from ultralytics import YOLO
    print(f"[YOLOv11] Loading {MODEL_VARIANT} on {DEVICE}...")
    _yolo_model = YOLO(MODEL_VARIANT)
    _yolo_model.to(DEVICE)
    print(f"[YOLOv11] {MODEL_VARIANT} ready on {DEVICE}.")
    return _yolo_model


def run_yolo(image_path: str, box_thresh: float = BOX_THRESHOLD) -> tuple:
    """
    YOLOv11 class-agnostic detection — detect all objects above threshold.

    YOLOv11 is pretrained on COCO. We treat all predictions as potential
    defect candidates. For precise detection, fine-tune on MVTec+VISA.

    Returns:
        bboxes: list of [x1, y1, x2, y2] in absolute pixels
        scores: list of confidence scores
    """
    model = load_yolo()

    results = model.predict(
        source=image_path,
        conf=box_thresh,
        iou=NMS_THRESHOLD,
        verbose=False,
        device=DEVICE,
    )

    if not results or len(results) == 0:
        return [], []

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], []

    # Class-agnostic: use ALL boxes regardless of class
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # (N, 4) absolute pixels
    boxes_conf = r.boxes.conf.cpu().numpy()     # (N,) confidence scores

    # Sort by confidence descending, cap at MAX_BOXES
    order = np.argsort(boxes_conf)[::-1]
    scores_np = boxes_conf[order]
    boxes_np  = boxes_xyxy[order]

    if len(scores_np) > MAX_BOXES:
        scores_np = scores_np[:MAX_BOXES]
        boxes_np  = boxes_np[:MAX_BOXES]

    # Clip to image bounds
    img = cv2.imread(image_path)
    if img is None:
        h, w = 480, 640
    else:
        h, w = img.shape[:2]

    xyxy = []
    for b in boxes_np:
        xyxy.append([
            float(max(0.0, min(w, b[0]))),
            float(max(0.0, min(h, b[1]))),
            float(max(0.0, min(w, b[2]))),
            float(max(0.0, min(h, b[3]))),
        ])

    return xyxy, scores_np.tolist()


# QWEN VL (description + Q&A)
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
    _qwen_proc  = AutoProcessor.from_pretrained(
        MODEL_NAME, token=hf_token, trust_remote_code=True)
    _qwen_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        token=hf_token, trust_remote_code=True,
    )
    _qwen_model.eval()
    print("[Qwen] Ready.")
    return _qwen_proc, _qwen_model


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
    return proc.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()


def qwen_qa(session_id: str, user_message: str) -> str:
    proc, model = load_qwen_vl()
    if model is None:
        return "Qwen VL is unavailable. Please run detection first."

    session = get_or_create_session(session_id)
    if session["description"] == "":
        return "Please run detection first before asking questions."

    n_dets = len(session["bboxes"])
    if n_dets == 0:
        context_block = (
            f"- Ảnh đã phân tích: sản phẩm không có defect rõ ràng.\n"
            f"- Mô tả: {session['description']}\n"
        )
    else:
        bbox_str = ", ".join(
            f"(x1={int(b[0])}, y1={int(b[1])}, x2={int(b[2])}, y2={int(b[3])}, conf={s:.2f})"
            for b, s in zip(session["bboxes"], session["scores"])
        )
        context_block = (
            f"- Lỗi đã phát hiện: {session['defect']}\n"
            f"- Số lượng bbox: {n_dets}\n"
            f"- Chi tiết bbox: {bbox_str}\n"
            f"- Mô tả: {session['description']}\n"
        )

    history = session["messages"][-6:]
    history_block = ""
    if history:
        lines = [
            f"- {'Inspector' if m['role'] == 'assistant' else 'User'}: {m['content']}"
            for m in history
        ]
        history_block = "\nLịch sử hội thoại:\n" + "\n".join(lines) + "\n"

    system_prompt = (
        "Bạn là một inspector AI chuyên về kiểm tra lỗi công nghiệp. "
        "Dựa trên thông tin phân tích ảnh dưới đây, trả lời câu hỏi của user "
        "một cách chi tiết và hữu ích.\n\n"
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

    session["messages"].append({"role": "user", "content": user_message})
    session["messages"].append({"role": "assistant", "content": answer})
    return answer


# PARSE
def parse_description(text: str) -> dict:
    defect   = re.search(r"Defect:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    location = re.search(r"Location:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    raw_defect = defect.group(1).strip() if defect else ""
    skip_guard = raw_defect.lower() in {"", "none", "good", "normal", "null", "n/a", "na"}
    return {
        "defect": raw_defect if raw_defect else "defect",
        "location": location.group(1).strip() if location else "",
        "skip_guard": skip_guard,
    }


# ANNOTATION
def annotate(img, bboxes, scores, label="defect"):
    if isinstance(img, Image.Image):
        img = np.array(img)[:, :, ::-1].copy()
    h, w = img.shape[:2]
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = [int(max(0, min(v, d))) for v, d in zip(bbox, [w, h, w, h])]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw, y1), (0, 0, 255), -1)
        cv2.putText(img, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# MAIN PIPELINE
def run_pipeline(image: Image.Image, box_thresh: float,
                 session_id: str, state: dict):
    tmp = "/tmp/pgmidd_yolo.jpg"
    image.save(tmp)

    # Step 1: YOLOv11 Detection
    bboxes, scores = run_yolo(tmp, box_thresh)
    n = len(bboxes)
    max_s = max(scores) if scores else 0.0

    # Step 2: Annotate or green banner
    if n > 0:
        annotated  = annotate(image, bboxes, scores, "defect")
        description = describe_defect(image)
        parsed = parse_description(description)
        defect_label = parsed.get("defect", "defect") if not parsed.get("skip_guard") else "defect"
    else:
        img_rgb = np.array(image.convert("RGB"))
        h, w = img_rgb.shape[:2]
        from PIL import Image as PILImage, ImageDraw as PILDraw
        pil_img = PILImage.fromarray(img_rgb)
        draw = PILDraw.Draw(pil_img)
        banner_h = 60
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

    # Step 3: Save session
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    session = get_or_create_session(session_id)
    session.update({
        "image": image,
        "description": description,
        "bboxes": bboxes,
        "scores": scores,
        "defect": defect_label,
        "messages": [],
        "model": f"YOLOv11 ({MODEL_VARIANT})",
    })

    summary = (
        f"**Session ID:** `{session_id}`\n"
        f"**Model:** `YOLOv11 ({MODEL_VARIANT})`\n"
        f"**Detections:** {n} | Max conf: {max_s:.3f}\n\n"
        f"**Qwen description:**\n{description}"
    )
    return annotated, description, summary, session_id, state


# CHAT
def chat_reply(session_id: str, user_msg: str, history: list):
    if session_id in _sessions and _sessions[session_id]["image"] is None:
        return history + [[user_msg,
            "Vui lòng chạy detection trước bằng cách upload ảnh và nhấn 'Run Pipeline'."]]
    answer = qwen_qa(session_id, user_msg)
    return history + [[user_msg, answer]]


# EXPORT
def export_session(session_id: str) -> dict:
    if session_id not in _sessions:
        return {"error": "Session not found"}
    s = _sessions[session_id]
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "description": s["description"],
        "defect": s["defect"],
        "n_detections": len(s["bboxes"]),
        "bboxes": s["bboxes"],
        "scores": s["scores"],
        "messages": s["messages"],
    }


# GRADIO UI
def build_demo():
    demo = gr.Blocks(
        title="PGMIDD — YOLOv11 Industrial Defect Detection",
        theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate"),
        css=".gradio-container { max-width: 1400px !important; }",
    )

    with demo:
        gr.Markdown(f"""
        # PGMIDD — Industrial Defect Detection
        **YOLOv11 ({MODEL_VARIANT}) + Qwen2.5-VL-7B (fine-tuned)**

        Upload an industrial image -> YOLOv11 detects potential defect regions ->
        Qwen VL describes defect -> Ask follow-up questions.

        *YOLOv11 is class-agnostic (pretrained on COCO). For full evaluation metrics,
         run the benchmark notebook on MVTec+VISA dataset.*
        """)

        session_id = gr.State("")

        with gr.Row(equal_height=False):
            # LEFT: Controls
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Controls")
                image_input = gr.Image(
                    label="Upload industrial product image",
                    type="pil", height=220,
                )
                gr.Markdown("#### Thresholds")
                box_thresh = gr.Slider(
                    label="Box Threshold", minimum=0.05, maximum=0.95,
                    value=BOX_THRESHOLD, step=0.05,
                )
                run_btn = gr.Button(" Run Pipeline", variant="primary", size="lg")

            # CENTER: Chat
            with gr.Column(scale=2):
                gr.Markdown("### Q&A")
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about the defect...",
                        lines=1, scale=4, container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                gr.Markdown(
                    "*Upload an image + Run Pipeline first, then ask questions here.*")

            # RIGHT: Results
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Detection Result")
                annotated_output = gr.Image(
                    label="Annotated (red = defect)",
                    type="numpy", height=260,
                )
                description_output = gr.Textbox(
                    label="Qwen VL Description",
                    lines=3, show_label=True,
                )
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=4, show_label=True,
                )
                gr.Markdown("#### Session")
                session_display = gr.JSON(label=None, show_label=False)

        # Event handlers
        run_btn.click(
            fn=run_pipeline,
            inputs=[image_input, box_thresh, session_id, gr.State({})],
            outputs=[annotated_output, description_output, summary_output,
                     session_id, gr.State({})],
        )

        def on_send(msg, history, session_id):
            if not msg.strip():
                return "", history
            return "", chat_reply(session_id, msg, history or [])

        send_btn.click(fn=on_send,
                       inputs=[msg, chatbot, session_id],
                       outputs=[msg, chatbot])
        msg.submit(fn=on_send,
                   inputs=[msg, chatbot, session_id],
                   outputs=[msg, chatbot])

        export_btn = gr.Button("Export Session", variant="secondary", size="sm")
        export_btn.click(fn=export_session,
                         inputs=[session_id],
                         outputs=[session_display])

    return demo


# ENTRYPOINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--variant",
        default=None,
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="YOLOv11 variant: n=nano, s=small, m=medium, l=large, x=xl",
    )
    args = parser.parse_args()

    # Override MODEL_VARIANT before loading so load_yolo() picks it up
    import sys
    if args.variant:
        sys.modules[__name__].MODEL_VARIANT = args.variant

    print(f"[YOLOv11] Pre-loading {MODEL_VARIANT}...")
    load_yolo()

    if not QWEN_DISABLED:
        print("[Pre-load] Loading Qwen2.5-VL-7B fine-tuned...")
        load_qwen_vl()

    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
