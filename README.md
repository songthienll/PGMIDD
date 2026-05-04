# PGMIDD

**Prompt-Guided Multimodal Industrial Defect Detection**

Industrial product defect detection using:
- **RT-DETR / YOLOv11 / YOLOv26** — real-time object detection (fine-tuned on MVTec-AD + VisA)
- **Qwen2.5-VL-7B** — multimodal VLM for defect description and Q&A (fine-tuned via QLoRA)

## Demo

RT-DETR (recommended):
```bash
python notebooks/demo/gradio_demo.py --port 7860 --share
```

YOLOv11:
```bash
python notebooks/demo_yolo11/gradio_demo_yolo11.py --port 7860 --share
```

Pipeline: upload an industrial image → detector finds defect regions → annotated bbox passed as visual prompt to Qwen VL → description + Q&A.

## Models

| Model | AUROC Image | AUROC Pixel | PR AUC | Macro PR AUC |
|-------|-------------|-------------|--------|--------------|
| **RT-DETR-L** | **0.9896** | 0.9426 | **0.9545** | 0.9511 |
| YOLOv11n | 0.9839 | **0.9594** | 0.7670 | **0.9553** |
| YOLOv26s | 0.8293 | 0.8313 | 0.8005 | 0.6936 |
| YOLOv26n | 0.8470 | 0.8435 | 0.8508 | 0.9219 |

Evaluated on MVTec-AD + VisA (combined defect/good split).

## Dataset Conversion

Convert MVTec-AD + VisA → YOLO format (extracts bboxes from masks, splits 80/10/10):

```bash
# 1. Verify bbox extraction quality first
python src/verify_mask_bbox_overlay.py \
    --mvtec /path/to/MVTec_AD \
    --visa  /path/to/VisA \
    --output data/verify_overlays \
    --count 20

# 2. Convert to YOLO format
python src/yolo_dataset_converter.py \
    --mvtec /path/to/mvtec_ad \
    --visa  /path/to/VisA \
    --output data/yolo
```

## Tech Stack

- **Detection:** RT-DETR-L, YOLOv11n, YOLOv26n/s (Ultralytics)
- **VLM:** Qwen2.5-VL-7B (QLoRA fine-tuned on MVTec-AD + VisA)
- **UI:** Gradio
- **Training:** PEFT (LoRA / QLoRA), bitsandbytes, PyTorch

## Project Structure

```
notebooks/
├── defect_detection_pipeline.ipynb              # Qwen evaluation (BLEU/ROUGE/METEOR)
├── finetune-qwen2-5vl-mvtec-visa-qlora.ipynb    # Qwen QLoRA fine-tune
├── demo/                                        # RT-DETR Gradio demo (AIP491_demo.ipynb)
├── demo_yolo11/                                 # YOLOv11 Gradio demo
└── rtdetr_fine_tune/                            # RT-DETR + YOLOv26 fine-tune
src/
├── data_processing.py
├── evaluation.py
├── qwen_vl_inference.py
├── text_visualization.py
├── visual_prompting.py
├── verify_mask_bbox_overlay.py                  # Bbox quality verifier
└── yolo_dataset_converter.py                    # MVTec+VisA → YOLO converter
```
