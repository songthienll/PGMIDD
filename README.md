# PGMIDD

**Prompt-Guided Multimodal Industrial Defect Detection**

Industrial product defect detection using:
- **GroundingDINO** — pretrained object detection with text prompts
- **Qwen2.5-VL-7B** — multimodal VLM for defect description and Q&A

## Demo

```bash
python notebooks/gradio_demo.py --port 7860 --share
```

Upload an industrial image → GroundingDINO detects defects → Qwen VL describes and answers questions.

## Models

| Model | AUROC Image | AUROC Pixel | PR AUC |
|-------|-------------|-------------|--------|
| GroundingDINO Base | **0.8421** | **0.7496** | **0.6800** |

Evaluated on MVTec+VISA dataset (278 defects / 2000 good samples).

## Tech Stack

- **Detection:** GroundingDINO Swin-OGC, OWL-ViT v2
- **VLM:** Qwen2.5-VL-7B (fine-tuned on MVTec+VISA)
- **UI:** Gradio
- **Training:** LoRA (Peft), PyTorch