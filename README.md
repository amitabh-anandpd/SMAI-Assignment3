---
title: Indian License Plate Recognition
emoji: 🚘
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.44.1
python_version: "3.10"
app_file: hf_push/app.py
pinned: false
license: mit
---

# Indian License Plate Recognition

An end-to-end **Automatic License Plate Recognition (ALPR)** system for Indian vehicles.

![Indian License Plate Recognition System](Indian%20License%20Plate%20Recognition%20System.png)

## What it does
The system takes an image or video frame, detects the license plate with **YOLOv8**, crops the plate region, and extracts text with **PaddleOCR**. A post-processing layer then corrects common OCR mistakes and validates the plate format.

## Pipeline
1. **Input**: image, batch of images, or video
2. **Detection**: YOLOv8 finds the plate bounding box
3. **Preprocessing**: crop, resize, and enhance the plate region
4. **OCR**: PaddleOCR reads the plate text
5. **Post-processing**: format correction and semantic validation
6. **Output**: annotated image, extracted plate text, and status

## Features
- Single-image mode with annotated output
- Batch mode with CSV export
- Video mode for frame-by-frame plate detection
- Indian plate format correction and flagging for invalid results

## Tech Stack
- `YOLOv8n` (Ultralytics)
- `PaddleOCR 2.9.1`
- `Streamlit`

## Repository Structure
- `hf_push/app.py` - Streamlit app for inference
- `yolo_detector.py` - YOLO detection wrapper
- `ocr_processor.py` - OCR and plate-format cleanup
- `utils.py` - helper functions
- `Code/Train code/train.ipynb` - training notebook
- `Code/Inference Code/Image2OCR2.ipynb` - inference notebook
- `Acuuracy testing/compare_plates.py` - accuracy comparison script

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained YOLO weights at:

```text
hf_push/best.pt
```

## Run the app

```bash
streamlit run hf_push/app.py
```

## Webapp link

```bash
https://huggingface.co/spaces/AnshuBhadiyadra/indian-license-plate-ocr
```

## Training flow
The training notebook downloads or loads the dataset, trains `yolov8n`, validates the model, and saves the best checkpoint as `best.pt`.

## Notes
- The model is designed for Indian plate detection, not generic OCR.
- For deployment, keep the same `best.pt` filename expected by `hf_push/app.py`.
- The comparison script expects prediction CSV files in the `Acuuracy testing` folder.
