---
title: Indian License Plate Recognition
emoji: 🚘
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🚘 Indian License Plate Recognition

An end-to-end **Automatic License Plate Recognition (ALPR)** system for Indian vehicles.

## Pipeline
1. **YOLOv8** detects the license plate bounding box
2. **PaddleOCR** reads the text from the cropped plate region
3. A **format-correction layer** enforces the standard Indian plate format (`AA-00-AA-0000`)

## Features
- 📷 **Single image** mode — annotated result + plate text + validation status
- 📂 **Batch mode** — process multiple images, download results as CSV
- ⚠️ Auto-flags plates with > 12 characters for manual review

## Tech Stack
- `YOLOv8n` (Ultralytics) · `PaddleOCR 2.9.1` · `Streamlit`
