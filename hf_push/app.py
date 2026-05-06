import os
import re
import cv2
import tempfile
import numpy as np
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ── Config and Constants ───────────────────────────────────────────────────────
st.set_page_config(page_title="🚘 Indian License Plate OCR", page_icon="🚘", layout="wide", initial_sidebar_state="collapsed")

TO_NUMBER = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}

OCR_VISUAL_GROUPS = [
    {"O", "0", "D", "Q", "U", "C", "G"},
    {"1", "I", "l", "L", "T", "J", "7"},
    {"8", "B", "S", "3"},
    {"5", "S"},
    {"2", "Z", "7"},
    {"A", "4", "H", "R"},
    {"E", "F", "P", "B"},
    {"M", "W", "N", "V"},
    {"K", "X", "Y"},
    {"6", "G", "b", "C"},
    {"9", "P", "g", "q"}
]

INDIAN_RTO_MAP_RAW = {
    "AN": ["01", "02", "03"],
    "AP": ["01", "02", "03", "04", "05", "07", "09", "10", "11", "12", "13", "15", "16", "18", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "35", "36", "37", "39", "40"],
    "AR": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "19", "20", "22"],
    "AS": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "29", "30", "31", "32", "33", "34"],
    "BR": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "19", "21", "22", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "37", "38", "39", "43", "44", "45", "46", "50", "51", "52", "53", "55", "56", "57"],
    "CH": ["01", "02", "03", "04"],
    "CG": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"],
    "DD": ["01", "02", "03"],
    "DL": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"],
    "DN": ["01", "02"],
    "GA": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "GJ": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39"],
    "HR": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99"],
    "HP": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99"],
    "JH": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"],
    "JK": ["01", "02", "03", "04", "05", "06", "08", "09", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"],
    "KA": ["01", "02", "03", "04", "05", "06", "07", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76"],
    "KL": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86"],
    "LA": ["01", "02"],
    "LD": ["01"],
    "MH": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55"],
    "ML": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    "MN": ["01", "02", "03", "04", "05", "06", "07", "08"],
    "MP": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "65", "66", "67", "68", "69", "70"],
    "MZ": ["01", "02", "03", "04", "05", "06", "07", "08"],
    "NL": ["01", "02", "03", "04", "05", "06", "07", "08"],
    "OD": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35"],
    "OR": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"],
    "PB": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91"],
    "PY": ["01", "02", "03", "04", "05"],
    "RJ": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "45", "46", "47", "48", "49", "50", "51", "52"],
    "SK": ["01", "02", "03", "04", "05", "06", "07", "08"],
    "TG": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36"],
    "TN": ["01", "02", "03", "04", "05", "06", "07", "09", "10", "11", "12", "13", "14", "15", "16", "18", "19", "20", "21", "22", "23", "24", "25", "28", "30", "31", "32", "33", "34", "36", "37", "38", "39", "40", "41", "42", "43", "45", "46", "47", "48", "49", "50", "51", "52", "54", "55", "56", "57", "58", "59", "60", "61", "63", "64", "65", "66", "67", "68", "69", "70", "72", "73", "74", "75", "76", "77", "78", "79", "81", "82", "84", "85", "86", "87", "88", "90", "91", "92", "93", "94", "95", "96", "97", "99"],
    "TR": ["01", "02", "03", "04", "05", "06", "07", "08"],
    "TS": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36"],
    "UA": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "UK": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
    "UP": ["11", "12", "13", "14", "15", "16", "17", "19", "20", "21", "22", "23", "24", "25", "26", "27", "30", "31", "32", "33", "34", "35", "36", "37", "38", "40", "41", "42", "43", "44", "45", "46", "47", "50", "51", "52", "53", "54", "55", "56", "57", "58", "60", "61", "62", "63", "64", "65", "66", "67", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "90", "91", "92", "93", "94", "95", "96", "97"],
    "WB": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99"]
}

INDIAN_RTO_MAP = {state: set(codes) for state, codes in INDIAN_RTO_MAP_RAW.items()}
STATE_CODES = sorted(code for code in INDIAN_RTO_MAP.keys())

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

    h1 {
        background: linear-gradient(90deg, #6ee7f7, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 700; font-size: 2.4rem; margin-bottom: 0.2rem;
    }
    .subtitle { color: #8892a4; font-size: 1rem; margin-bottom: 2rem; }
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.6rem 0;
    }
    .plate-text {
        font-size: 2rem; font-weight: 700; letter-spacing: 0.15em;
        color: #6ee7f7; text-align: center; padding: 0.5rem;
    }
    .status-ok   { color: #4ade80; font-size: 1rem; }
    .status-warn { color: #facc15; font-size: 1rem; }
    .status-fail { color: #f87171; font-size: 1rem; }

    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #6ee7f7, #a78bfa);
        color: #0f1117; font-weight: 700; border: none;
        border-radius: 8px; padding: 0.55rem 2rem;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🚘 Automatic License Plate Recognition</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLOv8 + PaddleOCR + Semantic RTO Correction</p>', unsafe_allow_html=True)

# ── Model loading (cached) ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

@st.cache_resource(show_spinner="Loading models… first run takes ~60 s")
def load_models():
    yolo = YOLO(MODEL_PATH)
    # PaddleOCR 3.x API for CPU-only
    ocr  = PaddleOCR(use_angle_cls=False, lang="en", device="cpu")
    return yolo, ocr

yolo_model, ocr = load_models()

# ── NLP Semantic Helpers ──────────────────────────────────────────────────────
def _visual_substitution_cost(char_a, char_b):
    if char_a == char_b:
        return 0
    for group in OCR_VISUAL_GROUPS:
        if char_a in group and char_b in group:
            return 0.5
    return 1

def _normalize_text(raw_text):
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper())

def _format_standard_plate(clean_text):
    fixed_text = ""
    length = len(clean_text)

    for i, char in enumerate(clean_text):
        if i < 2:
            fixed_text += TO_LETTER.get(char, char) if char.isdigit() else char
        elif i < 4:
            fixed_text += TO_NUMBER.get(char, char) if char.isalpha() else char
        elif length >= 8 and i >= length - 4:
            fixed_text += TO_NUMBER.get(char, char) if char.isalpha() else char
        else:
            fixed_text += TO_LETTER.get(char, char) if char.isdigit() else char
    return fixed_text

def _levenshtein_distance(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = _visual_substitution_cost(ca, cb)
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]

def _closest_code(target, candidates, max_distance=1):
    best = None
    best_distance = max_distance + 1
    best_count = 0

    for code in candidates:
        distance = _levenshtein_distance(target, code)
        if distance < best_distance:
            best = code
            best_distance = distance
            best_count = 1
        elif distance == best_distance:
            best_count += 1

    if best_distance <= max_distance and best_count == 1:
        return best
    return None

def _apply_semantic_correction(plate_text):
    if len(plate_text) < 4:
        return plate_text, False

    state_code = plate_text[:2]
    rto_code = plate_text[2:4]

    if state_code not in INDIAN_RTO_MAP:
        corrected_state = _closest_code(state_code, STATE_CODES)
        if corrected_state is None:
            return plate_text, True
        state_code = corrected_state

    rto_codes = INDIAN_RTO_MAP.get(state_code, set())
    if rto_code not in rto_codes:
        corrected_rto = _closest_code(rto_code, rto_codes)
        if corrected_rto is None:
            return plate_text, True
        rto_code = corrected_rto

    corrected_text = plate_text
    if state_code != plate_text[:2] or rto_code != plate_text[2:4]:
        corrected_text = state_code + rto_code + plate_text[4:]

    return corrected_text, False

def _normalize_and_validate_plate(raw_text):
    clean_text = _normalize_text(raw_text)
    if not clean_text:
        return "", False

    fixed_text = _format_standard_plate(clean_text)
    return _apply_semantic_correction(fixed_text)

def enforce_strict_plate_format(raw_text):
    fixed_text, _ = _normalize_and_validate_plate(raw_text)
    return fixed_text

def evaluate_plate(plate_text):
    clean_text = plate_text.replace(" ", "").replace("|", "")
    if clean_text in ("UNREADABLE", "NOPLATEDETECTED"):
        return "❌ Failed to read.", "fail"

    for candidate in [part.strip() for part in plate_text.split("|") if part.strip()]:
        _, flagged = _normalize_and_validate_plate(candidate)
        if flagged:
            return "⚠️ FLAGGED: Invalid State/RTO semantics.", "warn"

    if len(clean_text) > 12:
        return "⚠️ FLAGGED FOR MANUAL CHECK: Detected > 12 chars.", "warn"

    return "✅ Processed (Note: OCR predictions may still not be 100% perfect).", "ok"

# ── Inference Logic ───────────────────────────────────────────────────────────
def process_image(img_rgb: np.ndarray):
    results = yolo_model(img_rgb, verbose=False)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    plate_found = False
    extracted_plates = []

    for r in results:
        for box in r.boxes:
            plate_found = True
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped_plate = img_rgb[y1:y2, x1:x2]

            # --- ADVANCED PREPROCESSING FOR OCR ---
            # 1. Convert to Grayscale
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_RGB2GRAY)
            # 2. Resize (Upscale by 2x using Cubic interpolation for smoother edges)
            resized_plate = cv2.resize(gray_plate, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            # 3. Apply CLAHE (Locally enhances contrast to fight shadows and glare)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_plate = clahe.apply(resized_plate)
            # 4. Slight Gaussian Blur (Removes tiny speckles of dirt or noise)
            final_processed_plate = cv2.GaussianBlur(contrast_plate, (3, 3), 0)
            
            # PaddleOCR v3 expects a 3-channel image
            final_processed_plate = cv2.cvtColor(final_processed_plate, cv2.COLOR_GRAY2RGB)

            # --- PaddleOCR Inference ---
            paddle_results = ocr.ocr(final_processed_plate, cls=False)
            raw_text = ""
            if paddle_results and paddle_results[0] is not None:
                for line in paddle_results[0]:
                    raw_text += line[1][0]

            if raw_text.strip():
                final_plate_text = enforce_strict_plate_format(raw_text)
                extracted_plates.append(final_plate_text)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, final_plate_text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_bgr, "UNREADABLE", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if len(extracted_plates) > 0:
        final_output_text = " | ".join(extracted_plates)
    elif plate_found:
        final_output_text = "UNREADABLE"
    else:
        final_output_text = "NO PLATE DETECTED"
        cv2.putText(img_bgr, "NO PLATE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, final_output_text

# ── Video Processing ────────────────────────────────────────────────────────────
class VideoProcessor:
    @staticmethod
    def process_video(video_path, sample_rate_fps=2, progress_bar=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error opening video"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30
        
        frame_interval = int(max(1, round(fps / sample_rate_fps)))
        
        detected_plates = []
        best_annotated_frame = None
        first_frame_annotated = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_img, plate_text = process_image(img_rgb)
                
                if first_frame_annotated is None:
                    first_frame_annotated = annotated_img
                
                if plate_text and plate_text not in ["UNREADABLE", "NO PLATE DETECTED"]:
                    detected_plates.append(plate_text)
                    best_annotated_frame = annotated_img
            
            frame_count += 1
            if progress_bar and total_frames > 0:
                progress = min(1.0, frame_count / total_frames)
                progress_bar.progress(progress, text=f"Processing video frames... {int(progress*100)}%")

        cap.release()
        
        final_frame = best_annotated_frame if best_annotated_frame is not None else first_frame_annotated
        
        if not detected_plates:
            return final_frame, "NO PLATE DETECTED"
            
        # Majority voting
        counter = Counter(detected_plates)
        most_common_plate, _ = counter.most_common(1)[0]
        
        return final_frame, most_common_plate

# ── UI Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷  Single Image", "📂  Batch / Folder", "🎥  Video"])

# ── TAB 1: Single Image ───────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Upload a car image")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_upload",
            label_visibility="collapsed",
        )
        analyze_btn = st.button("🔍  Analyze Plate", key="analyze_btn", use_container_width=True)

        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Input image", use_container_width=True)

    with col_right:
        if analyze_btn and uploaded:
            with st.spinner("Running detection & advanced OCR preprocessing…"):
                img_rgb = np.array(img_pil)
                annotated, plate_text = process_image(img_rgb)
                status_msg, status_cls = evaluate_plate(plate_text)

            st.markdown("#### Result")
            st.image(annotated, caption="Annotated result", use_container_width=True)
            st.markdown(f'<div class="result-card"><p class="plate-text">{plate_text}</p></div>', unsafe_allow_html=True)
            css_cls = {"ok": "status-ok", "warn": "status-warn", "fail": "status-fail"}[status_cls]
            st.markdown(f'<p class="{css_cls}">{status_msg}</p>', unsafe_allow_html=True)

        elif analyze_btn and not uploaded:
            st.warning("Please upload an image first.")

# ── TAB 2: Batch ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Upload multiple car images")
    batch_files = st.file_uploader(
        "Select files",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed",
    )
    process_btn = st.button("⚡  Process Batch", key="batch_btn", use_container_width=True)

    if process_btn and batch_files:
        rows = []
        progress = st.progress(0, text="Processing…")
        for i, f in enumerate(batch_files):
            img_rgb = np.array(Image.open(f).convert("RGB"))
            _, plate_text = process_image(img_rgb)
            status_msg, _ = evaluate_plate(plate_text)
            rows.append({"Image Name": f.name, "Extracted Plate": plate_text, "Status": status_msg})
            progress.progress((i + 1) / len(batch_files), text=f"Processing {i+1}/{len(batch_files)}…")

        progress.empty()
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="⬇️  Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name="batch_results.csv",
            mime="text/csv",
        )

    elif process_btn and not batch_files:
        st.warning("Please upload at least one image.")

# ── TAB 3: Video ──────────────────────────────────────────────────────────────
with tab3:
    col_v_left, col_v_right = st.columns([1, 1], gap="large")

    with col_v_left:
        st.markdown("#### Upload a dashcam/CCTV video")
        uploaded_video = st.file_uploader(
            "Drag & drop or click to browse",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload",
            label_visibility="collapsed",
        )
        analyze_vid_btn = st.button("🎥  Analyze Video", key="analyze_vid_btn", use_container_width=True)

        if uploaded_video:
            st.video(uploaded_video)

    with col_v_right:
        if analyze_vid_btn and uploaded_video:
            with st.spinner("Preparing video..."):
                # Save uploaded video to a temp file for OpenCV
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.flush()
                tfile.close()

            progress_bar = st.progress(0, text="Initializing...")
            
            # Process the video
            annotated_frame, best_plate = VideoProcessor.process_video(tfile.name, sample_rate_fps=2, progress_bar=progress_bar)
            
            # Cleanup temp file
            os.remove(tfile.name)
            progress_bar.empty()

            st.markdown("#### Best Detection")
            if annotated_frame is not None:
                st.image(annotated_frame, caption="Frame with best detection", use_container_width=True)
            
            st.markdown(f'<div class="result-card"><p class="plate-text">{best_plate}</p></div>', unsafe_allow_html=True)
            
            status_msg, status_cls = evaluate_plate(best_plate)
            css_cls = {"ok": "status-ok", "warn": "status-warn", "fail": "status-fail"}[status_cls]
            st.markdown(f'<p class="{css_cls}">{status_msg} (Majority Vote)</p>', unsafe_allow_html=True)

        elif analyze_vid_btn and not uploaded_video:
            st.warning("Please upload a video first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="color:#8892a4;text-align:center;font-size:0.85rem;">SMAI Assignment 3 · YOLOv8 + PaddleOCR · Indian License Plate Recognition</p>', unsafe_allow_html=True)
