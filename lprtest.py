"""
 License Plate Recognition (LPR) & State Identification System (SIS)
GUI built with PySide6 | Detection via OpenCV | OCR via EasyOCR
"""

import cv2
import numpy as np
import easyocr
import re
import threading
import os
import warnings
import sys

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*MPS.*",
    category=UserWarning,
)

# =============================================
# GLOBAL COLOR SCHEME
# =============================================
COLORS = {
    "background": "#f9f8f3",      # Main background
    "foreground": "#6d325c",     # Top bar, canvas
    "accent_blue": "#6d325c",
    "accent_green": "#a6e3a1",
    "accent_pink": "#f38ba8",
    "accent_yellow": "#fab387",
    "accent_purple": "#cba6f7",
    "text_primary": "#a6e3a1",
    "text_secondary": "#000000",
    "text_muted": "#000000",
    "text_highlight": "#f9e2af",
    "error": "#f38ba8",
}

# =============================================
# MALAYSIAN STATE MAPPING
# =============================================
STATE_MAP = {
    # State/Territory prefixes
    "A": "Perak",
    "B": "Selangor",
    "C": "Pahang",
    "D": "Kelantan",
    "F": "Putrajaya",
    "J": "Johor",
    "K": "Kedah",
    "KV": "Langkawi",
    "L": "Labuan",
    "M": "Malacca",
    "N": "Negeri Sembilan",
    "P": "Penang",
    "R": "Perlis",
    "T": "Terengganu",
    "V": "Kuala Lumpur",
    "W": "Kuala Lumpur",

    # Sabah division prefixes
    "SA": "Sabah - West Coast",
    "SY": "Sabah - West Coast",
    "SJ": "Sabah - West Coast / Kota Kinabalu",
    "SG": "Sabah Government",
    "SMJ": "Sabah Government",
    "SS": "Sabah - Sandakan",
    "SM": "Sabah - Sandakan",
    "SB": "Sabah - Beaufort",
    "SK": "Sabah - Kudat",
    "ST": "Sabah - Tawau",
    "SW": "Sabah - Tawau",
    "SD": "Sabah - Lahad Datu",
    "SP": "Sabah - Lahad Datu",
    "SL": "Sabah - Labuan",
    "SU": "Sabah - Keningau",
    "S": "Sabah",

    # Sarawak division prefixes
    "QK": "Sarawak - Kuching",
    "QA": "Sarawak - Kuching",
    "QL": "Sarawak - Limbang",
    "QB": "Sarawak - Sri Aman and Betong",
    "QM": "Sarawak - Miri",
    "QS": "Sarawak - Sibu and Mukah",
    "QC": "Sarawak - Samarahan and Serian",
    "QP": "Sarawak - Kapit",
    "QT": "Sarawak - Bintulu",
    "QD": "Sarawak - Bintulu",
    "QR": "Sarawak - Sarikei",
    "Q": "Sarawak",

    # Taxi prefixes
    "HA": "Taxi - Perak",
    "HB": "Taxi - Selangor",
    "HC": "Taxi - Pahang",
    "HD": "Taxi - Kelantan",
    "HJ": "Taxi - Johor",
    "HK": "Taxi - Kedah",
    "HL": "Taxi - Labuan",
    "HM": "Taxi - Malacca",
    "HN": "Taxi - Negeri Sembilan",
    "HP": "Taxi - Penang",
    "HQ": "Taxi - Sarawak",
    "HR": "Taxi - Perlis",
    "HS": "Taxi - Sabah",
    "HT": "Taxi - Terengganu",
    "HW": "Taxi - Kuala Lumpur",
    "LIMO": "KLIA Limousine",

    # Military prefixes
    "ZA": "Military - Malaysian Army",
    "ZB": "Military - Malaysian Army",
    "ZC": "Military - Malaysian Army",
    "ZD": "Military - Malaysian Army",
    "ZL": "Military - Royal Malaysian Navy",
    "ZU": "Military - Royal Malaysian Air Force",
    "ZZ": "Military - Ministry of Defence",
    "TZ": "Military - Trailer",
    "Z": "Military - Malaysian Army",

    # Foreign missions
    "CD": "Diplomatic",
    "DC": "Diplomatic Corps",
    "CC": "Consular Corps",
    "UN": "United Nations",
    "PA": "International Organisation",

    # Trade plates
    "BA": "Trade Plate - Selangor",
    "WTP": "Trade Plate - Kuala Lumpur",

    # Trailer plates
    "TQ": "Trailer - Sarawak",
    "TS": "Trailer - Sabah",
    "TBD": "Trailer - Peninsular Malaysia",
}

TRADE_SUFFIX_MAP = {
    "J": "Trade Plate - Sabah",
    "Q": "Trade Plate - Sarawak",
}

VEHICLE_TYPES = ["Car", "Motorcycle", "Bus", "Truck", "Van"]
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
DIGIT_CONFUSIONS = str.maketrans({
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "B": "8",
    "G": "6",
})
DIGIT_CONFUSION_CHARS = set("ILOQSBG")
NOISY_LEADING_PREFIXES = {"ZA", "ZB", "ZC", "ZD", "ZL", "ZU", "ZZ", "TZ"}


def normalize_plate_text(text):
    """Keep only characters that can belong to a Malaysian plate."""
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not clean:
        return ""

    prefixes = sorted(STATE_MAP, key=len, reverse=True)

    def normalize_known_prefix(prefix, remainder):
        if not remainder:
            return prefix

        first_digit = next((i for i, ch in enumerate(remainder) if ch.isdigit()), None)
        if first_digit is None:
            return prefix + remainder

        letters = remainder[:first_digit]
        digits = remainder[first_digit:].translate(DIGIT_CONFUSIONS)

        # Military plates such as ZD1-2447 are often OCR'd as ZDI-2447.
        if len(prefix) >= 2 and len(letters) == 1 and letters in DIGIT_CONFUSION_CHARS:
            return prefix + letters.translate(DIGIT_CONFUSIONS) + digits

        return prefix + letters + digits

    for prefix in prefixes:
        if clean.startswith(prefix) and len(clean) > len(prefix):
            return normalize_known_prefix(prefix, clean[len(prefix):])

        # Recover leading OCR noise like IZD12447 without dropping legitimate
        # front letters from plates such as VCF2025 or TCB5944.
        prefix_at = clean.find(prefix)
        if (
            len(prefix) >= 2
            and 0 < prefix_at <= 2
            and (
                set(clean[:prefix_at]).issubset({"I", "1", "L"})
                or prefix in NOISY_LEADING_PREFIXES
            )
            and len(clean) > prefix_at + len(prefix)
        ):
            return normalize_known_prefix(prefix, clean[prefix_at + len(prefix):])

    match = re.match(r"^([A-Z]{1,3})([A-Z0-9]+)$", clean)
    if match and any(ch.isdigit() for ch in match.group(2)):
        letters, remainder = match.groups()
        first_digit = next((i for i, ch in enumerate(remainder) if ch.isdigit()), 0)
        return letters + remainder[:first_digit] + remainder[first_digit:].translate(DIGIT_CONFUSIONS)

    return clean


def is_probable_plate_text(text):
    clean = normalize_plate_text(text)
    if not 3 <= len(clean) <= 10:
        return False
    if not re.search(r"[A-Z]", clean) or not re.search(r"\d", clean):
        return False

    patterns = [
        r"^[A-Z]{1,4}\d{1,5}[A-Z]?$",   # WAA1234, ZD12447, SMJ1234
        r"^[1-9]\d{2,5}[A-Z]{1,2}$",     # trade-style suffix plates (e.g. 670J)
        r"^\d{1,3}[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$",
    ]
    return any(re.match(pattern, clean) for pattern in patterns)


def clamp_box(x, y, w, h, img_shape):
    ih, iw = img_shape[:2]
    x = max(0, min(int(x), iw - 1))
    y = max(0, min(int(y), ih - 1))
    w = max(1, min(int(w), iw - x))
    h = max(1, min(int(h), ih - y))
    return x, y, w, h


def expand_box(box, img_shape, x_scale=0.18, y_scale=0.45):
    x, y, w, h = box
    pad_x = max(6, int(w * x_scale))
    pad_y = max(5, int(h * y_scale))
    return clamp_box(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, img_shape)


def overlap_ratio(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0, min(ay + ah, by + bh) - max(ay, by))
    smaller = max(1, min(aw * ah, bw * bh))
    return (ox * oy) / smaller


# =============================================
# IMAGE PROCESSING FUNCTIONS (Core Logic)
# =============================================

def preprocess(img):
    """Full preprocessing pipeline for license plate detection."""
    resized = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    edges = cv2.Canny(blurred, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return resized, gray, enhanced, edges, closed


def detect_plates(resized, closed, min_area=0.003):
    """Find candidate plate regions via contour analysis."""
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h, w = resized.shape[:2]

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(ch) if ch > 0 else 0
        area_ratio = (cw * ch) / (w * h)

        if 1.4 < aspect < 8.5 and min_area < area_ratio < 0.15:
            candidates.append((x, y, cw, ch))

    # Keep largest and remove heavy overlaps
    candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
    filtered = []

    for box in candidates:
        x1, y1, w1, h1 = box
        dominated = False
        for fx, fy, fw, fh in filtered:
            ox = max(0, min(x1 + w1, fx + fw) - max(x1, fx))
            oy = max(0, min(y1 + h1, fy + fh) - max(y1, fy))
            if ox * oy > 0.5 * w1 * h1:
                dominated = True
                break
        if not dominated:
            filtered.append(box)

    return filtered[:8]


def detect_dark_plate_regions(resized, min_area=0.003):
    """Find dark horizontal regions such as black Malaysian plates."""
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    black = cv2.inRange(gray, 0, 95)
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(black, cv2.MORPH_CLOSE, horizontal, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(ch) if ch > 0 else 0
        area_ratio = (cw * ch) / (w * h)

        if not (1.6 < aspect < 8.5 and min_area < area_ratio < 0.12):
            continue

        crop = gray[y:y + ch, x:x + cw]
        if crop.size == 0:
            continue

        darkness = 1.0 - (float(np.mean(crop)) / 255.0)
        if darkness < 0.35:
            continue

        candidates.append(expand_box((x, y, cw, ch), resized.shape, 0.08, 0.25))

    candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
    filtered = []
    for box in candidates:
        if all(overlap_ratio(box, other) <= 0.55 for other in filtered):
            filtered.append(box)
    return filtered[:8]


def detect_bright_text_regions(resized, min_area=0.0002):
    """Find grouped bright plate characters on dark backgrounds."""
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    candidates = []

    for threshold in (150, 175):
        bright = cv2.inRange(gray, threshold, 255)
        close = cv2.morphologyEx(
            bright,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
            iterations=2,
        )
        close = cv2.morphologyEx(
            close,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(ch) if ch > 0 else 0
            area_ratio = (cw * ch) / (w * h)

            if not (1.4 < aspect < 10.0 and min_area < area_ratio < 0.035 and 12 < ch < 120):
                continue

            box = expand_box((x, y, cw, ch), resized.shape, 0.25, 0.80)
            bx, by, bw, bh = box
            crop = gray[by:by + bh, bx:bx + bw]
            if crop.size == 0:
                continue

            dark_fraction = float(np.mean(crop < 85))
            bright_fraction = float(np.mean(crop > threshold))
            if dark_fraction < 0.45 or bright_fraction < 0.04:
                continue

            score = dark_fraction + bright_fraction + min(aspect / 6.0, 1.0)
            candidates.append((score, box))

    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    filtered = []
    for _, box in candidates:
        if all(overlap_ratio(box, other) <= 0.55 for other in filtered):
            filtered.append(box)
    return filtered[:8]


def _readtext(reader, image, detail=0):
    kwargs = {
        "detail": detail,
        "allowlist": PLATE_ALLOWLIST,
        "paragraph": False,
    }
    try:
        return reader.readtext(image, **kwargs)
    except TypeError:
        kwargs.pop("paragraph", None)
        return reader.readtext(image, **kwargs)


def _merge_ocr_line(results):
    fragments = []
    for result in results:
        if len(result) < 2:
            continue

        pts = np.array(result[0], dtype=np.float32)
        text = normalize_plate_text(result[1])
        confidence = float(result[2]) if len(result) > 2 else 0.0
        if not text:
            continue

        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        fragments.append({
            "box": (x, y, w, h),
            "text": text,
            "confidence": confidence,
        })

    if not fragments:
        return "", 0.0

    fragments = sorted(fragments, key=lambda item: item["box"][0])
    grouped = []
    for fragment in fragments:
        x, y, w, h = fragment["box"]
        cy = y + h / 2.0
        added = False
        for group in grouped:
            gy = np.mean([item["box"][1] + item["box"][3] / 2.0 for item in group])
            gh = max(item["box"][3] for item in group)
            if abs(cy - gy) <= max(h, gh) * 1.2:
                group.append(fragment)
                added = True
                break
        if not added:
            grouped.append([fragment])

    best_text = ""
    best_score = -1.0
    best_confidence = 0.0
    for group in grouped:
        group = sorted(group, key=lambda item: item["box"][0])
        text = normalize_plate_text("".join(item["text"] for item in group))
        confidence = float(np.mean([item["confidence"] for item in group]))
        score = len(text) + confidence * 3.0
        if is_probable_plate_text(text):
            score += 6.0
        if score > best_score:
            best_text = text
            best_score = score
            best_confidence = confidence

    return best_text, best_confidence


def detect_ocr_plate_regions(resized, reader):
    """Use EasyOCR's rotated text boxes as plate candidates for angled views."""
    regions = []
    fragments = []

    try:
        results = _readtext(reader, resized, detail=1)
    except Exception:
        return regions

    for result in results:
        if len(result) < 2:
            continue

        poly, text = result[0], result[1]
        confidence = float(result[2]) if len(result) > 2 else 0.0
        clean = normalize_plate_text(text)
        if not clean:
            continue

        pts = np.array(poly, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        box = expand_box((x, y, w, h), resized.shape)
        fragments.append({
            "box": box,
            "poly": pts.astype(np.int32),
            "text": clean,
            "confidence": confidence,
        })

        if is_probable_plate_text(clean):
            regions.append({
                "box": box,
                "poly": pts.astype(np.int32),
                "text": clean,
                "confidence": confidence,
                "source": "ocr",
            })

    # EasyOCR can split angled plates into chunks such as "ZD1" and "2447".
    fragments = sorted(fragments, key=lambda item: (item["box"][1], item["box"][0]))
    for i, first in enumerate(fragments):
        group = [first]
        fx, fy, fw, fh = first["box"]
        first_cy = fy + fh / 2.0

        for second in fragments[i + 1:]:
            sx, sy, sw, sh = second["box"]
            second_cy = sy + sh / 2.0
            height = max(fh, sh)
            gap = sx - (fx + fw)
            same_line = abs(first_cy - second_cy) < height * 1.1
            near_enough = -height < gap < height * 5.0
            if same_line and near_enough:
                group.append(second)
                fx, fy, fw, fh = second["box"]

        if len(group) < 2:
            continue

        group = sorted(group, key=lambda item: item["box"][0])
        merged_text = "".join(item["text"] for item in group)
        if not is_probable_plate_text(merged_text):
            continue

        xs = [item["box"][0] for item in group]
        ys = [item["box"][1] for item in group]
        xe = [item["box"][0] + item["box"][2] for item in group]
        ye = [item["box"][1] + item["box"][3] for item in group]
        merged_box = expand_box(
            (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)),
            resized.shape,
            0.08,
            0.20,
        )
        merged_poly = cv2.convexHull(np.vstack([item["poly"] for item in group]))
        regions.append({
            "box": merged_box,
            "poly": merged_poly,
            "text": merged_text,
            "confidence": float(np.mean([item["confidence"] for item in group])),
            "source": "ocr-merged",
        })

    regions = sorted(
        regions,
        key=lambda item: (len(item["text"]), item["confidence"]),
        reverse=True,
    )
    filtered = []
    for region in regions:
        if all(overlap_ratio(region["box"], other["box"]) <= 0.65 for other in filtered):
            filtered.append(region)
    return filtered[:8]


def extract_plate_text(img_bgr, box, reader):
    """Crop, enhance, and run OCR on a single plate candidate."""
    x, y, w, h = box
    pad = 4
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(img_bgr.shape[1] - x, w + 2 * pad)
    h = min(img_bgr.shape[0] - y, h + 2 * pad)

    crop = img_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return "", 0.0

    # Upscale small crops
    if crop.shape[1] < 200:
        scale = 200 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    options = []
    for variant in (crop, clahe, binary, inverted):
        try:
            results = _readtext(reader, variant, detail=1)
        except Exception:
            continue

        merged_text, merged_confidence = _merge_ocr_line(results)
        if merged_text:
            options.append((merged_text, merged_confidence))

        for result in results:
            if len(result) < 2:
                continue
            text = normalize_plate_text(result[1])
            confidence = float(result[2]) if len(result) > 2 else 0.0
            if text:
                options.append((text, confidence))

    if not options:
        return "", 0.0

    options = sorted(
        options,
        key=lambda item: (is_probable_plate_text(item[0]), len(item[0]), item[1]),
        reverse=True,
    )
    return options[0]


def identify_state(plate_text):
    """Match longest prefix first for accurate state identification."""
    alnum_text = normalize_plate_text(plate_text)
    for suffix, state in TRADE_SUFFIX_MAP.items():
        if re.match(r"^[A-Z]{0,2}\d{2,5}[A-Z]{1,2}$", alnum_text) and alnum_text.endswith(suffix):
            return state, suffix

    for prefix in sorted(STATE_MAP, key=len, reverse=True):
        if not alnum_text.startswith(prefix):
            continue
        remainder = alnum_text[len(prefix):]
        if re.search(r"\d", remainder):
            return STATE_MAP[prefix], prefix

    # Diplomatic/organisation formats commonly end with CD/CC/DC/UN/PA.
    for suffix in ("CD", "CC", "DC", "UN", "PA"):
        if alnum_text.endswith(suffix) and re.match(r"^\d{2,4}[A-Z]{2,3}$", alnum_text):
            return STATE_MAP[suffix], suffix
    return "No Plate Detected", ""


def candidate_score(text, state, confidence=0.0):
    clean = normalize_plate_text(text)
    score = len(clean) + float(confidence) * 3.0
    if is_probable_plate_text(clean):
        score += 5.0
    if state != "No Plate Detected":
        score += 4.0
    if re.match(r"^Z[A-Z]{0,2}\d{3,5}$", clean):
        score += 2.0
    return score


def is_plausible_plate_box(box, img_shape, source="image"):
    x, y, w, h = box
    ih, iw = img_shape[:2]
    if w <= 0 or h <= 0:
        return False

    area_ratio = (w * h) / float(max(1, iw * ih))
    aspect = w / float(h)
    max_area = 0.12 if source.startswith("ocr") else 0.08

    if not (0.00022 < area_ratio < max_area):
        return False
    if not (1.6 < aspect < 9.0):
        return False
    if w < 45 or h < 12:
        return False
    return True


def is_strong_plate_candidate(text, state, confidence, box, img_shape, source="image"):
    clean = normalize_plate_text(text)
    if not is_probable_plate_text(clean):
        return False
    if not is_plausible_plate_box(box, img_shape, source):
        return False

    # OCR noise from signage or body stickers usually has low confidence and no
    # valid state match. Keep those out unless confidence is reasonably high.
    min_conf = 0.18 if source.startswith("ocr") else 0.30
    if state == "No Plate Detected" and confidence < min_conf:
        return False

    return True


def classify_vehicle(boxes, img_shape):
    """Simple heuristic-based vehicle classification."""
    if not boxes:
        return "Unknown"
    ih, iw = img_shape[:2]
    x, y, bw, bh = boxes[0]
    aspect = bw / float(bh) if bh > 0 else 1
    plate_y_ratio = (y + bh) / ih

    if aspect > 5 and plate_y_ratio > 0.85:
        return "Bus / Truck"
    if aspect < 2.5:
        return "Motorcycle"
    return "Car / Van"


# =============================================
# GUI APPLICATION
# =============================================

class AppSignals(QObject):
    ocr_ready = Signal(object)
    ocr_error = Signal(str)
    detection_ready = Signal(object)
    detection_error = Signal(str)


class ImagePreview(QLabel):
    def __init__(self):
        super().__init__("Load an image to begin")
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("imagePreview")

    def set_cv_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.setText("")
        self._refresh_pixmap()

    def clear_image(self):
        self._pixmap = None
        self.clear()
        self.setText("Load an image to begin")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class LPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Recognition & State Identification System")
        self.setMinimumSize(1100, 700)

        self.orig_img = None
        self.result_img = None
        self.reader = None
        self.reader_loaded = False
        self._current_img_bgr = None

        self.signals = AppSignals()
        self.signals.ocr_ready.connect(self._on_ocr_ready)
        self.signals.ocr_error.connect(self._on_ocr_error)
        self.signals.detection_ready.connect(self._update_results)
        self.signals.detection_error.connect(self._on_detection_error)

        self._build_ui()
        self._load_ocr_async()

    # ------------------- UI Construction -------------------
    def _build_ui(self):
        self._apply_styles()

        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        root_layout.addWidget(self._build_topbar())

        main = QWidget()
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)
        root_layout.addWidget(main, 1)

        main_layout.addWidget(self._build_controls())
        main_layout.addWidget(self._build_image_area(), 1)
        main_layout.addWidget(self._build_results())

    def _build_topbar(self):
        topbar = QFrame()
        topbar.setObjectName("topbar")
        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(20, 10, 20, 10)

        title = QLabel("LPR & State Identification System")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setObjectName("topbarTitle")

        self.status_lbl = QLabel("Loading OCR engine...")
        self.status_lbl.setObjectName("statusLabel")
        self.status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.status_lbl)
        return topbar

    def _build_controls(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(240)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        heading = QLabel("Controls")
        heading.setObjectName("sectionHeading")
        heading.setAlignment(Qt.AlignCenter)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.run_btn = QPushButton("Run Detection")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_detection)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)

        self.vehicle_combo = QComboBox()
        self.vehicle_combo.addItems(["Auto-detect"] + VEHICLE_TYPES)

        self.show_steps_checkbox = QCheckBox("Enable")

        self.area_value_lbl = QLabel("Min plate area: 0.3%")
        self.area_slider = QSlider(Qt.Horizontal)
        self.area_slider.setRange(1, 20)
        self.area_slider.setValue(3)
        self.area_slider.valueChanged.connect(self._update_area_label)

        layout.addWidget(heading)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.clear_btn)
        layout.addSpacing(14)
        layout.addWidget(self._small_label("Vehicle type"))
        layout.addWidget(self.vehicle_combo)
        layout.addSpacing(10)
        layout.addWidget(self._small_label("Show preprocessing steps"))
        layout.addWidget(self.show_steps_checkbox)
        layout.addSpacing(10)
        layout.addWidget(self.area_value_lbl)
        layout.addWidget(self.area_slider)
        layout.addStretch(1)
        return panel

    def _build_image_area(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel("Image preview")
        label.setObjectName("smallLabel")
        self.image_preview = ImagePreview()

        layout.addWidget(label)
        layout.addWidget(self.image_preview, 1)
        return panel

    def _build_results(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        heading = QLabel("Detection results")
        heading.setObjectName("sectionHeading")

        self.plate_lbl = self._result_value("Not detected", "plateValue")
        self.state_lbl = self._result_value("-", "stateValue")
        self.prefix_lbl = self._result_value("-", "normalValue")
        self.vtype_lbl = self._result_value("-", "vehicleValue")
        self.count_lbl = self._result_value("-", "normalValue")
        self.cand_list = QListWidget()
        self.cand_list.setObjectName("candidateList")
        self.cand_list.setMinimumHeight(90)

        self.save_btn = QPushButton("Save result image")
        self.save_btn.clicked.connect(self.save_result)

        layout.addWidget(heading)
        layout.addWidget(self._small_label("Plate text"))
        layout.addWidget(self.plate_lbl)
        layout.addWidget(self._small_label("Registered state"))
        layout.addWidget(self.state_lbl)
        layout.addWidget(self._small_label("Prefix detected"))
        layout.addWidget(self.prefix_lbl)
        layout.addWidget(self._small_label("Vehicle type"))
        layout.addWidget(self.vtype_lbl)
        layout.addWidget(self._small_label("Plates found"))
        layout.addWidget(self.count_lbl)
        layout.addWidget(self._small_label("All candidates"))
        layout.addWidget(self.cand_list)
        layout.addStretch(1)
        layout.addWidget(self.save_btn)
        return panel

    def _small_label(self, text):
        label = QLabel(text)
        label.setObjectName("smallLabel")
        return label

    def _result_value(self, text, object_name):
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        return label

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget#root {{
                background: {COLORS["background"]};
                color: {COLORS["text_secondary"]};
                font-family: "Segoe UI", sans-serif;
                font-size: 10pt;
            }}
            QFrame#topbar {{
                background: {COLORS["foreground"]};
            }}
            QLabel#topbarTitle {{
                color: {COLORS["text_primary"]};
            }}
            QLabel#statusLabel {{
                color: {COLORS["text_secondary"]};
            }}
            QFrame#sidePanel {{
                background: {COLORS["background"]};
            }}
            QLabel#sectionHeading {{
                color: {COLORS["text_primary"]};
                font-weight: 700;
                font-size: 12pt;
            }}
            QLabel#smallLabel {{
                color: {COLORS["text_muted"]};
                font-size: 9pt;
            }}
            QLabel#imagePreview {{
                background: {COLORS["foreground"]};
                color: white;
            }}
            QPushButton {{
                border: 0;
                padding: 9px 12px;
                background: {COLORS["foreground"]};
                color: #1e1e2e;
            }}
            QPushButton:hover {{
                background: {COLORS["accent_purple"]};
            }}
            QPushButton:disabled {{
                background: #cfc8c2;
                color: #777777;
            }}
            QComboBox, QListWidget {{
                background: {COLORS["foreground"]};
                color: {COLORS["text_primary"]};
                border: 0;
                padding: 6px;
            }}
            QCheckBox {{
                color: {COLORS["text_primary"]};
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: #313244;
            }}
            QSlider::handle:horizontal {{
                width: 16px;
                margin: -4px 0;
                background: white;
                border: 1px solid #777777;
            }}
            QLabel#plateValue {{
                color: {COLORS["accent_pink"]};
                font: 700 20pt "Courier New";
            }}
            QLabel#stateValue {{
                color: {COLORS["accent_green"]};
                font-weight: 700;
                font-size: 14pt;
            }}
            QLabel#vehicleValue {{
                color: {COLORS["accent_yellow"]};
            }}
            QLabel#normalValue {{
                color: {COLORS["text_secondary"]};
            }}
        """)

    # ------------------- OCR Loading -------------------
    def _load_ocr_async(self):
        def load():
            try:
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.signals.ocr_ready.emit(reader)
            except Exception as exc:
                self.signals.ocr_error.emit(str(exc))

        threading.Thread(target=load, daemon=True).start()

    def _on_ocr_ready(self, reader):
        self.reader = reader
        self.reader_loaded = True
        self._set_status("Ready", COLORS["accent_green"])
        self._sync_run_button()

    def _on_ocr_error(self, message):
        self._set_status(f"OCR error: {message}", COLORS["error"])

    # ------------------- Image Handling -------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp)",
        )
        if not path:
            return

        self.orig_img = cv2.imread(path)
        if self.orig_img is None:
            self._set_status("Failed to load image", COLORS["error"])
            return

        self.result_img = self.orig_img.copy()
        self._display(self.orig_img)
        self._set_status(f"Loaded: {os.path.basename(path)}", COLORS["text_secondary"])
        self._reset_results()
        self._sync_run_button()

    def _display(self, img_bgr):
        self._current_img_bgr = img_bgr.copy()
        self.image_preview.set_cv_image(self._current_img_bgr)

    # ------------------- Detection Pipeline -------------------
    def run_detection(self):
        if self.orig_img is None or not self.reader_loaded:
            return

        self._detection_options = {
            "min_area": (self.area_slider.value() / 10.0) / 100.0,
            "vehicle_type": self.vehicle_combo.currentText(),
            "show_steps": self.show_steps_checkbox.isChecked(),
        }
        self.run_btn.setEnabled(False)
        self._set_status("Processing...", COLORS["text_highlight"])
        threading.Thread(target=self._detect_thread, daemon=True).start()

    def _detect_thread(self):
        try:
            img = self.orig_img.copy()
            options = self._detection_options
            resized, gray, enhanced, edges, closed = preprocess(img)
            min_area = options["min_area"]
            contour_boxes = detect_plates(resized, closed, min_area)
            dark_boxes = detect_dark_plate_regions(resized, min_area)
            bright_boxes = detect_bright_text_regions(resized)
            ocr_regions = detect_ocr_plate_regions(resized, self.reader)

            plate_regions = list(ocr_regions)
            image_regions = []
            for box in bright_boxes + contour_boxes + dark_boxes:
                if not is_plausible_plate_box(box, resized.shape, source="image"):
                    continue
                if all(overlap_ratio(box, region["box"]) <= 0.65 for region in image_regions):
                    image_regions.append({
                        "box": box,
                        "poly": None,
                        "text": "",
                        "confidence": 0.0,
                        "source": "image",
                    })
            plate_regions.extend(image_regions)
            plate_regions = plate_regions[:14]
            boxes = [region["box"] for region in plate_regions]

            draw = resized.copy()
            filtered_candidates = []
            best_text = ""
            best_state = "No Plate Detected"
            best_prefix = ""
            best_score = -1.0

            for region in plate_regions:
                box = region["box"]
                source = region.get("source", "image")
                text = normalize_plate_text(region.get("text", ""))
                confidence = float(region.get("confidence", 0.0))

                if not text:
                    text, extracted_confidence = extract_plate_text(resized, box, self.reader)
                    confidence = max(confidence, extracted_confidence)
                text = normalize_plate_text(text)
                if len(text) < 3:
                    continue
                state, prefix = identify_state(text)

                # OCR regions can be oversized on stacked motorcycle plates.
                # Re-anchor to a plausible overlapping image-derived box when available.
                if source.startswith("ocr") and not is_plausible_plate_box(box, resized.shape, source):
                    anchors = []
                    for image_region in image_regions:
                        anchor_box = image_region["box"]
                        if overlap_ratio(box, anchor_box) > 0.55 and is_plausible_plate_box(
                            anchor_box, resized.shape, source="image"
                        ):
                            anchors.append(anchor_box)
                    if anchors:
                        box = max(anchors, key=lambda b: b[2] * b[3])

                if not is_strong_plate_candidate(
                    text, state, confidence, box, resized.shape, source
                ):
                    continue

                score = candidate_score(text, state, confidence)
                if source.startswith("ocr"):
                    score += 1.0

                filtered_candidates.append({
                    "text": text,
                    "state": state,
                    "prefix": prefix,
                    "score": score,
                    "box": box,
                    "poly": region.get("poly"),
                })

            filtered_candidates = sorted(
                filtered_candidates, key=lambda item: item["score"], reverse=True
            )

            selected = []
            best_candidate_score = filtered_candidates[0]["score"] if filtered_candidates else -1.0
            for cand in filtered_candidates:
                if cand["score"] < 14.0:
                    continue
                if cand["score"] < best_candidate_score - 2.5:
                    continue
                if any(overlap_ratio(cand["box"], other["box"]) > 0.70 for other in selected):
                    continue
                if any(
                    cand["text"] == other["text"] and overlap_ratio(cand["box"], other["box"]) > 0.40
                    for other in selected
                ):
                    continue
                selected.append(cand)
                if len(selected) >= 6:
                    break

            candidates = [(item["text"], item["state"]) for item in selected]
            boxes = [item["box"] for item in selected] or boxes

            for item in selected:
                x, y, w, h = item["box"]
                state = item["state"]
                text = item["text"]

                color = (0, 255, 100) if state != "No Plate Detected" else (0, 180, 255)
                if item["poly"] is not None:
                    cv2.polylines(draw, [item["poly"].astype(np.int32)], True, color, 2)
                cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
                label = f"{text} | {state}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(draw, (x, y - lh - 10), (x + lw + 8, y), color, -1)
                cv2.putText(draw, label, (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                if item["score"] > best_score:
                    best_score = item["score"]
                    best_text = text
                    best_state = state
                    best_prefix = item["prefix"]

            vtype = options["vehicle_type"]
            if vtype == "Auto-detect":
                vtype = classify_vehicle(boxes, resized.shape)

            if options["show_steps"]:
                steps = [
                    ("Grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
                    ("CLAHE enhanced", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)),
                    ("Canny edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
                ]
                th = 120
                strip_imgs = []
                for name, simg in steps:
                    s = cv2.resize(simg, (int(simg.shape[1] * th / simg.shape[0]), th))
                    cv2.putText(s, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 180), 1)
                    strip_imgs.append(s)

                strip = np.zeros((th + 4, draw.shape[1], 3), dtype=np.uint8)
                offset = 4
                for s in strip_imgs:
                    if offset + s.shape[1] > draw.shape[1]:
                        break
                    strip[2:th + 2, offset:offset + s.shape[1]] = s
                    offset += s.shape[1] + 4
                draw = np.vstack([draw, strip])

            self.signals.detection_ready.emit({
                "text": best_text,
                "state": best_state,
                "prefix": best_prefix,
                "vehicle_type": vtype,
                "count": len(candidates),
                "candidates": candidates,
                "draw": draw,
            })

        except Exception as exc:
            self.signals.detection_error.emit(str(exc))

    def _update_results(self, result):
        self.result_img = result["draw"]
        self.plate_lbl.setText(result["text"] if result["text"] else "Not detected")
        self.state_lbl.setText(result["state"])
        self.prefix_lbl.setText(result["prefix"] if result["prefix"] else "-")
        self.vtype_lbl.setText(result["vehicle_type"])
        self.count_lbl.setText(str(result["count"]))

        self.cand_list.clear()
        for text, state in result["candidates"]:
            self.cand_list.addItem(f"{text:<12} -> {state}")

        self._display(result["draw"])
        self._set_status("Detection complete", COLORS["accent_green"])
        self._sync_run_button()

    def _on_detection_error(self, message):
        self._set_status(f"Error: {message}", COLORS["error"])
        self._sync_run_button()

    # ------------------- Utility Methods -------------------
    def _update_area_label(self, value):
        self.area_value_lbl.setText(f"Min plate area: {value / 10.0:.1f}%")

    def _set_status(self, text, color):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color: {color};")

    def _sync_run_button(self):
        self.run_btn.setEnabled(self.orig_img is not None and self.reader_loaded)

    def _reset_results(self):
        self.plate_lbl.setText("Not detected")
        self.state_lbl.setText("-")
        self.prefix_lbl.setText("-")
        self.vtype_lbl.setText("-")
        self.count_lbl.setText("-")
        self.cand_list.clear()

    def clear_all(self):
        self.orig_img = None
        self.result_img = None
        self._current_img_bgr = None
        self.image_preview.clear_image()
        self._reset_results()
        self._sync_run_button()
        status = "Ready" if self.reader_loaded else "Loading OCR engine..."
        self._set_status(status, COLORS["accent_green"])

    def save_result(self):
        if self.result_img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save result image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;HEIC (*.heic)",
        )
        if path:
            cv2.imwrite(path, self.result_img)
            self._set_status(f"Saved: {os.path.basename(path)}", COLORS["accent_green"])


# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LPRApp()
    window.show()
    sys.exit(app.exec())
