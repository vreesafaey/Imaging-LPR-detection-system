"""
License Plate Recognition (LPR) & State Identification System (SIS)
GUI built with Tkinter | Detection via OpenCV | OCR via EasyOCR
"""

import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import easyocr
import re
import threading
import os
import warnings

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
        r"^\d{1,5}[A-Z]{1,3}$",          # trade and some special plates
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
        return ""

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
        return ""

    options = sorted(
        options,
        key=lambda item: (is_probable_plate_text(item[0]), len(item[0]), item[1]),
        reverse=True,
    )
    return options[0][0]


def identify_state(plate_text):
    """Match longest prefix first for accurate state identification."""
    alnum_text = normalize_plate_text(plate_text)
    for suffix, state in TRADE_SUFFIX_MAP.items():
        if re.match(r"^[A-Z]{0,2}\d+[A-Z]?$", alnum_text) and alnum_text.endswith(suffix):
            return state, suffix

    plate_text = re.sub(r"[^A-Z]", "", alnum_text)
    for prefix in sorted(STATE_MAP, key=len, reverse=True):
        if plate_text.startswith(prefix):
            return STATE_MAP[prefix], prefix
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

class LPRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("License Plate Recognition & State Identification System")
        self.configure(bg=COLORS["background"])
        self.resizable(True, True)
        self.minsize(1100, 700)

        # State variables
        self.orig_img = None
        self.result_img = None
        self.reader = None
        self.reader_loaded = False
        self._current_img_bgr = None

        self._build_ui()
        self._load_ocr_async()

    # ------------------- UI Construction -------------------
    def _build_ui(self):
        self._build_topbar()
        self._build_main_content()

    def _build_topbar(self):
        topbar = tk.Frame(self, bg=COLORS["foreground"], pady=10)
        topbar.pack(fill="x")

        tk.Label(topbar, text="LPR & State Identification System",
                 font=("Segoe UI", 16, "bold"), fg=COLORS["text_primary"],
                 bg=COLORS["foreground"]).pack(side="left", padx=20)

        self.status_lbl = tk.Label(topbar, text="Loading OCR engine...",
                                   font=("Segoe UI", 10), fg=COLORS["text_secondary"],
                                   bg=COLORS["foreground"])
        self.status_lbl.pack(side="right", padx=20)

    def _build_main_content(self):
        main = tk.Frame(self, bg=COLORS["background"])
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # Left: Controls
        left = tk.Frame(main, bg=COLORS["background"], width=240)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._build_controls(left)

        # Centre: Image Canvas
        centre = tk.Frame(main, bg=COLORS["background"])
        centre.pack(side="left", fill="both", expand=True)
        self._build_canvas(centre)

        # Right: Results
        right = tk.Frame(main, bg=COLORS["background"], width=280)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_results(right)

    def _build_controls(self, parent):
        tk.Label(parent, text="Controls", font=("Segoe UI", 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["background"]).pack(pady=(16, 8))

        btn_style = {
            "font": ("Segoe UI", 10),
            "relief": "flat",
            "cursor": "hand2",
            "pady": 8
        }

        self.load_btn = tk.Button(parent, text="Load Image",
                                  bg=COLORS["accent_blue"], fg="#1e1e2e",
                                  command=self.load_image, **btn_style)
        self.load_btn.pack(fill="x", padx=16, pady=4)

        self.run_btn = tk.Button(parent, text="Run Detection",
                                 bg=COLORS["accent_green"], fg="#1e1e2e",
                                 command=self.run_detection, state="disabled", **btn_style)
        self.run_btn.pack(fill="x", padx=16, pady=4)

        tk.Button(parent, text="Clear", bg=COLORS["foreground"],
                  fg=COLORS["text_primary"], command=self.clear_all, **btn_style).pack(fill="x", padx=16, pady=4)

        # Vehicle Type
        tk.Label(parent, text="Vehicle type", font=("Segoe UI", 9),
                 fg=COLORS["text_secondary"], bg=COLORS["background"]).pack(pady=(20, 4))
        self.vehicle_var = tk.StringVar(value="Auto-detect")
        ttk.Combobox(parent, textvariable=self.vehicle_var,
                     values=["Auto-detect"] + VEHICLE_TYPES, state="readonly").pack(fill="x", padx=16)

        # Preprocessing toggle
        tk.Label(parent, text="Show preprocessing steps", font=("Segoe UI", 9),
                 fg=COLORS["text_secondary"], bg=COLORS["background"]).pack(pady=(20, 4))
        self.show_steps_var = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Enable", variable=self.show_steps_var,
                       bg=COLORS["background"], fg=COLORS["text_primary"],
                       selectcolor="#313244", activebackground=COLORS["background"],
                       font=("Segoe UI", 9)).pack(padx=16, anchor="w")

        # Min area slider
        tk.Label(parent, text="Min plate area (%)", font=("Segoe UI", 9),
                 fg=COLORS["text_secondary"], bg=COLORS["background"]).pack(pady=(16, 4))
        self.area_var = tk.DoubleVar(value=0.3)
        tk.Scale(parent, from_=0.1, to=2.0, resolution=0.1,
                 variable=self.area_var, orient="horizontal",
                 bg=COLORS["background"], fg=COLORS["text_primary"],
                 troughcolor="#313244", highlightthickness=0).pack(fill="x", padx=16)

    def _build_canvas(self, parent):
        tk.Label(parent, text="Image preview", font=("Segoe UI", 10),
                 fg=COLORS["text_secondary"], bg=COLORS["background"]).pack(anchor="w", pady=(0, 4))

        canvas_frame = tk.Frame(parent, bg=COLORS["foreground"], relief="flat", bd=1)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=COLORS["foreground"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas())

        self.canvas_hint = self.canvas.create_text(
            400, 250, text="Load an image to begin",
            font=("Segoe UI", 14), fill="#ffffff")

    def _build_results(self, parent):
        tk.Label(parent, text="Detection results", font=("Segoe UI", 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["background"]).pack(pady=(16, 8), padx=16, anchor="w")

        self.plate_var = tk.StringVar(value="—")
        self.state_var = tk.StringVar(value="—")
        self.prefix_var = tk.StringVar(value="—")
        self.vtype_var = tk.StringVar(value="—")
        self.count_var = tk.StringVar(value="—")

        self._result_row(parent, "Plate text")
        tk.Label(parent, textvariable=self.plate_var,
                 font=("Courier New", 20, "bold"), fg=COLORS["accent_pink"],
                 bg=COLORS["background"]).pack(padx=16, anchor="w")

        self._result_row(parent, "Registered state")
        tk.Label(parent, textvariable=self.state_var,
                 font=("Segoe UI", 14, "bold"), fg=COLORS["accent_green"],
                 bg=COLORS["background"]).pack(padx=16, anchor="w")

        self._result_row(parent, "Prefix detected")
        tk.Label(parent, textvariable=self.prefix_var,
                 font=("Segoe UI", 11), fg="#89dceb", bg=COLORS["background"]).pack(padx=16, anchor="w")

        self._result_row(parent, "Vehicle type")
        tk.Label(parent, textvariable=self.vtype_var,
                 font=("Segoe UI", 11), fg=COLORS["accent_yellow"],
                 bg=COLORS["background"]).pack(padx=16, anchor="w")

        self._result_row(parent, "Plates found")
        tk.Label(parent, textvariable=self.count_var,
                 font=("Segoe UI", 11), fg=COLORS["text_primary"],
                 bg=COLORS["background"]).pack(padx=16, anchor="w")

        self._result_row(parent, "All candidates")
        self.cand_list = tk.Listbox(parent, bg=COLORS["foreground"], fg=COLORS["text_primary"],
                                    font=("Courier New", 10), height=6,
                                    selectbackground="#585b70", relief="flat",
                                    highlightthickness=0)
        self.cand_list.pack(fill="x", padx=16, pady=(4, 0))

        tk.Button(parent, text="Save result image", font=("Segoe UI", 9),
                  bg=COLORS["accent_purple"], fg="#1e1e2e", relief="flat", cursor="hand2",
                  command=self.save_result).pack(fill="x", padx=16, pady=16)

    def _result_row(self, parent, label):
        tk.Label(parent, text=label, font=("Segoe UI", 8),
                 fg=COLORS["text_muted"], bg=COLORS["background"]).pack(
            padx=16, anchor="w", pady=(12, 0))

    # ------------------- OCR Loading -------------------
    def _load_ocr_async(self):
        def load():
            try:
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.reader_loaded = True
                self.after(0, lambda: self.status_lbl.config(
                    text="Ready", fg=COLORS["accent_green"]))
                if self.orig_img is not None:
                    self.after(0, lambda: self.run_btn.config(state="normal"))
            except Exception as e:
                self.after(0, lambda: self.status_lbl.config(
                    text=f"OCR error: {e}", fg=COLORS["error"]))
        threading.Thread(target=load, daemon=True).start()

    # ------------------- Image Handling -------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp")])
        if not path:
            return

        self.orig_img = cv2.imread(path)
        if self.orig_img is None:
            self.status_lbl.config(text="Failed to load image", fg=COLORS["error"])
            return

        self.result_img = self.orig_img.copy()
        self._display(self.orig_img)
        self.status_lbl.config(text=f"Loaded: {os.path.basename(path)}",
                               fg=COLORS["text_secondary"])

        if self.reader_loaded:
            self.run_btn.config(state="normal")

        self._reset_results()

    def _display(self, img_bgr):
        self._current_img_bgr = img_bgr.copy()
        self._refresh_canvas()

    def _refresh_canvas(self):
        if not hasattr(self, "_current_img_bgr") or self._current_img_bgr is None:
            return

        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 600
        ch = self.canvas.winfo_height() or 400

        img = self._current_img_bgr
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(pil)

        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.create_image(ox, oy, anchor="nw", image=self._tk_img)

    # ------------------- Detection Pipeline -------------------
    def run_detection(self):
        if self.orig_img is None or not self.reader_loaded:
            return

        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Processing...", fg=COLORS["text_highlight"])

        threading.Thread(target=self._detect_thread, daemon=True).start()

    def _detect_thread(self):
        try:
            img = self.orig_img.copy()
            resized, gray, enhanced, edges, closed = preprocess(img)
            min_area = self.area_var.get() / 100.0
            contour_boxes = detect_plates(resized, closed, min_area)
            dark_boxes = detect_dark_plate_regions(resized, min_area)
            bright_boxes = detect_bright_text_regions(resized)
            ocr_regions = detect_ocr_plate_regions(resized, self.reader)

            plate_regions = list(ocr_regions)
            image_regions = []
            for box in bright_boxes + contour_boxes + dark_boxes:
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
            candidates = []
            best_text = ""
            best_state = "Unknown"
            best_prefix = ""
            best_score = -1.0

            for region in plate_regions:
                box = region["box"]
                text = region.get("text") or extract_plate_text(resized, box, self.reader)
                text = normalize_plate_text(text)
                if len(text) < 3:
                    continue
                state, prefix = identify_state(text)
                score = candidate_score(text, state, region.get("confidence", 0.0))
                candidates.append((text, state))

                x, y, w, h = box
                color = (0, 255, 100) if state != "No Plate Detected" else (0, 180, 255)
                if region.get("poly") is not None:
                    cv2.polylines(draw, [region["poly"].astype(np.int32)], True, color, 2)
                cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
                label = f"{text} | {state}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(draw, (x, y - lh - 10), (x + lw + 8, y), color, -1)
                cv2.putText(draw, label, (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                if score > best_score:
                    best_score = score
                    best_text, best_state, best_prefix = text, state, prefix

            if not best_text and candidates:
                best_text, best_state = candidates[0]
                _, best_prefix = identify_state(best_text)

            vtype = self.vehicle_var.get()
            if vtype == "Auto-detect":
                vtype = classify_vehicle(boxes, resized.shape)

            # Preprocessing steps visualization
            if self.show_steps_var.get():
                # (Same logic as before - kept unchanged for brevity)
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

            self.result_img = draw
            self.after(0, lambda: self._update_results(
                best_text, best_state, best_prefix, vtype,
                len([c for c in candidates if c[0]]), candidates, draw))

        except Exception as e:
            self.after(0, lambda: self.status_lbl.config(
                text=f"Error: {e}", fg=COLORS["error"]))
            self.after(0, lambda: self.run_btn.config(state="normal"))

    def _update_results(self, text, state, prefix, vtype, count, candidates, draw):
        self.plate_var.set(text if text else "Not detected")
        self.state_var.set(state)
        self.prefix_var.set(prefix if prefix else "—")
        self.vtype_var.set(vtype)
        self.count_var.set(str(count))

        self.cand_list.delete(0, "end")
        for t, s in candidates:
            self.cand_list.insert("end", f"{t:<12} → {s}")

        self._display(draw)
        self.status_lbl.config(text="Detection complete", fg=COLORS["accent_green"])
        self.run_btn.config(state="normal")

    # ------------------- Utility Methods -------------------
    def _reset_results(self):
        for var in [self.plate_var, self.state_var, self.prefix_var,
                    self.vtype_var, self.count_var]:
            var.set("—")
        self.cand_list.delete(0, "end")

    def clear_all(self):
        self.orig_img = None
        self.result_img = None
        self._current_img_bgr = None
        self.canvas.delete("all")
        self.canvas.create_text(400, 250, text="Load an image to begin",
                                font=("Segoe UI", 14), fill="#45475a")
        self._reset_results()
        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Ready", fg=COLORS["accent_green"])

    def save_result(self):
        if self.result_img is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("HEIC", "*.heic")])
        if path:
            cv2.imwrite(path, self.result_img)
            self.status_lbl.config(text=f"Saved: {os.path.basename(path)}",
                                   fg=COLORS["accent_green"])


# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    app = LPRApp()
    app.mainloop()
