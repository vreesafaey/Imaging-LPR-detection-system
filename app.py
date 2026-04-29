"""
License Plate Recognition (LPR) & State Identification System (SIS)
NiceGUI dashboard | Detection via OpenCV | OCR via EasyOCR
"""

import asyncio
import base64
import re
import warnings
from typing import Any

import cv2
import easyocr
import numpy as np
from nicegui import events, ui

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


def plate_position_score(box, img_shape):
    """Prefer the lower bumper/front regions where bus/car plates usually sit."""
    x, y, w, h = box
    ih, iw = img_shape[:2]
    center_y = (y + h / 2.0) / float(max(1, ih))
    width_ratio = w / float(max(1, iw))

    score = 0.0
    if center_y > 0.52:
        score += 1.2
    elif center_y < 0.35:
        score -= 1.5

    if width_ratio > 0.28 and center_y < 0.70:
        score -= 1.2
    return score


def is_suspicious_non_plate_region(text, box, img_shape, source="image"):
    """Reject short OCR text from large upper/mid vehicle branding or windows."""
    clean = normalize_plate_text(text)
    x, y, w, h = box
    ih, iw = img_shape[:2]
    center_y = (y + h / 2.0) / float(max(1, ih))
    width_ratio = w / float(max(1, iw))
    area_ratio = (w * h) / float(max(1, iw * ih))

    if source.startswith("ocr") and center_y < 0.34:
        return True
    if len(clean) <= 5 and center_y < 0.72 and (width_ratio > 0.18 or area_ratio > 0.012):
        return True
    return False


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
    width_ratio = bw / float(max(1, iw))
    height_ratio = bh / float(max(1, ih))
    plate_y_ratio = (y + bh) / ih

    if aspect > 5 and plate_y_ratio > 0.85:
        return "Bus / Truck"
    if aspect < 2.5 and width_ratio > 0.10 and height_ratio > 0.045 and plate_y_ratio < 0.75:
        return "Motorcycle"
    return "Car / Van"
# =============================================
# NICEGUI APPLICATION
# =============================================

reader = None
state: dict[str, Any] = {
    "image": None,
    "processed_image": None,
    "uploaded_name": "",
    "result": None,
}


def get_reader():
    """Load EasyOCR once and reuse it across detections."""
    global reader
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return reader


def image_to_data_url(img_bgr, ext=".jpg"):
    ok, encoded = cv2.imencode(ext, img_bgr)
    if not ok:
        return ""
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def bytes_to_data_url(data: bytes, content_type: str | None):
    mime = content_type or "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def decode_upload(data: bytes):
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def detect_plate(image_bgr, options=None):
    """Wrap the existing detection pipeline and return UI-friendly results."""
    options = options or {}
    reader_instance = get_reader()
    resized, gray, enhanced, edges, closed = preprocess(image_bgr)
    min_area = float(options.get("min_area", 0.003))
    small_plate_min_area = min(min_area, 0.0008)
    contour_boxes = detect_plates(resized, closed, small_plate_min_area)
    dark_boxes = detect_dark_plate_regions(resized, small_plate_min_area)
    bright_boxes = detect_bright_text_regions(resized)
    ocr_regions = detect_ocr_plate_regions(resized, reader_instance)

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
            text, extracted_confidence = extract_plate_text(resized, box, reader_instance)
            confidence = max(confidence, extracted_confidence)
        text = normalize_plate_text(text)
        if len(text) < 3:
            continue
        plate_state, prefix = identify_state(text)

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

        if is_suspicious_non_plate_region(text, box, resized.shape, source):
            continue

        if not is_strong_plate_candidate(
            text, plate_state, confidence, box, resized.shape, source
        ):
            continue

        score = candidate_score(text, plate_state, confidence)
        if source.startswith("ocr"):
            score += 1.0
        score += plate_position_score(box, resized.shape)
        if len(text) <= 5:
            score -= 1.0

        filtered_candidates.append({
            "text": text,
            "state": plate_state,
            "prefix": prefix,
            "score": score,
            "box": box,
            "poly": region.get("poly"),
            "confidence": confidence,
        })

    filtered_candidates = sorted(
        filtered_candidates, key=lambda item: item["score"], reverse=True
    )

    selected = []
    best_candidate_score = filtered_candidates[0]["score"] if filtered_candidates else -1.0
    for candidate in filtered_candidates:
        if candidate["score"] < 14.0:
            continue
        if candidate["score"] < best_candidate_score - 1.5:
            continue
        if any(overlap_ratio(candidate["box"], other["box"]) > 0.70 for other in selected):
            continue
        if any(
            candidate["text"] == other["text"]
            and overlap_ratio(candidate["box"], other["box"]) > 0.40
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 6:
            break

    candidate_images = []
    boxes = [item["box"] for item in selected] or boxes
    for item in selected:
        x, y, w, h = item["box"]
        crop = resized[y:y + h, x:x + w]
        if crop.size:
            candidate_images.append({
                "text": item["text"],
                "state": item["state"],
                "image": crop.copy(),
            })

        color = (0, 255, 100) if item["state"] != "No Plate Detected" else (0, 180, 255)
        if item["poly"] is not None:
            cv2.polylines(draw, [item["poly"].astype(np.int32)], True, color, 2)
        cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
        label = f"{item['text']} | {item['state']}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(draw, (x, y - lh - 10), (x + lw + 8, y), color, -1)
        cv2.putText(
            draw,
            label,
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if item["score"] > best_score:
            best_score = item["score"]
            best_text = item["text"]
            best_state = item["state"]
            best_prefix = item["prefix"]

    vehicle_type = options.get("vehicle_type", "Auto-detect")
    if vehicle_type == "Auto-detect":
        vehicle_type = classify_vehicle(boxes, resized.shape)

    if options.get("show_steps", False):
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

    return {
        "plate_text": best_text,
        "registered_state": best_state,
        "prefix": best_prefix,
        "vehicle_type": vehicle_type,
        "plates_found": len(selected),
        "candidates": selected,
        "candidate_images": candidate_images,
        "processed_image": draw,
    }


ui.dark_mode().disable()
ui.add_head_html("""
<style>
html, body {
    min-height: 100%;
    overflow-y: auto;
    background: #f6f7fb;
    color: #1e293b;
}
.nicegui-content { padding: 0; }
.dashboard-bg {
    min-height: 100vh;
    overflow: visible;
    background:
        radial-gradient(circle at top left, rgba(245, 158, 11, .16), transparent 26%),
        linear-gradient(135deg, #f6f7fb 0%, #eef2f7 52%, #f8fafc 100%);
    box-sizing: border-box;
}
.hero-panel {
    background: linear-gradient(135deg, #ffffff 0%, #fff7ed 100%);
    border: 1px solid #e2e8f0;
    border-left: 6px solid #f59e0b;
    box-shadow: 0 18px 45px rgba(15, 23, 42, .08);
}
.status-badge {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fcd34d;
    border-radius: 999px;
    padding: .45rem .85rem;
    font-weight: 700;
}
.dashboard-main { min-height: calc(100vh - 8rem); }
.glass-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-radius: 18px !important;
    box-shadow: 0 18px 45px rgba(15, 23, 42, .08);
    min-height: 0;
    box-sizing: border-box;
}
.preview-frame {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    min-height: 34rem;
}
.scroll-panel { overflow: visible; min-height: 0; }
.crop-strip {
    min-height: 9rem;
    overflow-x: auto;
    overflow-y: visible;
    flex-wrap: nowrap;
}
.compact-upload { overflow: visible; }
.compact-upload .q-uploader { min-height: 0; }
.compact-upload .q-uploader__header {
    min-height: 3.25rem;
    padding: .35rem .65rem;
    background: #f59e0b !important;
}
.compact-upload .q-uploader__list { max-height: none; overflow: visible; padding: .35rem; }
.compact-upload .q-uploader__file { min-height: 8rem; }
.compact-upload .q-uploader__file--img {
    height: 8rem;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
}
.preview-frame .q-img,
.preview-frame .q-img__container,
.preview-frame .q-img__image {
    width: 100%;
    height: 100%;
}
.preview-frame .q-img__image {
    object-fit: contain !important;
}
.control-spacer { height: .25rem; }
.results-list { overflow-y: auto; min-height: 4rem; }
.result-value { color: #0f172a; font-weight: 800; }
.muted { color: #64748b; }
.primary-button {
    background: #f59e0b !important;
    color: #111827 !important;
    box-shadow: 0 10px 20px rgba(245, 158, 11, .22);
}
.primary-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 26px rgba(245, 158, 11, .30);
}
.secondary-button:hover,
.candidate-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(15, 23, 42, .12);
}
.danger-text { color: #dc2626; }
.success-text { color: #16a34a; }
.candidate-card {
    transition: transform .16s ease, box-shadow .16s ease;
    border: 1px solid #e2e8f0;
}
.q-field__control,
.q-slider__track-container,
.q-checkbox__inner {
    color: #f59e0b !important;
}
</style>
""")


with ui.column().classes("dashboard-bg w-full p-5 gap-5"):
    with ui.card().classes("hero-panel w-full p-5"):
        with ui.row().classes("w-full items-center justify-between gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("Smart City LPR Console").classes("text-3xl font-black text-slate-900 leading-tight")
                ui.label("Traffic control dashboard for image-based license plate recognition").classes("text-slate-600 text-base")
            with ui.row().classes("items-center gap-3"):
                detection_spinner = ui.spinner("dots", size="sm", color="orange")
                detection_spinner.set_visibility(False)
                status_badge = ui.label("Ready").classes("status-badge")
        status_label = ui.label("Upload an image to begin").classes("text-slate-500 text-sm")

    with ui.row().classes("dashboard-main w-full gap-5 items-start"):
        with ui.card().classes("glass-card w-72 p-5 gap-4 scroll-panel"):
            ui.label("Controls").classes("text-xl font-bold text-slate-900")
            upload = ui.upload(auto_upload=True).props("accept=image/* label='Upload image'").classes("compact-upload w-full")

            vehicle_select = ui.select(
                ["Auto-detect"] + VEHICLE_TYPES,
                value="Auto-detect",
                label="Vehicle type",
            ).classes("w-full")
            min_area_slider = ui.slider(min=0.1, max=2.0, step=0.1, value=0.3).props("label-always color=orange")
            ui.label("Minimum plate area (%)").classes("muted text-xs")
            show_steps = ui.checkbox("Show preprocessing strip", value=False).classes("text-slate-700")

            ui.element("div").classes("control-spacer")
            ui.label("Detection progress").classes("muted text-xs")
            detection_progress = ui.linear_progress(value=0).props("color=orange rounded").classes("w-full")
            detection_progress.set_visibility(False)
            progress_step_label = ui.label("Idle").classes("text-sm text-slate-600")

            run_button = ui.button("Run Detection", icon="search").props("dense unelevated").classes("primary-button w-full")
            clear_button = ui.button("Clear", icon="refresh").props("outline color=grey-7 dense").classes("secondary-button w-full")
            download_button = ui.button("Download result", icon="download").props("outline color=green dense").classes("secondary-button w-full")
            download_button.disable()

        with ui.card().classes("glass-card flex-1 p-5 gap-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Image Preview").classes("text-xl font-bold text-slate-900")
                file_label = ui.label("No image uploaded").classes("text-slate-500")
            with ui.element("div").classes("preview-frame rounded-xl w-full flex items-center justify-center p-4"):
                preview_image = ui.image().props("fit=contain").classes("w-full h-full")
                preview_image.set_visibility(False)
                empty_preview = ui.label("Upload an image to begin").classes("text-slate-400 text-lg")

            ui.label("Candidate plate crops").classes("text-base font-bold text-slate-900")
            candidate_row = ui.row().classes("crop-strip w-full gap-3")

        with ui.card().classes("glass-card w-80 p-5 gap-4 scroll-panel"):
            ui.label("Detection Results").classes("text-xl font-bold text-slate-900")

            def result_block(title: str, initial: str):
                with ui.column().classes("gap-1"):
                    ui.label(title).classes("muted text-sm")
                    value = ui.label(initial).classes("result-value text-lg")
                return value

            plate_value = result_block("Plate text", "Not detected")
            state_value = result_block("Registered state", "-")
            vehicle_value = result_block("Vehicle type", "-")
            prefix_value = result_block("Prefix", "-")
            count_value = result_block("Plates found", "-")

            ui.separator().classes("bg-slate-200")
            ui.label("All candidates").classes("muted text-sm")
            candidates_list = ui.column().classes("results-list w-full gap-2")


def reset_results():
    plate_value.set_text("Not detected")
    state_value.set_text("-")
    vehicle_value.set_text("-")
    prefix_value.set_text("-")
    count_value.set_text("-")
    candidates_list.clear()
    candidate_row.clear()
    download_button.disable()
    detection_progress.set_visibility(False)
    detection_progress.set_value(0)
    progress_step_label.set_text("Idle")


def set_detection_running(is_running: bool):
    if is_running:
        run_button.disable()
        clear_button.disable()
        download_button.disable()
        upload.disable()
        detection_spinner.set_visibility(True)
        detection_progress.set_visibility(True)
        status_badge.set_text("Processing")
        return

    run_button.enable()
    clear_button.enable()
    upload.enable()
    detection_spinner.set_visibility(False)
    if state.get("processed_image") is not None:
        download_button.enable()


async def update_detection_step(message: str, progress: float):
    status_label.set_text(message)
    progress_step_label.set_text(message)
    detection_progress.set_value(progress)
    await asyncio.sleep(0.25)


async def handle_upload(event: events.UploadEventArguments):
    data = await event.file.read()
    image = decode_upload(data)
    if image is None:
        ui.notify("Could not read uploaded image", type="negative")
        return

    state["image"] = image
    state["processed_image"] = None
    state["uploaded_name"] = event.file.name
    state["result"] = None

    preview_image.set_source(bytes_to_data_url(data, event.file.content_type))
    preview_image.set_visibility(True)
    empty_preview.set_visibility(False)
    file_label.set_text(event.file.name)
    status_label.set_text("Image ready for detection")
    status_badge.set_text("Ready")
    reset_results()


async def run_detection_handler():
    image = state.get("image")
    if image is None:
        ui.notify("Upload an image first", type="warning")
        return

    set_detection_running(True)
    options = {
        "min_area": float(min_area_slider.value) / 100.0,
        "vehicle_type": vehicle_select.value,
        "show_steps": show_steps.value,
    }

    try:
        await update_detection_step("Preparing image...", 0.18)
        await update_detection_step("Running plate detection...", 0.42)
        result = await asyncio.to_thread(detect_plate, image.copy(), options)
        await update_detection_step("Reading plate text...", 0.76)
        await update_detection_step("Finalizing results...", 0.94)
    except Exception as exc:
        status_label.set_text("Detection failed")
        status_badge.set_text("Error")
        ui.notify(f"Detection error: {exc}", type="negative")
        detection_progress.set_visibility(False)
        detection_progress.set_value(0)
        progress_step_label.set_text("Detection failed")
        set_detection_running(False)
        return

    state["result"] = result
    state["processed_image"] = result["processed_image"]
    preview_image.set_source(image_to_data_url(result["processed_image"]))
    preview_image.set_visibility(True)
    empty_preview.set_visibility(False)

    plate_value.set_text(result["plate_text"] or "Not detected")
    state_value.set_text(result["registered_state"])
    vehicle_value.set_text(result["vehicle_type"])
    prefix_value.set_text(result["prefix"] or "-")
    count_value.set_text(str(result["plates_found"]))

    candidates_list.clear()
    for candidate in result["candidates"]:
        with candidates_list:
            ui.label(f"{candidate['text']} -> {candidate['state']}").classes(
                "candidate-card w-full rounded-lg bg-amber-50 px-3 py-2 text-slate-900"
            )

    candidate_row.clear()
    for candidate in result["candidate_images"]:
        with candidate_row:
            with ui.card().classes("candidate-card p-2 w-44"):
                ui.image(image_to_data_url(candidate["image"])).props("fit=contain").classes("w-full h-24 bg-slate-100 rounded")
                ui.label(candidate["text"]).classes("text-slate-900 font-bold")
                ui.label(candidate["state"]).classes("text-slate-500 text-xs")

    if result["processed_image"] is not None:
        download_button.enable()
    detection_progress.set_value(1)
    status_label.set_text("Detection complete")
    status_badge.set_text("Complete")
    progress_step_label.set_text("Complete")
    set_detection_running(False)


async def handle_download():
    image = state.get("processed_image")
    if image is None:
        ui.notify("No processed image to download", type="warning")
        return
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        ui.notify("Could not encode result image", type="negative")
        return
    ui.download(encoded.tobytes(), filename="lpr-result.png")


def handle_clear():
    state["image"] = None
    state["processed_image"] = None
    state["uploaded_name"] = ""
    state["result"] = None
    preview_image.set_source("")
    preview_image.set_visibility(False)
    empty_preview.set_visibility(True)
    file_label.set_text("No image uploaded")
    status_label.set_text("Upload an image to begin")
    status_badge.set_text("Ready")
    upload.reset()
    reset_results()


upload.on_upload(handle_upload)
run_button.on_click(run_detection_handler)
clear_button.on_click(handle_clear)
download_button.on_click(handle_download)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="LPR NiceGUI Dashboard", host="127.0.0.1", port=18080, reload=False, show=False)
