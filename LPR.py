"""
License Plate Recognition (LPR) & State Identification System (SIS)
Malaysia - APU IPPR Group Assignment
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

#import to skip SSL certificate checking 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#  Malaysian plate prefix → state mapping

STATE_MAP = {
    # Selangor
    "B": "Selangor",
    # Kuala Lumpur
    "W": "Kuala Lumpur", "V": "Kuala Lumpur",
    # Johor
    "J": "Johor",
    # Kedah
    "K": "Kedah",
    # Kelantan
    "D": "Kelantan",
    # Malacca
    "M": "Malacca",
    # Negeri Sembilan
    "N": "Negeri Sembilan",
    # Pahang
    "C": "Pahang",
    # Penang
    "P": "Penang",
    # Perak
    "A": "Perak",
    # Perlis
    "R": "Perlis",
    # Sabah
    "S": "Sabah", "SA": "Sabah", "SB": "Sabah", "SC": "Sabah",
    "SD": "Sabah", "SE": "Sabah", "SF": "Sabah", "SG": "Sabah",
    "SH": "Sabah", "SK": "Sabah", "SL": "Sabah",
    # Sarawak
    "Q": "Sarawak", "QA": "Sarawak", "QB": "Sarawak", "QC": "Sarawak",
    "QD": "Sarawak", "QE": "Sarawak", "QF": "Sarawak",
    # Terengganu
    "T": "Terengganu",
    # Federal Territory
    "F": "Putrajaya",
    # Labuan
    "L": "Labuan",
    # Special / Military
    "Z": "Military", "AM": "Military",

}

VEHICLE_TYPES = ["Car", "Motorcycle", "Bus", "Truck", "Van"]



#  Image processing functions

# runs the loaded image 
def preprocess(img):
    """Full preprocessing pipeline."""
    #resizes the image into a 1280 then puts grey scale filter
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


def detect_plates(resized, closed):
    """Find candidate plate regions via contour analysis."""
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h, w = resized.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(ch) if ch > 0 else 0
        area_ratio = (cw * ch) / (w * h)
        # Malaysian plates: aspect ~2–6, reasonable size
        if 1.5 < aspect < 7.0 and 0.003 < area_ratio < 0.15:
            candidates.append((x, y, cw, ch))
    # Remove overlapping boxes to keep largest area
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
    return filtered[:8]  # top 8 candidates


def extract_plate_text(img_bgr, box, reader):
    """Crop, deskew, binarise, and OCR a single candidate box."""
    x, y, w, h = box
    pad = 4
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(img_bgr.shape[1] - x, w + 2 * pad)
    h = min(img_bgr.shape[0] - y, h + 2 * pad)
    crop = img_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    # Upscale small crops for better OCR
    if crop.shape[1] < 200:
        scale = 200 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = reader.readtext(binary, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return " ".join(results).upper().replace(" ", "")


def identify_state(plate_text):
    """Match longest prefix first for accuracy."""
    plate_text = re.sub(r"[^A-Z]", "", plate_text.upper())
    for length in [3, 2, 1]:
        prefix = plate_text[:length]
        if prefix in STATE_MAP:
            return STATE_MAP[prefix], prefix
    return "Unknown", ""


def classify_vehicle(boxes, img_shape):
    """Simple heuristic: large images with wide low plate → bus/truck."""
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



#  GUI application

class LPRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("License Plate Recognition & State Identification System")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self.minsize(1100, 700)

        # State
        self.orig_img = None
        self.result_img = None
        self.reader = None
        self.reader_loaded = False
        self._build_ui()
        self._load_ocr_async()

    # ── UI construction ──────────────────────
    def _build_ui(self):
        # Top bar
        topbar = tk.Frame(self, bg="#11111b", pady=10)
        topbar.pack(fill="x")
        tk.Label(topbar, text="LPR & State Identification System",
                 font=("Segoe UI", 16, "bold"), fg="#cdd6f4", bg="#11111b").pack(side="left", padx=20)
        self.status_lbl = tk.Label(topbar, text="Loading OCR engine...",
                                   font=("Segoe UI", 10), fg="#a6adc8", bg="#11111b")
        self.status_lbl.pack(side="right", padx=20)

        # Main content
        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # Left: controls
        left = tk.Frame(main, bg="#181825", width=240)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._build_controls(left)

        # Centre: image canvas
        centre = tk.Frame(main, bg="#1e1e2e")
        centre.pack(side="left", fill="both", expand=True)
        self._build_canvas(centre)

        # Right: results panel
        right = tk.Frame(main, bg="#181825", width=280)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_results(right)

    def _build_controls(self, parent):
        tk.Label(parent, text="Controls", font=("Segoe UI", 12, "bold"),
                 fg="#cdd6f4", bg="#181825").pack(pady=(16, 8))

        btn_style = {"font": ("Segoe UI", 10), "relief": "flat",
                     "cursor": "hand2", "pady": 8}

        self.load_btn = tk.Button(parent, text="Load Image", bg="#89b4fa", fg="#1e1e2e",
                                  command=self.load_image, **btn_style)
        self.load_btn.pack(fill="x", padx=16, pady=4)

        self.run_btn = tk.Button(parent, text="Run Detection", bg="#a6e3a1", fg="#1e1e2e",
                                 command=self.run_detection, state="disabled", **btn_style)
        self.run_btn.pack(fill="x", padx=16, pady=4)

        tk.Button(parent, text="Clear", bg="#45475a", fg="#cdd6f4",
                  command=self.clear_all, **btn_style).pack(fill="x", padx=16, pady=4)

        # Vehicle type selector
        tk.Label(parent, text="Vehicle type", font=("Segoe UI", 9),
                 fg="#a6adc8", bg="#181825").pack(pady=(20, 4))
        self.vehicle_var = tk.StringVar(value="Auto-detect")
        combo = ttk.Combobox(parent, textvariable=self.vehicle_var,
                             values=["Auto-detect"] + VEHICLE_TYPES, state="readonly")
        combo.pack(fill="x", padx=16)

        # Preprocessing steps toggle
        tk.Label(parent, text="Show preprocessing steps",
                 font=("Segoe UI", 9), fg="#a6adc8", bg="#181825").pack(pady=(20, 4))
        self.show_steps_var = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Enable", variable=self.show_steps_var,
                       bg="#181825", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#181825", font=("Segoe UI", 9)).pack(padx=16, anchor="w")

        # Confidence threshold
        tk.Label(parent, text="Min plate area (%)",
                 font=("Segoe UI", 9), fg="#a6adc8", bg="#181825").pack(pady=(16, 4))
        self.area_var = tk.DoubleVar(value=0.3)
        tk.Scale(parent, from_=0.1, to=2.0, resolution=0.1,
                 variable=self.area_var, orient="horizontal",
                 bg="#181825", fg="#cdd6f4", troughcolor="#313244",
                 highlightthickness=0).pack(fill="x", padx=16)

    def _build_canvas(self, parent):
        tk.Label(parent, text="Image preview", font=("Segoe UI", 10),
                 fg="#a6adc8", bg="#1e1e2e").pack(anchor="w", pady=(0, 4))
        canvas_frame = tk.Frame(parent, bg="#11111b", relief="flat", bd=1)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#11111b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas())
        # Drop hint
        self.canvas_hint = self.canvas.create_text(
            400, 250, text="Load an image to begin",
            font=("Segoe UI", 14), fill="#45475a")

    def _build_results(self, parent):
        tk.Label(parent, text="Detection results", font=("Segoe UI", 12, "bold"),
                 fg="#cdd6f4", bg="#181825").pack(pady=(16, 8), padx=16, anchor="w")

        # Plate text
        self._result_row(parent, "Plate text")
        self.plate_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self.plate_var,
                 font=("Courier New", 20, "bold"), fg="#f38ba8",
                 bg="#181825").pack(padx=16, anchor="w")

        # State
        self._result_row(parent, "Registered state")
        self.state_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self.state_var,
                 font=("Segoe UI", 14, "bold"), fg="#a6e3a1",
                 bg="#181825").pack(padx=16, anchor="w")

        # Prefix
        self._result_row(parent, "Prefix detected")
        self.prefix_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self.prefix_var,
                 font=("Segoe UI", 11), fg="#89dceb",
                 bg="#181825").pack(padx=16, anchor="w")

        # Vehicle type
        self._result_row(parent, "Vehicle type")
        self.vtype_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self.vtype_var,
                 font=("Segoe UI", 11), fg="#fab387",
                 bg="#181825").pack(padx=16, anchor="w")

        # Plates found
        self._result_row(parent, "Plates found")
        self.count_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self.count_var,
                 font=("Segoe UI", 11), fg="#cdd6f4",
                 bg="#181825").pack(padx=16, anchor="w")

        # All candidates listbox
        self._result_row(parent, "All candidates")
        self.cand_list = tk.Listbox(parent, bg="#313244", fg="#cdd6f4",
                                    font=("Courier New", 10), height=6,
                                    selectbackground="#585b70", relief="flat",
                                    highlightthickness=0)
        self.cand_list.pack(fill="x", padx=16, pady=(4, 0))

        # Save button
        tk.Button(parent, text="Save result image", font=("Segoe UI", 9),
                  bg="#cba6f7", fg="#1e1e2e", relief="flat", cursor="hand2",
                  command=self.save_result).pack(fill="x", padx=16, pady=16)

    def _result_row(self, parent, label):
        tk.Label(parent, text=label, font=("Segoe UI", 8),
                 fg="#6c7086", bg="#181825").pack(padx=16, anchor="w", pady=(12, 0))

    # ── OCR loading ──────────────────────────
    def _load_ocr_async(self):
        def load():
            try:
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.reader_loaded = True
                self.status_lbl.config(text="Ready", fg="#a6e3a1")
                if self.orig_img is not None:
                    self.run_btn.config(state="normal")
            except Exception as e:
                self.status_lbl.config(text=f"OCR error: {e}", fg="#f38ba8")
        threading.Thread(target=load, daemon=True).start()

    # ── Image load / display ─────────────────
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp")])
        if not path:
            return
        self.orig_img = cv2.imread(path)
        if self.orig_img is None:
            self.status_lbl.config(text="Failed to load image", fg="#f38ba8")
            return
        self.result_img = self.orig_img.copy()
        self._display(self.orig_img)
        self.status_lbl.config(text=f"Loaded: {os.path.basename(path)}", fg="#a6adc8")
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

    # ── Detection pipeline ───────────────────
    def run_detection(self):
        if self.orig_img is None or not self.reader_loaded:
            return
        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Processing...", fg="#f9e2af")
        threading.Thread(target=self._detect_thread, daemon=True).start()

    def _detect_thread(self):
        try:
            img = self.orig_img.copy()
            resized, gray, enhanced, edges, closed = preprocess(img)

            # Override area threshold from slider
            min_area = self.area_var.get() / 100.0
            boxes = detect_plates(resized, closed)

            draw = resized.copy()
            candidates = []
            best_text = ""
            best_state = "Unknown"
            best_prefix = ""

            for i, box in enumerate(boxes):
                text = extract_plate_text(resized, box, self.reader)
                if len(text) < 3:
                    continue
                state, prefix = identify_state(text)
                candidates.append((text, state))
                x, y, w, h = box
                color = (0, 255, 100) if state != "Unknown" else (0, 180, 255)
                cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)
                label = f"{text} | {state}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(draw, (x, y - lh - 10), (x + lw + 8, y), color, -1)
                cv2.putText(draw, label, (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                if not best_text and state != "Unknown":
                    best_text, best_state, best_prefix = text, state, prefix

            if not best_text and candidates:
                best_text, best_state = candidates[0]
                _, best_prefix = identify_state(best_text)

            vtype = self.vehicle_var.get()
            if vtype == "Auto-detect":
                vtype = classify_vehicle(boxes, resized.shape)

            # Show preprocessing steps if toggled
            if self.show_steps_var.get():
                steps = [
                    ("Grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
                    ("CLAHE enhanced", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)),
                    ("Canny edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
                ]
                th = 120
                strip_imgs = []
                for name, simg in steps:
                    s = cv2.resize(simg, (int(simg.shape[1] * th / simg.shape[0]), th))
                    cv2.putText(s, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 220, 180), 1)
                    strip_imgs.append(s)
                max_w = min(draw.shape[1], sum(s.shape[1] for s in strip_imgs) + 10)
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
                text=f"Error: {e}", fg="#f38ba8"))
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
        self.status_lbl.config(text="Detection complete", fg="#a6e3a1")
        self.run_btn.config(state="normal")

    # ── Utilities ────────────────────────────
    def _reset_results(self):
        for var in [self.plate_var, self.state_var, self.prefix_var,
                    self.vtype_var, self.count_var]:
            var.set("—")
        self.cand_list.delete(0, "end")

    def clear_all(self):
        self.orig_img = None
        self.result_img = None
        if hasattr(self, "_current_img_bgr"):
            self._current_img_bgr = None
        self.canvas.delete("all")
        self.canvas.create_text(400, 250, text="Load an image to begin",
                                font=("Segoe UI", 14), fill="#45475a")
        self._reset_results()
        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Ready", fg="#a6e3a1")

    def save_result(self):
        if self.result_img is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.result_img)
            self.status_lbl.config(text=f"Saved: {os.path.basename(path)}", fg="#a6e3a1")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = LPRApp()
    app.mainloop()