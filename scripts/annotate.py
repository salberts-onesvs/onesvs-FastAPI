"""
Annotation Tool — Visual ML Workflow Builder
Loads images and COCO JSON, draw bounding boxes, auto-saves progress.

Usage:
    python scripts/annotate.py --project svs_plumbing
    python scripts/annotate.py --images /path/to/images --json /path/to/labels.json
    python scripts/annotate.py --url https://example.com/photo.jpg --json /path/to/labels.json
    python scripts/annotate.py --url https://a.com/1.jpg --url https://b.com/2.jpg --json /path/to/labels.json
"""

import argparse
import io
import json
import os
import sys
import threading
import tkinter as tk
import urllib.request
import tempfile

# DB sync (non-blocking background updates to ml_workspace.db)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from db import client as _db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
from tkinter import ttk, messagebox, colorchooser
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
import numpy as np

# SAM lazy-loaded on first use
_sam_predictor = None
_sam_embedding_lock = threading.Lock()   # guards set_image calls
_MODEL_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Annotator", "Model")
SAM_MODEL_PATH = os.path.join(_MODEL_DIR, "mobile_sam.pt")   # MobileSAM (~40 MB, ~10x faster)
_SAM_FALLBACK  = os.path.join(_MODEL_DIR, "sam_vit_b.pth")   # fallback if MobileSAM missing

# Known best.pt locations (checked in order)
_BEST_PT_SEARCH = [
    os.path.join(_MODEL_DIR, "best.pt"),
    os.path.join(_MODEL_DIR, "pii_v4_best.pt"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt"),
]

def _find_best_pt():
    for p in _BEST_PT_SEARCH:
        if os.path.exists(p):
            return p
    return None


def _get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    try:
        if os.path.exists(SAM_MODEL_PATH):
            # MobileSAM — fast CPU inference
            from mobile_sam import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_t"](checkpoint=SAM_MODEL_PATH)
        elif os.path.exists(_SAM_FALLBACK):
            # Fall back to original SAM ViT-B
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_b"](checkpoint=_SAM_FALLBACK)
        else:
            return None
        sam.to("cpu")
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f"[SAM] Failed to load: {e}")
        _sam_predictor = None
    return _sam_predictor


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#8BC34A", "#FF5722", "#607D8B", "#795548", "#FFC107",
    "#03A9F4", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
]


# ── Main App ──────────────────────────────────────────────────────────────────
class AnnotationTool:
    def __init__(self, root, images_dir, json_path, class_names, project=None,
                 url_images=None, url_map=None, lazy_url_map=None):
        """
        url_images    : dict {filename: PIL.Image} — pre-fetched images keyed by filename.
        url_map       : dict {filename: source_url} — original cloud URL for each filename.
        lazy_url_map  : dict {filename: url} — like url_map but images are fetched on demand.
                        Used by --index mode to avoid pre-loading hundreds of images.
        """
        self.root = root
        self.root.title("Annotation Tool — Visual ML Workflow Builder")
        self.root.configure(bg="#1a1a2e")
        self._project = project  # used by DB sync

        self.images_dir = images_dir
        self.json_path = json_path
        self.class_names = class_names
        self.color_map = {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                          for i, name in enumerate(class_names)}

        # URL mode — images live in memory, not on disk
        self.url_images   = url_images   or {}  # {filename: PIL.Image}  pre-fetched
        self.url_map      = url_map      or {}  # {filename: source_url}
        self.lazy_url_map = lazy_url_map or {}  # {filename: url}  fetched on demand

        # State
        if self.url_images or self.lazy_url_map:
            # URL mode — use pre-fetched or lazy URL list
            self.image_files = sorted(
                self.url_images.keys() if self.url_images else self.lazy_url_map.keys()
            )
        else:
            self.image_files = sorted([
                f for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        self.current_idx = 0
        self.selected_class = tk.StringVar(value=class_names[0] if class_names else "")
        self.active_class = class_names[0] if class_names else "object"  # sticky active class
        self.selected_ann_idx = None
        self.draw_start = None
        self.preview_rect = None
        self.tk_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        # Drawing mode: "bbox" or "polygon"
        self.draw_mode = tk.StringVar(value="bbox")
        # Polygon in-progress state
        self.poly_points = []       # canvas coords being placed
        self.poly_preview_items = []  # canvas item ids for in-progress poly
        # Polygon edit state
        self.edit_ann_id = None       # annotation id being edited
        self.drag_vertex_idx = None   # index of vertex being dragged
        self.drag_start = None        # canvas coord at drag start
        self.ghost_dot = None         # canvas item id for edge suggestion dot
        self.hull_ann_ids = []        # annotation ids collected for union merge

        # SAM state
        self.sam_embedded = False     # whether current image is embedded in SAM
        self.sam_embedding_img = None # which image is currently embedded
        self.sam_points = []          # list of [img_x, img_y, label] (1=fg, 0=bg)
        self.sam_preview_items = []   # canvas item ids for preview overlay
        self.sam_session_active = False

        # Annotation overlay toggle
        self.show_annotations = True

        # Flagged images set
        self.flagged_images = set()

        # Load or init COCO JSON
        self.coco = self._load_or_init_coco()
        self._rebuild_lookup()

        self._build_ui()
        self._load_image(0)
        self._bind_keys()
        self.root.after(100, self._refresh_image_browser)

        # Warm up SAM in background so first "Tighten" is fast
        threading.Thread(target=self._sam_warmup, daemon=True).start()

    # ── COCO I/O ──────────────────────────────────────────────────────────────

    def _load_or_init_coco(self):
        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                coco = json.load(f)
            # Ensure categories match class_names
            existing = {c["name"] for c in coco.get("categories", [])}
            max_id = max((c["id"] for c in coco["categories"]), default=0)
            for name in self.class_names:
                if name not in existing:
                    max_id += 1
                    coco["categories"].append({"id": max_id, "name": name})
            return coco
        else:
            return {
                "info": {"description": "Annotation Tool — Visual ML Workflow Builder"},
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": i + 1, "name": name}
                    for i, name in enumerate(self.class_names)
                ],
            }

    def _save(self):
        self._rebuild_lookup()
        self._set_status(f"Saved → {os.path.basename(self.json_path)}")
        self._sync_db()
        payload = json.dumps(self.coco)
        path    = self.json_path
        threading.Thread(target=lambda: open(path, "w").write(payload), daemon=True).start()

    def _autosave(self, status_msg=""):
        self._rebuild_lookup()
        if status_msg:
            self._set_status(f"{status_msg}  ✓ saved")
        self._sync_db()
        payload = json.dumps(self.coco)
        path    = self.json_path
        threading.Thread(target=lambda: open(path, "w").write(payload), daemon=True).start()

    def _sync_db(self):
        """Push annotation stats to ml_workspace.db in the background (non-blocking)."""
        if not _DB_AVAILABLE or not self._project:
            return
        project = self._project
        json_path = self.json_path
        threading.Thread(
            target=_db.sync_annotations_from_coco,
            args=(project, json_path),
            daemon=True,
        ).start()

    def _rebuild_lookup(self):
        """Build O(1) lookup dicts from coco lists. Call after any structural change."""
        self._img_by_fname = {img["file_name"]: img for img in self.coco["images"]}
        self._ann_by_img_id = {}
        for a in self.coco["annotations"]:
            self._ann_by_img_id.setdefault(a["image_id"], []).append(a)

    def _get_image_record(self, filename):
        return self._img_by_fname.get(filename)

    def _ensure_image_record(self, filename, w, h):
        rec = self._img_by_fname.get(filename)
        if rec is None:
            new_id = max(self._img_by_fname[f]["id"] for f in self._img_by_fname) + 1 if self._img_by_fname else 1
            rec = {"id": new_id, "file_name": filename, "width": w, "height": h}
            if filename in self.url_map:
                rec["url"] = self.url_map[filename]
            self.coco["images"].append(rec)
            self._img_by_fname[filename] = rec
            self._ann_by_img_id.setdefault(new_id, [])
        return rec

    def _get_annotations(self, image_id):
        return self._ann_by_img_id.get(image_id, [])

    def _next_ann_id(self):
        if not self.coco["annotations"]:
            return 1
        return max(a["id"] for a in self.coco["annotations"]) + 1

    def _cat_id(self, name):
        for c in self.coco["categories"]:
            if c["name"] == name:
                return c["id"]
        return 1

    def _cat_name(self, cat_id):
        for c in self.coco["categories"]:
            if c["id"] == cat_id:
                return c["name"]
        return "unknown"

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, minsize=180)
        self.root.rowconfigure(0, weight=1)

        # Left panel
        left = tk.Frame(self.root, bg="#16213e", width=180)
        left.grid(row=0, column=0, sticky="ns", padx=(8, 0), pady=8)
        left.grid_propagate(False)

        # Draw mode toggle
        mode_frame = tk.Frame(left, bg="#16213e")
        mode_frame.pack(fill=tk.X, padx=6, pady=(10, 4))
        self.bbox_btn = tk.Radiobutton(
            mode_frame, text="BBox", variable=self.draw_mode, value="bbox",
            bg="#16213e", fg="white", selectcolor="#0f3460",
            activebackground="#16213e", font=("Helvetica", 9, "bold"),
            command=self._on_mode_change
        )
        self.bbox_btn.pack(side=tk.LEFT, expand=True)
        self.poly_btn = tk.Radiobutton(
            mode_frame, text="Polygon", variable=self.draw_mode, value="polygon",
            bg="#16213e", fg="white", selectcolor="#0f3460",
            activebackground="#16213e", font=("Helvetica", 9, "bold"),
            command=self._on_mode_change
        )
        self.poly_btn.pack(side=tk.LEFT, expand=True)

        # SAM Box mode row
        sam_frame = tk.Frame(left, bg="#16213e")
        sam_frame.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Radiobutton(
            sam_frame, text="SAM Box", variable=self.draw_mode, value="sambox",
            bg="#16213e", fg="#00d4ff", selectcolor="#0f3460",
            activebackground="#16213e", font=("Helvetica", 9, "bold"),
            command=self._on_mode_change
        ).pack(side=tk.LEFT, expand=True)

        tk.Label(left, text="Classes", bg="#16213e", fg="#00d4ff",
                 font=("Helvetica", 11, "bold")).pack(pady=(6, 2))

        self.class_listbox = tk.Listbox(
            left, selectmode=tk.SINGLE, bg="#0f3460", fg="white",
            selectbackground="#00d4ff", selectforeground="#000",
            font=("Helvetica", 10), relief="flat", bd=0,
            highlightthickness=0, exportselection=0
        )
        for name in self.class_names:
            self.class_listbox.insert(tk.END, name)
        self.class_listbox.select_set(0)
        self.class_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.class_listbox.bind("<<ListboxSelect>>", self._on_class_select)
        self.class_listbox.bind("<Double-Button-1>", lambda e: self._change_class_color())

        # Add / remove / color class buttons
        cls_btn_frame = tk.Frame(left, bg="#16213e")
        cls_btn_frame.pack(fill=tk.X, padx=6, pady=(0, 2))
        tk.Button(cls_btn_frame, text="+ Class", command=self._add_class,
                  bg="#2ecc71", fg="#000", relief="flat",
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.remove_class_btn = tk.Button(
            cls_btn_frame, text="− Class", command=self._remove_class,
            bg="#e67e22", fg="white", relief="flat",
            font=("Helvetica", 8, "bold"), cursor="hand2"
        )
        self.remove_class_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.color_btn = tk.Button(
            left, text="⬤  Change Color", command=self._change_class_color,
            bg="#2c2c54", fg="white", relief="flat",
            font=("Helvetica", 8), cursor="hand2"
        )
        self.color_btn.pack(fill=tk.X, padx=6, pady=(0, 6))
        self._update_color_btn()

        # Brightness slider
        tk.Label(left, text="Brightness", bg="#16213e", fg="#aaa",
                 font=("Helvetica", 8)).pack(pady=(6, 0))
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.brightness_slider = tk.Scale(
            left, from_=0.2, to=3.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.brightness_var,
            bg="#16213e", fg="white", troughcolor="#0f3460",
            highlightthickness=0, showvalue=True,
            font=("Helvetica", 8), length=160,
            command=self._on_brightness_change
        )
        self.brightness_slider.pack(padx=6, pady=(0, 2))
        tk.Button(
            left, text="Reset", command=self._reset_brightness,
            bg="#2c2c54", fg="#aaa", relief="flat",
            font=("Helvetica", 8), cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        # Annotation list
        tk.Label(left, text="Annotations", bg="#16213e", fg="#00d4ff",
                 font=("Helvetica", 11, "bold")).pack(pady=(10, 4))

        self.ann_listbox = tk.Listbox(
            left, selectmode=tk.SINGLE, bg="#0f3460", fg="white",
            selectbackground="#e74c3c", selectforeground="white",
            font=("Helvetica", 9), relief="flat", bd=0,
            highlightthickness=0, height=10, exportselection=0
        )
        self.ann_listbox.pack(fill=tk.X, padx=6, pady=4)
        self.ann_listbox.bind("<<ListboxSelect>>", self._on_ann_select)
        self.ann_listbox.bind("<Delete>", lambda e: self._delete_selected())
        self.ann_listbox.bind("<BackSpace>", lambda e: self._delete_selected())

        tk.Button(
            left, text="Edit Polygon", command=self._enter_edit_mode,
            bg="#1abc9c", fg="white", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 2))
        tk.Button(
            left, text="✦  Tighten with SAM", command=self._tighten_selected,
            bg="#00838f", fg="white", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 2))
        self.relabel_btn = tk.Button(
            left, text="Relabel Selected", command=self._relabel_selected,
            bg="#9b59b6", fg="white", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        )
        self.relabel_btn.pack(fill=tk.X, padx=6, pady=(2, 2))
        tk.Button(
            left, text="Delete Selected", command=self._delete_selected,
            bg="#e74c3c", fg="white", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 2))
        self.hull_add_btn = tk.Button(
            left, text="⬡  Add to Union (0)", command=self._hull_add,
            bg="#2c3e50", fg="#1abc9c", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        )
        self.hull_add_btn.pack(fill=tk.X, padx=6, pady=(2, 2))
        self.hull_merge_btn = tk.Button(
            left, text="⬡  Merge Union", command=self._hull_merge,
            bg="#2c3e50", fg="#888", relief="flat", font=("Helvetica", 9),
            cursor="hand2", state=tk.DISABLED
        )
        self.hull_merge_btn.pack(fill=tk.X, padx=6, pady=(2, 10))
        tk.Button(
            left, text="🤖  Pre-annotate (YOLO bbox)", command=self._pre_annotate,
            bg="#1a3a2a", fg="#2ecc71", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 2))
        tk.Button(
            left, text="🤖✦ Pre-annotate (YOLO+SAM)", command=self._pre_annotate_sam,
            bg="#0d2a1a", fg="#1abc9c", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 2))
        tk.Button(
            left, text="⚡ Batch Pre-annotate All", command=self._batch_pre_annotate,
            bg="#1a2a0a", fg="#f1c40f", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 2))
        tk.Button(
            left, text="📷  Export Annotated Image", command=self._export_annotated_image,
            bg="#2c3e50", fg="#f1c40f", relief="flat", font=("Helvetica", 9),
            cursor="hand2"
        ).pack(fill=tk.X, padx=6, pady=(2, 8))

        # Right panel — image browser
        right = tk.Frame(self.root, bg="#16213e", width=180)
        right.grid(row=0, column=2, sticky="ns", padx=(0, 8), pady=8)
        right.grid_propagate(False)

        tk.Label(right, text="Images", bg="#16213e", fg="#00d4ff",
                 font=("Helvetica", 11, "bold")).pack(pady=(8, 2))

        # Search bar
        self.img_search_var = tk.StringVar()
        search_entry = tk.Entry(right, textvariable=self.img_search_var,
                                bg="#0f3460", fg="white", insertbackground="white",
                                relief="flat", font=("Helvetica", 9))
        search_entry.pack(fill=tk.X, padx=6, pady=(0, 4))
        search_entry.bind("<KeyRelease>", lambda e: self._refresh_image_browser())

        # Filter buttons
        filter_frame = tk.Frame(right, bg="#16213e")
        filter_frame.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.img_filter_var = tk.StringVar(value="all")
        for val, label in [("all", "All"), ("done", "✓"), ("todo", "○"), ("flag", "⚑"), ("model", "🤖")]:
            tk.Radiobutton(
                filter_frame, text=label, variable=self.img_filter_var, value=val,
                bg="#16213e", fg="white", selectcolor="#0f3460",
                activebackground="#16213e", font=("Helvetica", 8),
                command=self._refresh_image_browser
            ).pack(side=tk.LEFT, expand=True)

        # Image list
        img_list_frame = tk.Frame(right, bg="#16213e")
        img_list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        img_scroll = tk.Scrollbar(img_list_frame, orient=tk.VERTICAL)
        self.img_listbox = tk.Listbox(
            img_list_frame, bg="#0f3460", fg="white",
            selectbackground="#00d4ff", selectforeground="#000",
            font=("Helvetica", 8), relief="flat", bd=0,
            highlightthickness=0, exportselection=0,
            yscrollcommand=img_scroll.set
        )
        img_scroll.config(command=self.img_listbox.yview)
        img_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.img_listbox.pack(fill=tk.BOTH, expand=True)
        self.img_listbox.bind("<<ListboxSelect>>", self._on_img_browser_select)
        self._img_browser_indices = []  # maps listbox row → image_files index

        # Center canvas
        center = tk.Frame(self.root, bg="#1a1a2e")
        center.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(center, bg="#0d0d0d", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Double-Button-1>", self._on_poly_double_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)

        # Bottom bar
        bottom = tk.Frame(self.root, bg="#16213e", height=44)
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))

        btn_style = {"bg": "#0f3460", "fg": "white", "relief": "flat",
                     "font": ("Helvetica", 10), "cursor": "hand2",
                     "padx": 16, "pady": 6}

        tk.Button(bottom, text="◀  Prev", command=self._prev_image,
                  **btn_style).pack(side=tk.LEFT, padx=(8, 4), pady=6)
        tk.Button(bottom, text="Next  ▶", command=self._next_image,
                  **btn_style).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="⏭  Skip to Unannotated", command=self._jump_to_unannotated,
                  **btn_style).pack(side=tk.LEFT, padx=4, pady=6)
        self.flag_btn = tk.Button(bottom, text="⚑  Flag", command=self._toggle_flag,
                  bg="#0f3460", fg="white", relief="flat",
                  font=("Helvetica", 10), cursor="hand2",
                  padx=12, pady=6)
        self.flag_btn.pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="🗑  Delete Photo", command=self._delete_current_image,
                  bg="#6b1a1a", fg="white", relief="flat",
                  font=("Helvetica", 10), cursor="hand2",
                  padx=12, pady=6).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="📊  Class Balance", command=self._show_class_balance,
                  **btn_style).pack(side=tk.LEFT, padx=4, pady=6)

        self.progress_label = tk.Label(
            bottom, text="", bg="#16213e", fg="#aaa", font=("Helvetica", 10)
        )
        self.progress_label.pack(side=tk.LEFT, padx=16)

        self.status_label = tk.Label(
            bottom, text="", bg="#16213e", fg="#2ecc71", font=("Helvetica", 10)
        )
        self.status_label.pack(side=tk.RIGHT, padx=16)

    def _on_brightness_change(self, _=None):
        # Reset SAM embedding so next Shift+click uses brightened image
        self.sam_embedded = False
        self.sam_embedding_img = None
        self._render()

    def _reset_brightness(self):
        self.brightness_var.set(1.0)
        self.sam_embedded = False
        self.sam_embedding_img = None
        self._render()

    def _get_display_image(self):
        """Return PIL image with brightness applied (original untouched)."""
        factor = self.brightness_var.get()
        if abs(factor - 1.0) < 0.01:
            return self.pil_image
        from PIL import ImageEnhance
        return ImageEnhance.Brightness(self.pil_image).enhance(factor)

    # ── Annotation overlay toggle ─────────────────────────────────────────────

    def _toggle_annotations(self):
        self.show_annotations = not self.show_annotations
        self._render()
        self._set_status("Annotations hidden (H to show)" if not self.show_annotations
                         else "Annotations visible")

    # ── Jump to unannotated ───────────────────────────────────────────────────

    def _jump_to_unannotated(self):
        start = self.current_idx
        for offset in range(1, len(self.image_files)):
            idx = (start + offset) % len(self.image_files)
            fname = self.image_files[idx]
            rec = self._get_image_record(fname)
            if not rec or not self._get_annotations(rec["id"]):
                self._load_image(idx)
                self._set_status(f"Jumped to unannotated: {fname}")
                return
        self._set_status("All images are annotated!")

    # ── Delete current image ──────────────────────────────────────────────────

    def _delete_current_image(self):
        fname = self.image_files[self.current_idx]
        if not messagebox.askyesno("Delete Photo",
                                   f"Permanently delete '{fname}' from disk and remove its annotations?"):
            return
        # Remove from JSON
        rec = self._get_image_record(fname)
        if rec:
            self.coco["annotations"] = [
                a for a in self.coco["annotations"] if a["image_id"] != rec["id"]
            ]
            self.coco["images"] = [
                i for i in self.coco["images"] if i["id"] != rec["id"]
            ]
        # Move file to recycle folder next to images_dir
        if self.images_dir:
            path = os.path.join(self.images_dir, fname)
            recycle_dir = os.path.join(os.path.dirname(self.images_dir), "recycle")
            os.makedirs(recycle_dir, exist_ok=True)
            try:
                import shutil
                shutil.move(path, os.path.join(recycle_dir, fname))
            except Exception as e:
                print(f"[annotate] Could not move {path} to recycle: {e}")
        # Remove from in-memory lists
        self.url_images.pop(fname, None)
        self.url_map.pop(fname, None)
        self.lazy_url_map.pop(fname, None)
        self.flagged_images.discard(fname)
        self.image_files.pop(self.current_idx)
        self._save()
        if not self.image_files:
            self._set_status("No images remaining.")
            return
        next_idx = min(self.current_idx, len(self.image_files) - 1)
        self._load_image(next_idx)
        self._set_status(f"Deleted {fname}")

    # ── Flag image ────────────────────────────────────────────────────────────

    def _toggle_flag(self):
        fname = self.image_files[self.current_idx]
        if fname in self.flagged_images:
            self.flagged_images.discard(fname)
            self.flag_btn.config(bg="#0f3460", fg="white")
            self._set_status("Flag removed.")
        else:
            self.flagged_images.add(fname)
            self.flag_btn.config(bg="#e74c3c", fg="white")
            self._set_status("Image flagged for review.")

    def _update_flag_btn(self):
        fname = self.image_files[self.current_idx] if self.image_files else ""
        if fname in self.flagged_images:
            self.flag_btn.config(bg="#e74c3c", fg="white")
        else:
            self.flag_btn.config(bg="#0f3460", fg="white")

    # ── Class balance chart ───────────────────────────────────────────────────

    def _show_class_balance(self):
        from collections import Counter
        cat_counts = Counter()
        for ann in self.coco["annotations"]:
            name = self._cat_name(ann["category_id"])
            cat_counts[name] += 1

        win = tk.Toplevel(self.root)
        win.title("Class Balance")
        win.configure(bg="#1a1a2e")
        win.geometry("500x420")
        win.transient(self.root)

        tk.Label(win, text="Annotation Count per Class", bg="#1a1a2e",
                 fg="#00d4ff", font=("Helvetica", 12, "bold")).pack(pady=(12, 6))

        frame = tk.Frame(win, bg="#1a1a2e")
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        sorted_items = sorted(cat_counts.items(), key=lambda x: -x[1])
        max_count = sorted_items[0][1] if sorted_items else 1
        total = sum(cat_counts.values())

        for name, count in sorted_items:
            color = self.color_map.get(name, "#ffffff")
            row = tk.Frame(frame, bg="#1a1a2e")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{name}", bg="#1a1a2e", fg="white",
                     font=("Helvetica", 9), width=22, anchor="w").pack(side=tk.LEFT)
            bar_w = max(4, int(220 * count / max_count))
            tk.Frame(row, bg=color, width=bar_w, height=14).pack(side=tk.LEFT, padx=4)
            tk.Label(row, text=str(count), bg="#1a1a2e", fg="#aaa",
                     font=("Helvetica", 9)).pack(side=tk.LEFT)

        tk.Label(win, text=f"Total: {total} annotations across {len(self.image_files)} images",
                 bg="#1a1a2e", fg="#aaa", font=("Helvetica", 9)).pack(pady=(4, 10))

    def _bind_keys(self):
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Delete>", lambda e: self._delete_selected())
        self.root.bind("<BackSpace>", lambda e: self._delete_selected())
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<h>", lambda e: self._toggle_annotations())
        self.root.bind("<H>", lambda e: self._toggle_annotations())
        self.root.bind("<Return>", lambda e: self._sam_confirm())

    # ── Image Loading ─────────────────────────────────────────────────────────

    def _load_image(self, idx):
        if not self.image_files:
            self._set_status("No images found.")
            return

        self._exit_edit_mode()
        self._cancel_polygon()
        self._sam_cancel()
        self.sam_embedded = False
        self._save()  # auto-save on every image change

        # Cancel any pending union when navigating
        if self.hull_ann_ids:
            self.hull_ann_ids = []
            self._hull_update_ui()

        self.current_idx = idx
        filename = self.image_files[idx]

        if self.url_images and filename in self.url_images:
            # Pre-fetched URL mode
            self.pil_image = self.url_images[filename].copy()
        elif self.lazy_url_map and filename in self.lazy_url_map:
            # Lazy URL mode — fetch in a background thread so the UI stays responsive
            url = self.lazy_url_map[filename]
            # Show a placeholder immediately
            placeholder = Image.new("RGB", (800, 600), color=(30, 30, 50))
            self.pil_image = placeholder
            self._set_status(f"Loading {filename} …")

            def _fetch(fname=filename, u=url, target_idx=idx):
                try:
                    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    img = Image.open(io.BytesIO(data))
                    img = ImageOps.exif_transpose(img).convert("RGB")
                except Exception as e:
                    print(f"[annotate] Failed to fetch {u}: {e}")
                    img = None

                def _apply():
                    # Only update if the user hasn't navigated away
                    if self.current_idx != target_idx:
                        return
                    if img is not None:
                        self.url_images[fname] = img
                        self.url_map[fname]    = u
                        self.pil_image = img.copy()
                        self.img_w, self.img_h = self.pil_image.size
                        self._render()
                        self._set_status("")
                        print(f"[annotate] Loaded {fname} ({img.width}x{img.height})")
                    else:
                        self._set_status(f"Failed to load {fname}")

                try:
                    self.root.after(0, _apply)
                except RuntimeError:
                    pass  # root not yet in main loop — skip UI update

            threading.Thread(target=_fetch, daemon=True).start()
        else:
            path = os.path.join(self.images_dir, filename)
            try:
                self.pil_image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception as e:
                print(f"[annotate] Skipping corrupt image {filename}: {e}")
                self.pil_image = Image.new("RGB", (800, 600), color=(30, 30, 50))
        self.img_w, self.img_h = self.pil_image.size

        self._ensure_image_record(filename, self.img_w, self.img_h)
        self._render()

        annotated = sum(
            1 for f in self.image_files
            if self._get_image_record(f) and
            self._get_annotations(self._get_image_record(f)["id"])
        )
        flagged_marker = "  ⚑" if filename in self.flagged_images else ""
        self.progress_label.config(
            text=f"Image {idx + 1} / {len(self.image_files)}  |  "
                 f"{annotated} annotated  |  {filename}{flagged_marker}"
        )
        self._update_flag_btn()
        self._refresh_ann_list()
        self._refresh_image_browser()
        self._set_status("")

        # Pre-embed current image into SAM in the background
        if _get_sam_predictor() is not None and self.sam_embedding_img != filename:
            img_array = np.array(self._get_display_image())
            threading.Thread(
                target=self._sam_prefetch, args=(filename, img_array), daemon=True
            ).start()

    def _render(self):
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600

        scale_x = cw / self.img_w
        scale_y = ch / self.img_h
        self.scale = min(scale_x, scale_y, 1.0)

        disp_w = int(self.img_w * self.scale)
        disp_h = int(self.img_h * self.scale)
        self.offset_x = (cw - disp_w) // 2
        self.offset_y = (ch - disp_h) // 2

        resized = self._get_display_image().resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y,
                                 anchor=tk.NW, image=self.tk_image)
        if self.show_annotations:
            self._draw_annotations()

    def _img_to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _canvas_to_img(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_annotations(self):
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        for i, ann in enumerate(anns):
            name = self._cat_name(ann["category_id"])
            color = self.color_map.get(name, "#ffffff")
            selected = i == self.selected_ann_idx
            lw = 4 if selected else 2
            fill_stipple = "gray25" if selected else "gray12"
            seg = ann.get("segmentation")

            if seg and any(ring and len(ring) >= 6 for ring in seg):
                label_x, label_y = None, None
                for ring in seg:
                    if not ring or len(ring) < 6:
                        continue
                    canvas_pts = []
                    for j in range(0, len(ring), 2):
                        cx, cy = self._img_to_canvas(ring[j], ring[j + 1])
                        canvas_pts.extend([cx, cy])
                    # Semi-transparent fill layer
                    self.canvas.create_polygon(
                        canvas_pts, outline="", fill=color,
                        stipple=fill_stipple, tags="ann"
                    )
                    # Solid outline on top
                    self.canvas.create_polygon(
                        canvas_pts, outline=color, fill="", width=lw, tags="ann"
                    )
                    if label_x is None:
                        label_x, label_y = canvas_pts[0], canvas_pts[1]
                # Dashed line between ring centroids (multi-ring annotation)
                if len(seg) > 1:
                    centroids = []
                    for ring in seg:
                        if not ring or len(ring) < 6:
                            continue
                        xs = ring[0::2]; ys = ring[1::2]
                        cx, cy = self._img_to_canvas(sum(xs)/len(xs), sum(ys)/len(ys))
                        centroids.append((cx, cy))
                    for k in range(len(centroids) - 1):
                        self.canvas.create_line(
                            centroids[k][0], centroids[k][1],
                            centroids[k+1][0], centroids[k+1][1],
                            fill=color, dash=(6, 4), width=1, tags="ann"
                        )
            else:
                x, y, w, h = ann["bbox"]
                x1, y1 = self._img_to_canvas(x, y)
                x2, y2 = self._img_to_canvas(x + w, y + h)
                # Semi-transparent fill layer
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="", fill=color,
                    stipple=fill_stipple, tags="ann"
                )
                # Solid outline on top
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline=color, fill="", width=lw, tags="ann"
                )
                label_x, label_y = x1, y1

            # Label with solid background for readability
            label_text = f" {name} "
            # Background pill
            self.canvas.create_rectangle(
                label_x, label_y - 16, label_x + len(label_text) * 7, label_y,
                fill=color, outline="", tags="ann"
            )
            self.canvas.create_text(
                label_x + 4, label_y - 8, text=name,
                anchor=tk.W, fill="#000000",
                font=("Helvetica", 8, "bold"), tags="ann"
            )

        # Draw vertex handles when in edit mode
        if self.edit_ann_id is not None:
            ann = self._get_ann_by_id(self.edit_ann_id)
            if ann and ann.get("segmentation") and ann["segmentation"][0]:
                pts = ann["segmentation"][0]
                edit_color = self.color_map.get(
                    self._cat_name(ann["category_id"]), "#1abc9c"
                )
                n = len(pts) // 2
                for i in range(n):
                    cx, cy = self._img_to_canvas(pts[i * 2], pts[i * 2 + 1])
                    # Vertex handle
                    r = 6
                    self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        fill=edit_color, outline="white", width=2, tags="edit"
                    )
                    self.canvas.create_text(
                        cx, cy, text=str(i), fill="black",
                        font=("Helvetica", 7, "bold"), tags="edit"
                    )

    def _mode_cursor(self):
        """Return the canvas cursor for the current draw mode."""
        m = self.draw_mode.get()
        return {"bbox": "crosshair", "polygon": "pencil", "sambox": "tcross"}.get(m, "crosshair")

    def _on_mode_change(self):
        self._cancel_polygon()
        self._exit_edit_mode()
        mode = self.draw_mode.get()
        self.canvas.config(cursor=self._mode_cursor())
        self._set_status("BBox mode" if mode == "bbox" else
                         "Polygon mode — click to place points, double-click to finish, Esc to cancel")

    def _cancel_polygon(self):
        for item in self.poly_preview_items:
            self.canvas.delete(item)
        self.poly_preview_items = []
        self.poly_points = []

    # ── Polygon edit mode ─────────────────────────────────────────────────────

    def _get_ann_by_id(self, ann_id):
        for a in self.coco["annotations"]:
            if a["id"] == ann_id:
                return a
        return None

    def _enter_edit_mode(self):
        if self.selected_ann_idx is None:
            self._set_status("Select a polygon annotation first.")
            return
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        if self.selected_ann_idx >= len(anns):
            return
        ann = anns[self.selected_ann_idx]
        seg = ann.get("segmentation")
        if not seg or not seg[0] or len(seg[0]) < 6:
            self._set_status("Selected annotation has no polygon — draw one first.")
            return
        self.edit_ann_id = ann["id"]
        self.drag_vertex_idx = None
        self.canvas.config(cursor="fleur")
        self._render()
        self._set_status(
            "Edit mode — drag vertex to move  |  click edge to insert point  "
            "|  right-click vertex to delete  |  Esc to finish"
        )

    def _exit_edit_mode(self):
        if self.edit_ann_id is None:
            return
        self.edit_ann_id = None
        self.drag_vertex_idx = None
        self.drag_start = None
        if self.ghost_dot:
            self.canvas.delete(self.ghost_dot)
            self.ghost_dot = None
        self.canvas.config(cursor=self._mode_cursor())
        self._render()
        self._set_status("Edit complete.")

    # ── SAM auto-segment ──────────────────────────────────────────────────────

    # ── SAM ───────────────────────────────────────────────────────────────────

    def _sam_box_segment(self, ix0, iy0, ix1, iy1):
        """SAM Box mode: run SAM with drawn bbox, save tight polygon."""
        predictor = self._sam_ensure_embedded()
        if predictor is None:
            return
        self._set_status("SAM tightening... please wait")
        self.root.update_idletasks()
        box = np.array([ix0, iy0, ix1, iy1])
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]
        self._sam_save_mask(best_mask)

    def _tighten_selected(self):
        """Tighten Selected: replace chosen annotation's polygon using SAM bbox prompt."""
        if self.selected_ann_idx is None:
            self._set_status("Select an annotation first.")
            return
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        if self.selected_ann_idx >= len(anns):
            return
        ann = anns[self.selected_ann_idx]
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            self._set_status("No bbox on this annotation — cannot tighten.")
            return

        predictor = self._sam_ensure_embedded()
        if predictor is None:
            return

        self._set_status("SAM tightening... please wait")
        self.root.update_idletasks()

        ix0, iy0, bw, bh = bbox
        ix1, iy1 = ix0 + bw, iy0 + bh
        box = np.array([ix0, iy0, ix1, iy1])
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]

        import cv2
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._set_status("SAM found no contour — try adjusting the bbox first.")
            return
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            self._set_status("SAM contour too simple — try again.")
            return

        pts_flat = []
        for pt in approx:
            x, y = float(pt[0][0]), float(pt[0][1])
            pts_flat.extend([
                max(0.0, min(float(self.img_w), x)),
                max(0.0, min(float(self.img_h), y)),
            ])

        ys, xs = np.where(best_mask)
        nbx, nby = float(xs.min()), float(ys.min())
        nbw, nbh = float(xs.max() - xs.min()), float(ys.max() - ys.min())

        # Update annotation in place
        for a in self.coco["annotations"]:
            if a["id"] == ann["id"]:
                a["segmentation"] = [pts_flat]
                a["bbox"] = [nbx, nby, nbw, nbh]
                a["area"] = float(nbw * nbh)
                break

        # Release edit mode so user can immediately start the next annotation
        self._exit_edit_mode()
        self.selected_ann_idx = None
        self._rebuild_lookup()
        self._render()
        self._refresh_ann_list()
        self._autosave(f"Tightened: {self._cat_name(ann['category_id'])} ({len(approx)} pts)")

    def _sam_confirm_point(self, canvas_x, canvas_y):
        """Shift+click: one-shot SAM segment at point, save immediately."""
        predictor = self._sam_ensure_embedded()
        if predictor is None:
            return
        img_x, img_y = self._canvas_to_img(canvas_x, canvas_y)
        img_x = max(0, min(int(img_x), self.img_w - 1))
        img_y = max(0, min(int(img_y), self.img_h - 1))
        self.sam_points = [[img_x, img_y, 1]]
        mask = self._sam_run_predict()
        if mask is not None:
            self._sam_save_mask(mask)
        self._sam_cancel()

    def _sam_warmup(self):
        """Load SAM model at startup then embed the first image — runs in background."""
        predictor = _get_sam_predictor()
        if predictor is None or not self.image_files:
            return
        filename = self.image_files[self.current_idx]
        if self.sam_embedding_img == filename:
            return
        try:
            img_array = np.array(self._get_display_image())
            with _sam_embedding_lock:
                predictor.set_image(img_array)
                self.sam_embedding_img = filename
                self.sam_embedded = True
            print("[SAM] Warmed up and ready.")
        except Exception as e:
            print(f"[SAM warmup] {e}")

    def _sam_prefetch(self, filename, img_array):
        """Background thread: embed image into SAM so it's ready when needed."""
        predictor = _get_sam_predictor()
        if predictor is None:
            return
        with _sam_embedding_lock:
            # Skip if the user has already moved on
            if self.sam_embedding_img == filename:
                return
            try:
                predictor.set_image(img_array)
                self.sam_embedding_img = filename
                self.sam_embedded = True
            except Exception as e:
                print(f"[SAM prefetch] {e}")

    def _sam_ensure_embedded(self):
        """Embed current image into SAM predictor if not already done."""
        predictor = _get_sam_predictor()
        if predictor is None:
            if not os.path.exists(SAM_MODEL_PATH):
                self._set_status("SAM model not found — place sam_vit_b.pth in Annotator/Model/.")
            else:
                self._set_status("SAM failed to load (check torch/segment_anything install).")
            return None
        current_file = self.image_files[self.current_idx]
        if self.sam_embedding_img != current_file:
            self._set_status("SAM embedding... please wait")
            self.root.update_idletasks()
            with _sam_embedding_lock:
                predictor.set_image(np.array(self._get_display_image()))
                self.sam_embedding_img = current_file
                self.sam_embedded = True
        return predictor

    def _sam_run_predict(self):
        """Run SAM with all current session points, draw preview, return mask."""
        predictor = _get_sam_predictor()
        if not predictor or not self.sam_points:
            return None
        coords = np.array([[p[0], p[1]] for p in self.sam_points])
        labels = np.array([p[2] for p in self.sam_points])
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]
        self._sam_draw_preview(best_mask)
        return best_mask

    def _sam_draw_preview(self, mask):
        """Overlay the SAM mask and point dots on the canvas."""
        import cv2
        # Clear previous preview items
        for item in self.sam_preview_items:
            self.canvas.delete(item)
        self.sam_preview_items = []

        color = self.color_map.get(self._current_class(), "#00d4ff")

        # Draw mask contour as dashed polygon
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                canvas_pts = []
                for pt in approx:
                    cx, cy = self._img_to_canvas(float(pt[0][0]), float(pt[0][1]))
                    canvas_pts.extend([cx, cy])
                item = self.canvas.create_polygon(
                    canvas_pts, outline=color, fill=color,
                    width=2, dash=(6, 3), stipple="gray25", tags="sam_preview"
                )
                self.sam_preview_items.append(item)
                item2 = self.canvas.create_polygon(
                    canvas_pts, outline=color, fill="",
                    width=2, dash=(6, 3), tags="sam_preview"
                )
                self.sam_preview_items.append(item2)

        # Draw point dots
        for px, py, label in self.sam_points:
            cx, cy = self._img_to_canvas(px, py)
            dot_color = "#2ecc71" if label == 1 else "#e74c3c"
            r = 7
            item = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=dot_color, outline="white", width=2, tags="sam_preview"
            )
            self.sam_preview_items.append(item)
            sym = "+" if label == 1 else "−"
            item2 = self.canvas.create_text(
                cx, cy, text=sym, fill="white",
                font=("Helvetica", 10, "bold"), tags="sam_preview"
            )
            self.sam_preview_items.append(item2)

    def _sam_add_point(self, canvas_x, canvas_y, label):
        """Add a SAM point (label=1 foreground, label=0 background) and update preview."""
        predictor = self._sam_ensure_embedded()
        if predictor is None:
            return
        img_x, img_y = self._canvas_to_img(canvas_x, canvas_y)
        img_x = max(0, min(int(img_x), self.img_w - 1))
        img_y = max(0, min(int(img_y), self.img_h - 1))
        self.sam_points.append([img_x, img_y, label])
        self.sam_session_active = True
        self._sam_run_predict()
        fg = sum(1 for p in self.sam_points if p[2] == 1)
        bg = sum(1 for p in self.sam_points if p[2] == 0)
        self._set_status(
            f"SAM: {fg} foreground (+)  {bg} background (−)  |  "
            "Enter to confirm  |  Esc to cancel"
        )

    def _sam_save_mask(self, best_mask):
        """Convert a SAM binary mask to a COCO annotation and save."""
        import cv2
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._set_status("SAM found no contour — try clicking closer to the object.")
            return
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            self._set_status("SAM contour too simple — try again.")
            return
        pts_flat = []
        for pt in approx:
            x, y = float(pt[0][0]), float(pt[0][1])
            pts_flat.extend([
                max(0.0, min(float(self.img_w), x)),
                max(0.0, min(float(self.img_h), y)),
            ])
        ys, xs = np.where(best_mask)
        bx, by = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        current_file = self.image_files[self.current_idx]
        rec = self._ensure_image_record(current_file, self.img_w, self.img_h)
        class_name = self._current_class()
        ann = {
            "id": self._next_ann_id(),
            "image_id": rec["id"],
            "category_id": self._cat_id(class_name),
            "segmentation": [pts_flat],
            "bbox": [bx, by, bw, bh],
            "area": float(bw * bh),
            "iscrowd": 0,
        }
        self.coco["annotations"].append(ann)
        self._rebuild_lookup()
        self._render()
        self._refresh_ann_list()
        self._autosave(f"SAM: {class_name} added ({len(approx)} pts)")

    def _sam_cancel(self):
        """Cancel SAM session and clear preview."""
        for item in self.sam_preview_items:
            self.canvas.delete(item)
        self.sam_preview_items = []
        self.sam_points = []
        self.sam_session_active = False

    def _edit_get_canvas_pts(self):
        ann = self._get_ann_by_id(self.edit_ann_id)
        if not ann:
            return []
        pts = ann["segmentation"][0]
        canvas_pts = []
        for i in range(0, len(pts), 2):
            cx, cy = self._img_to_canvas(pts[i], pts[i + 1])
            canvas_pts.append((cx, cy))
        return canvas_pts

    def _point_in_polygon(self, px, py, polygon):
        """Ray-casting point-in-polygon test. polygon is list of (x, y) tuples."""
        n = len(polygon)
        inside = False
        x, y = px, py
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        return inside

    def _pt_dist(self, ax, ay, bx, by):
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _pt_to_seg_dist(self, px, py, ax, ay, bx, by):
        """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return self._pt_dist(px, py, ax, ay)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        return self._pt_dist(px, py, ax + t * dx, ay + t * dy)

    def _edit_mouse_down(self, event):
        canvas_pts = self._edit_get_canvas_pts()
        if not canvas_pts:
            return
        HIT_R = 10  # px radius for vertex hit

        # Check vertex hit
        for i, (cx, cy) in enumerate(canvas_pts):
            if self._pt_dist(event.x, event.y, cx, cy) <= HIT_R:
                self.drag_vertex_idx = i
                self.drag_start = (event.x, event.y)
                return

        # Check edge hit — insert new point
        EDGE_R = 8
        n = len(canvas_pts)
        for i in range(n):
            ax, ay = canvas_pts[i]
            bx, by = canvas_pts[(i + 1) % n]
            if self._pt_to_seg_dist(event.x, event.y, ax, ay, bx, by) <= EDGE_R:
                # Insert new vertex at click position in image coords
                ix, iy = self._canvas_to_img(event.x, event.y)
                ann = self._get_ann_by_id(self.edit_ann_id)
                pts = ann["segmentation"][0]
                insert_at = (i + 1) * 2
                pts.insert(insert_at, round(iy, 2))
                pts.insert(insert_at, round(ix, 2))
                self.drag_vertex_idx = i + 1
                self.drag_start = (event.x, event.y)
                self._render()
                self._set_status(f"Inserted point at index {i + 1}")
                return

    def _edit_mouse_drag(self, event):
        if self.drag_vertex_idx is None:
            return
        ann = self._get_ann_by_id(self.edit_ann_id)
        if not ann:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        ix = max(0.0, min(ix, self.img_w))
        iy = max(0.0, min(iy, self.img_h))
        pts = ann["segmentation"][0]
        vi = self.drag_vertex_idx * 2
        pts[vi] = round(ix, 2)
        pts[vi + 1] = round(iy, 2)
        self._render()

    def _edit_mouse_up(self, event):
        if self.drag_vertex_idx is None:
            return
        # Recalculate bbox from updated polygon
        ann = self._get_ann_by_id(self.edit_ann_id)
        if ann:
            pts = ann["segmentation"][0]
            xs, ys = pts[0::2], pts[1::2]
            bx, by = min(xs), min(ys)
            bw, bh = max(xs) - bx, max(ys) - by
            ann["bbox"] = [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)]
            ann["area"] = round(bw * bh, 2)
        self.drag_vertex_idx = None
        self.drag_start = None
        self._autosave("Polygon updated.")

    def _on_right_click(self, event):
        # If already in edit mode, handle vertex deletion or exit edit mode
        if self.edit_ann_id is not None:
            self._on_edit_right_click(event)
            return
        # Otherwise try to enter edit mode for clicked polygon
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        for i, ann in enumerate(anns):
            seg = ann.get("segmentation")
            if not seg:
                continue
            for ring in seg:
                if not ring or len(ring) < 6:
                    continue
                canvas_pts = []
                for j in range(0, len(ring), 2):
                    cx, cy = self._img_to_canvas(ring[j], ring[j + 1])
                    canvas_pts.append((cx, cy))
                if self._point_in_polygon(event.x, event.y, canvas_pts):
                    self.selected_ann_idx = i
                    self.ann_listbox.select_clear(0, tk.END)
                    self.ann_listbox.select_set(i)
                    self.ann_listbox.see(i)
                    self.edit_ann_id = ann["id"]
                    self.drag_vertex_idx = None
                    self._render()
                    self._set_status(
                        f"Editing: {self._cat_name(ann['category_id'])}  |  "
                        "drag vertex to move  |  click edge to insert  |  right-click vertex to delete  |  Esc to finish"
                    )
                    return
        # Clicked empty space — deselect
        self.selected_ann_idx = None
        self.ann_listbox.select_clear(0, tk.END)
        self._render()
        self._set_status("Deselected.")

    def _on_edit_right_click(self, event):
        if self.edit_ann_id is None:
            return
        canvas_pts = self._edit_get_canvas_pts()
        HIT_R = 10
        for i, (cx, cy) in enumerate(canvas_pts):
            if self._pt_dist(event.x, event.y, cx, cy) <= HIT_R:
                if len(canvas_pts) <= 3:
                    self._set_status("Cannot delete — polygon needs at least 3 points.")
                    return
                ann = self._get_ann_by_id(self.edit_ann_id)
                pts = ann["segmentation"][0]
                del pts[i * 2: i * 2 + 2]
                self._render()
                self._autosave(f"Deleted vertex {i}.")
                return
        # Missed all vertices — exit edit mode, ready for new annotation
        self._exit_edit_mode()
        self.selected_ann_idx = None
        self.ann_listbox.select_clear(0, tk.END)
        self._render()
        self._set_status("Deselected — click a polygon or start drawing.")

    def _on_mouse_motion(self, event):
        if self.edit_ann_id is None or self.drag_vertex_idx is not None:
            if self.ghost_dot:
                self.canvas.delete(self.ghost_dot)
                self.ghost_dot = None
            return

        canvas_pts = self._edit_get_canvas_pts()
        if not canvas_pts:
            return

        EDGE_R = 14  # px snap distance to show ghost
        HIT_R = 10   # suppress ghost when near a vertex
        n = len(canvas_pts)

        # Suppress ghost near existing vertices — show hand2 cursor instead
        for cx, cy in canvas_pts:
            if self._pt_dist(event.x, event.y, cx, cy) <= HIT_R:
                if self.ghost_dot:
                    self.canvas.delete(self.ghost_dot)
                    self.ghost_dot = None
                self.canvas.config(cursor="hand2")
                return

        self.canvas.config(cursor="fleur")

        # Find closest edge and project mouse onto it
        best_dist = float("inf")
        best_proj = None
        for i in range(n):
            ax, ay = canvas_pts[i]
            bx, by = canvas_pts[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                continue
            t = max(0.0, min(1.0, ((event.x - ax) * dx + (event.y - ay) * dy) / seg_len_sq))
            proj_x, proj_y = ax + t * dx, ay + t * dy
            dist = self._pt_dist(event.x, event.y, proj_x, proj_y)
            if dist < best_dist:
                best_dist = dist
                best_proj = (proj_x, proj_y)

        if self.ghost_dot:
            self.canvas.delete(self.ghost_dot)
            self.ghost_dot = None

        if best_proj and best_dist <= EDGE_R:
            px, py = best_proj
            ann = self._get_ann_by_id(self.edit_ann_id)
            color = self.color_map.get(self._cat_name(ann["category_id"]), "#1abc9c")
            r = 5
            self.ghost_dot = self.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill="", outline=color, width=2,
                dash=(3, 3), tags="ghost"
            )

    # ── BBox mouse handlers ───────────────────────────────────────────────────

    def _on_mouse_down(self, event):
        # Shift+click → SAM auto-segment
        if event.state & 0x0001:  # Shift held
            if self.edit_ann_id is None:
                self._sam_confirm_point(event.x, event.y)
                return
        # Edit mode — drag vertices
        if self.edit_ann_id is not None:
            self._edit_mouse_down(event)
            return
        # Polygon draw mode — place points
        if self.draw_mode.get() == "polygon":
            self._on_poly_click(event)
            return
        # BBox draw mode
        self.draw_start = (event.x, event.y)
        self.preview_rect = None

    def _on_mouse_drag(self, event):
        if self.edit_ann_id is not None:
            self._edit_mouse_drag(event)
            return
        if self.draw_mode.get() == "polygon" or self.draw_start is None:
            return
        if self.preview_rect:
            self.canvas.delete(self.preview_rect)
        x0, y0 = self.draw_start
        is_sam = self.draw_mode.get() == "sambox"
        color = "#00d4ff" if is_sam else self.color_map.get(self._current_class(), "#00d4ff")
        dash = (8, 4) if is_sam else (4, 2)
        self.preview_rect = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline=color, width=2, dash=dash
        )

    def _on_mouse_up(self, event):
        if self.edit_ann_id is not None:
            self._edit_mouse_up(event)
            return
        if self.draw_mode.get() == "polygon" or self.draw_start is None:
            return
        if self.preview_rect:
            self.canvas.delete(self.preview_rect)
            self.preview_rect = None

        x0, y0 = self.draw_start
        x1, y1 = event.x, event.y
        self.draw_start = None

        if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
            return

        ix0, iy0 = self._canvas_to_img(min(x0, x1), min(y0, y1))
        ix1, iy1 = self._canvas_to_img(max(x0, x1), max(y0, y1))
        ix0 = max(0.0, min(ix0, self.img_w))
        iy0 = max(0.0, min(iy0, self.img_h))
        ix1 = max(0.0, min(ix1, self.img_w))
        iy1 = max(0.0, min(iy1, self.img_h))
        bw, bh = ix1 - ix0, iy1 - iy0
        if bw < 2 or bh < 2:
            return

        # SAM Box mode — run SAM on drawn region instead of saving raw bbox
        if self.draw_mode.get() == "sambox":
            self._sam_box_segment(ix0, iy0, ix1, iy1)
            return

        filename = self.image_files[self.current_idx]
        rec = self._ensure_image_record(filename, self.img_w, self.img_h)
        ann = {
            "id": self._next_ann_id(),
            "image_id": rec["id"],
            "category_id": self._cat_id(self._current_class()),
            "bbox": [round(ix0, 2), round(iy0, 2), round(bw, 2), round(bh, 2)],
            "area": round(bw * bh, 2),
            "iscrowd": 0,
            "segmentation": [],
        }
        self.coco["annotations"].append(ann)
        self._rebuild_lookup()
        self._render()
        self._refresh_ann_list()
        self._autosave(f"Added bbox: {self._current_class()}")

    # ── Polygon mouse handlers ────────────────────────────────────────────────

    def _on_poly_click(self, event):
        color = self.color_map.get(self._current_class(), "#00d4ff")

        self.poly_points.append((event.x, event.y))

        # Draw vertex dot
        r = 4
        dot = self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill=color, outline="white", width=1
        )
        self.poly_preview_items.append(dot)

        # Draw edge from previous point
        if len(self.poly_points) > 1:
            px, py = self.poly_points[-2]
            line = self.canvas.create_line(
                px, py, event.x, event.y,
                fill=color, width=2, dash=(4, 2)
            )
            self.poly_preview_items.append(line)

        self._set_status(
            f"Polygon: {len(self.poly_points)} points — double-click to finish, Esc to cancel"
        )

    def _on_poly_double_click(self, event):
        if self.draw_mode.get() != "polygon" or self.edit_ann_id is not None:
            return
        if len(self.poly_points) < 3:
            self._set_status("Need at least 3 points for a polygon.")
            return

        pts = self.poly_points[:]
        self._cancel_polygon()

        # Convert canvas → image coords
        img_pts = []
        for cx, cy in pts:
            ix, iy = self._canvas_to_img(cx, cy)
            ix = max(0.0, min(ix, self.img_w))
            iy = max(0.0, min(iy, self.img_h))
            img_pts.extend([round(ix, 2), round(iy, 2)])

        # Derive bbox from polygon extents
        xs = img_pts[0::2]
        ys = img_pts[1::2]
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by

        filename = self.image_files[self.current_idx]
        rec = self._ensure_image_record(filename, self.img_w, self.img_h)
        ann = {
            "id": self._next_ann_id(),
            "image_id": rec["id"],
            "category_id": self._cat_id(self._current_class()),
            "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
            "area": round(bw * bh, 2),
            "iscrowd": 0,
            "segmentation": [img_pts],
        }
        self.coco["annotations"].append(ann)
        self._rebuild_lookup()
        self._render()
        self._refresh_ann_list()
        self._autosave(f"Added polygon: {self._current_class()}")

    # ── Class management ──────────────────────────────────────────────────────

    def _make_dialog(self, title, width, height):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg="#16213e", bd=2, relief="groove")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        self.root.update_idletasks()
        rx = self.root.winfo_x()
        ry = self.root.winfo_y()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        x = rx + (rw - width) // 2
        y = ry + (rh - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        return dialog

    def _add_class(self):
        self._exit_edit_mode()
        dialog = self._make_dialog("Add Class", 280, 120)
        dialog.grab_set()
        tk.Label(dialog, text="Class name:", bg="#16213e", fg="white",
                 font=("Helvetica", 10)).pack(pady=(12, 4))
        entry = tk.Entry(dialog, bg="#0f3460", fg="white", insertbackground="white",
                         relief="flat", font=("Helvetica", 10))
        entry.pack(padx=16, fill=tk.X)
        entry.focus()

        def confirm(e=None):
            name = entry.get().strip()
            if not name or name in self.class_names:
                dialog.destroy()
                return
            self.class_names.append(name)
            new_id = max((c["id"] for c in self.coco["categories"]), default=0) + 1
            self.coco["categories"].append({"id": new_id, "name": name})
            self.color_map[name] = DEFAULT_COLORS[
                (len(self.class_names) - 1) % len(DEFAULT_COLORS)
            ]
            self.class_listbox.insert(tk.END, name)
            dialog.destroy()
            self._set_status(f"Class added: {name}")

        entry.bind("<Return>", confirm)
        tk.Button(dialog, text="Add", command=confirm, bg="#2ecc71", fg="#000",
                  relief="flat", font=("Helvetica", 10, "bold")).pack(pady=6)

    def _remove_class(self):
        self._exit_edit_mode()
        sel = self.class_listbox.curselection()
        if not sel:
            return
        name = self.class_names[sel[0]]
        ann_count = sum(
            1 for a in self.coco["annotations"]
            if self._cat_name(a["category_id"]) == name
        )
        msg = f"Remove '{name}'?"
        if ann_count:
            msg += f"\n{ann_count} annotation(s) with this class will also be deleted."
        else:
            msg += "\nNo annotations use this class."
        if messagebox.askyesno("Remove Class", msg):
            # Remove ALL categories with this name (handles duplicates like new_Class)
            cat_ids = {c["id"] for c in self.coco["categories"] if c["name"] == name}
            self.coco["categories"] = [
                c for c in self.coco["categories"] if c["name"] != name
            ]
            if ann_count:
                self.coco["annotations"] = [
                    a for a in self.coco["annotations"]
                    if a["category_id"] not in cat_ids
                ]
            # Remove all occurrences from class_names / listbox
            self.class_listbox.delete(0, tk.END)
            self.class_names = [n for n in self.class_names if n != name]
            for n in self.class_names:
                self.class_listbox.insert(tk.END, n)
            if self.class_names:
                self.class_listbox.select_set(0)
            self._update_color_btn()
            self._save()
            self._render()
            self._set_status(f"Class removed: {name}")

    def _change_class_color(self):
        self._exit_edit_mode()
        sel = self.class_listbox.curselection()
        if not sel:
            self._set_status("Select a class first.")
            return
        name = self.class_names[sel[0]]
        current = self.color_map.get(name, "#ffffff")
        result = colorchooser.askcolor(color=current, title=f"Choose color for '{name}'")
        if result and result[1]:
            new_color = result[1]
            self.color_map[name] = new_color
            self._update_color_btn()
            self._render()
            self._set_status(f"Color updated: {name} → {new_color}")

    def _update_color_btn(self):
        sel = self.class_listbox.curselection()
        if sel and self.class_names:
            name = self.class_names[sel[0]]
            color = self.color_map.get(name, "#ffffff")
            self.color_btn.config(bg=color, fg="black" if self._is_light(color) else "white")
        else:
            self.color_btn.config(bg="#2c2c54", fg="white")

    def _is_light(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (r * 0.299 + g * 0.587 + b * 0.114) > 140

    # ── Annotation List ───────────────────────────────────────────────────────

    def _refresh_ann_list(self):
        self.ann_listbox.delete(0, tk.END)
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        for ann in self._get_annotations(rec["id"]):
            name = self._cat_name(ann["category_id"])
            x, y, w, h = [int(v) for v in ann["bbox"]]
            self.ann_listbox.insert(tk.END, f"{name}  [{x},{y} {w}×{h}]")

    def _on_ann_select(self, event):
        sel = self.ann_listbox.curselection()
        self.selected_ann_idx = sel[0] if sel else None

        # If already in edit mode, switch to editing the newly selected annotation
        if self.edit_ann_id is not None and sel:
            rec = self._get_image_record(self.image_files[self.current_idx])
            if rec:
                anns = self._get_annotations(rec["id"])
                if sel[0] < len(anns):
                    ann = anns[sel[0]]
                    seg = ann.get("segmentation")
                    if seg and seg[0] and len(seg[0]) >= 6:
                        self.edit_ann_id = ann["id"]
                        self.drag_vertex_idx = None
                        self._render()
                        self._set_status(f"Editing: {self._cat_name(ann['category_id'])}")
                        return
                    else:
                        self._exit_edit_mode()
                        self._set_status("No polygon on this annotation — edit mode off.")
                        self._render()
                        return

        self._render()

    def _relabel_selected(self):
        self._exit_edit_mode()
        if self.selected_ann_idx is None:
            self._set_status("Select an annotation first.")
            return
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        if self.selected_ann_idx >= len(anns):
            return
        target_ann = anns[self.selected_ann_idx]
        current_name = self._cat_name(target_ann["category_id"])

        w, h = 220, 300
        self.root.update_idletasks()
        bx = self.remove_class_btn.winfo_rootx() + self.remove_class_btn.winfo_width() + 4
        by = self.remove_class_btn.winfo_rooty()
        dialog = tk.Toplevel(self.root)
        dialog.title("")
        dialog.configure(bg="#16213e", bd=2, relief="groove")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.geometry(f"{w}x{h}+{bx}+{by}")
        dialog.grab_set()

        tk.Label(dialog, text=f"Current: {current_name}", bg="#16213e",
                 fg="#aaa", font=("Helvetica", 9)).pack(pady=(10, 2))
        tk.Label(dialog, text="Select new class:", bg="#16213e",
                 fg="white", font=("Helvetica", 10)).pack(pady=(0, 6))

        lb = tk.Listbox(dialog, bg="#0f3460", fg="white",
                        selectbackground="#9b59b6", selectforeground="white",
                        font=("Helvetica", 10), relief="flat",
                        highlightthickness=0)
        for name in self.class_names:
            lb.insert(tk.END, name)
        # Pre-select current class
        if current_name in self.class_names:
            lb.select_set(self.class_names.index(current_name))
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        def confirm(e=None):
            sel = lb.curselection()
            if not sel:
                dialog.destroy()
                return
            new_name = self.class_names[sel[0]]
            new_cat_id = self._cat_id(new_name)
            # Update in place across all annotations
            for a in self.coco["annotations"]:
                if a["id"] == target_ann["id"]:
                    a["category_id"] = new_cat_id
                    break
            dialog.destroy()
            self._render()
            self._refresh_ann_list()
            self._autosave(f"Relabeled: {current_name} → {new_name}")

        lb.bind("<Double-Button-1>", confirm)
        tk.Button(dialog, text="Apply", command=confirm,
                  bg="#9b59b6", fg="white", relief="flat",
                  font=("Helvetica", 10, "bold")).pack(pady=6)

    def _delete_selected(self):
        self._exit_edit_mode()
        if self.selected_ann_idx is None:
            return
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        if self.selected_ann_idx >= len(anns):
            return
        target_id = anns[self.selected_ann_idx]["id"]
        self.coco["annotations"] = [
            a for a in self.coco["annotations"] if a["id"] != target_id
        ]
        self._rebuild_lookup()
        self.selected_ann_idx = None
        self._render()
        self._refresh_ann_list()
        self._autosave("Annotation deleted.")

    # ── Convex Hull Merge ──────────────────────────────────────────────────────

    @staticmethod
    def _convex_hull(points):
        """Jarvis march (gift wrapping) convex hull. Returns ordered hull points."""
        pts = list({(round(x, 2), round(y, 2)) for x, y in points})
        if len(pts) < 3:
            return pts
        start = min(pts, key=lambda p: (p[0], p[1]))
        hull = [start]
        while True:
            candidate = pts[0]
            for p in pts[1:]:
                v1x = candidate[0] - hull[-1][0]
                v1y = candidate[1] - hull[-1][1]
                v2x = p[0] - hull[-1][0]
                v2y = p[1] - hull[-1][1]
                cross = v1x * v2y - v1y * v2x
                if cross < 0 or (cross == 0 and
                   (v2x ** 2 + v2y ** 2) > (v1x ** 2 + v1y ** 2)):
                    candidate = p
            if candidate == hull[0]:
                break
            hull.append(candidate)
        return hull

    def _hull_update_ui(self):
        n = len(self.hull_ann_ids)
        self.hull_add_btn.config(text=f"⬡  Add to Union ({n})")
        if n >= 2:
            self.hull_merge_btn.config(state=tk.NORMAL, fg="#1abc9c")
        else:
            self.hull_merge_btn.config(state=tk.DISABLED, fg="#888")

    def _hull_add(self):
        if self.selected_ann_idx is None:
            self._set_status("Select an annotation first.")
            return
        rec = self._get_image_record(self.image_files[self.current_idx])
        if not rec:
            return
        anns = self._get_annotations(rec["id"])
        if self.selected_ann_idx >= len(anns):
            return
        ann = anns[self.selected_ann_idx]
        seg = ann.get("segmentation")
        if not seg or not any(r and len(r) >= 6 for r in seg):
            self._set_status("Selected annotation has no polygon.")
            return
        ann_id = ann["id"]
        if ann_id in self.hull_ann_ids:
            # Toggle off — remove from collection
            self.hull_ann_ids.remove(ann_id)
            self._set_status(f"Removed from hull collection. ({len(self.hull_ann_ids)} collected)")
        else:
            self.hull_ann_ids.append(ann_id)
            self._set_status(
                f"Added to hull ({len(self.hull_ann_ids)} collected)  |  "
                "Add more then click 'Merge Hull'."
            )
        self._hull_update_ui()
        self._render()

    def _hull_merge(self):
        if len(self.hull_ann_ids) < 2:
            self._set_status("Need at least 2 annotations to merge.")
            return
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except ImportError:
            self._set_status("Run: pip install shapely")
            return

        first_ann = None
        shapely_polys = []
        for ann_id in self.hull_ann_ids:
            ann = self._get_ann_by_id(ann_id)
            if ann is None:
                continue
            if first_ann is None:
                first_ann = ann
            seg = ann.get("segmentation", [])
            rings = [r for r in seg if r and len(r) >= 6]
            if not rings:
                # Fall back to bbox as rectangle polygon
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    rings = [[x, y, x+w, y, x+w, y+h, x, y+h]]
            for ring in rings:
                coords = [(ring[i], ring[i + 1]) for i in range(0, len(ring), 2)]
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and not poly.is_empty:
                        shapely_polys.append(poly)
                except Exception:
                    pass

        if not shapely_polys or first_ann is None:
            self._set_status("Not enough valid polygons to merge.")
            return

        merged = unary_union(shapely_polys)

        # Convert result back to COCO segmentation rings
        new_rings = []
        geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        for geom in geoms:
            if geom.is_empty:
                continue
            coords = list(geom.exterior.coords)
            flat = []
            for x, y in coords[:-1]:  # drop closing duplicate
                flat.extend([round(x, 2), round(y, 2)])
            if len(flat) >= 6:
                new_rings.append(flat)

        if not new_rings:
            self._set_status("Merge produced no valid polygon.")
            return

        # Update first annotation
        first_ann["segmentation"] = new_rings
        all_xs = [v for ring in new_rings for v in ring[0::2]]
        all_ys = [v for ring in new_rings for v in ring[1::2]]
        bx, by = min(all_xs), min(all_ys)
        bw, bh = max(all_xs) - bx, max(all_ys) - by
        first_ann["bbox"] = [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)]
        first_ann["area"] = round(bw * bh, 2)

        # Delete all other collected annotations
        ids_to_delete = set(self.hull_ann_ids) - {first_ann["id"]}
        self.coco["annotations"] = [
            a for a in self.coco["annotations"] if a["id"] not in ids_to_delete
        ]

        # Reset hull state
        self.hull_ann_ids = []
        self._hull_update_ui()
        self.selected_ann_idx = None
        self._rebuild_lookup()
        self._render()
        self._refresh_ann_list()
        self._autosave("Polygon union merge complete.")
        n_rings = len(new_rings)
        msg = "Merged → one polygon." if n_rings == 1 else f"Merged → {n_rings} rings (segments don't overlap — draw overlap or use Join)."
        self._set_status(msg)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev_image(self):
        if self.current_idx > 0:
            self._load_image(self.current_idx - 1)

    def _next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self._load_image(self.current_idx + 1)

    # ── Image Browser ─────────────────────────────────────────────────────────

    def _refresh_image_browser(self):
        query = self.img_search_var.get().lower()
        filt = self.img_filter_var.get()
        self.img_listbox.delete(0, tk.END)
        self._img_browser_indices = []
        for i, fname in enumerate(self.image_files):
            # Search filter
            if query and query not in fname.lower():
                continue
            # Status filter
            rec = self._get_image_record(fname)
            anns = self._get_annotations(rec["id"]) if rec else []
            has_anns = bool(anns)
            is_flagged = fname in self.flagged_images
            has_model_anns = any("score" in a for a in anns)
            if filt == "done"  and not has_anns:
                continue
            if filt == "todo"  and has_anns:
                continue
            if filt == "flag"  and not is_flagged:
                continue
            if filt == "model" and not has_model_anns:
                continue
            # Status indicator
            if is_flagged:
                indicator = "⚑ "
            elif has_model_anns:
                indicator = "🤖 "
            elif has_anns:
                indicator = "✓ "
            else:
                indicator = "○ "
            self.img_listbox.insert(tk.END, indicator + fname)
            self._img_browser_indices.append(i)
            # Highlight current image
            if i == self.current_idx:
                self.img_listbox.itemconfig(tk.END, bg="#1a4a6e", fg="white")

    def _on_img_browser_select(self, event):
        sel = self.img_listbox.curselection()
        if not sel:
            return
        img_idx = self._img_browser_indices[sel[0]]
        if img_idx != self.current_idx:
            self._load_image(img_idx)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_class(self):
        return self.active_class

    def _on_class_select(self, event):
        self._exit_edit_mode()
        sel = self.class_listbox.curselection()
        if sel:
            self.active_class = self.class_names[sel[0]]
            self.selected_class.set(self.active_class)
            self._update_color_btn()

    def _on_escape(self, event=None):
        if self.sam_session_active:
            self._sam_cancel()
            self._render()
            self._set_status("SAM session cancelled.")
        elif self.edit_ann_id is not None:
            self._exit_edit_mode()
        else:
            self._cancel_polygon()

    def _set_status(self, msg):
        self.status_label.config(text=msg)

    # ── Pre-annotation ────────────────────────────────────────────────────────

    def _pre_annotate(self, conf_threshold=0.35):
        """Run best.pt inference on the current image and add predictions as annotations."""
        best_pt = _find_best_pt()
        if not best_pt:
            self._set_status("best.pt not found — train a model first.")
            return

        if not self.image_files:
            return

        self._set_status("Running model inference... please wait")
        self.root.update_idletasks()

        try:
            from ultralytics import YOLO
        except ImportError:
            self._set_status("ultralytics not installed — run: pip3 install ultralytics")
            return

        try:
            img_path = self.image_files[self.current_idx]
            full_img_path = os.path.join(self.images_dir, img_path)
            model = YOLO(best_pt)
            results = model.predict(full_img_path, conf=conf_threshold, verbose=False)[0]
        except Exception as e:
            self._set_status(f"Inference error: {e}")
            return

        rec = self._get_image_record(img_path)
        if not rec:
            self._set_status("No image record — save the JSON first.")
            return

        W, H = rec["width"], rec["height"]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self._set_status("Model found no objects above confidence threshold.")
            return

        # Build category name → id map from current COCO JSON
        cat_name_to_id = {c["name"]: c["id"] for c in self.coco["categories"]}

        # Model class names from the YOLO result
        model_names = results.names  # dict: int → str

        added = 0
        skipped = 0
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            conf    = float(boxes.conf[i].item())
            cls_name = model_names.get(cls_idx, f"cls_{cls_idx}")

            if cls_name not in cat_name_to_id:
                skipped += 1
                continue

            # xyxy pixel coords
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = (max(0, x1), max(0, y1),
                               min(W, x2), min(H, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2:
                continue

            # Check for significant overlap with existing annotations (skip duplicates)
            existing_anns = self._get_annotations(rec["id"])
            duplicate = False
            for existing in existing_anns:
                ex, ey, ew, eh = existing["bbox"]
                ix1, iy1 = max(x1, ex), max(y1, ey)
                ix2, iy2 = min(x2, ex + ew), min(y2, ey + eh)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = bw * bh + ew * eh - inter
                    if union > 0 and inter / union > 0.5:
                        duplicate = True
                        break
            if duplicate:
                skipped += 1
                continue

            ann_id = max((a["id"] for a in self.coco["annotations"]), default=0) + 1
            self.coco["annotations"].append({
                "id": ann_id,
                "image_id": rec["id"],
                "category_id": cat_name_to_id[cls_name],
                "bbox": [x1, y1, bw, bh],
                "segmentation": [],
                "area": bw * bh,
                "iscrowd": 0,
                "score": round(conf, 3),
            })
            added += 1

        self._autosave()
        self._refresh_ann_list()
        self._render()
        msg = f"Pre-annotated: {added} added"
        if skipped:
            msg += f", {skipped} skipped (unknown class or duplicate)"
        self._set_status(msg)

    def _pre_annotate_sam(self, conf_threshold=0.35):
        """YOLO detects objects → SAM tightens each bbox into a precise polygon."""
        import cv2

        best_pt = _find_best_pt()
        if not best_pt:
            self._set_status("best.pt not found — train a model first.")
            return

        predictor = _get_sam_predictor()
        if predictor is None:
            self._set_status("SAM model not found — check models/sam_vit_b.pth.")
            return

        if not self.image_files:
            return

        img_path = self.image_files[self.current_idx]
        rec = self._get_image_record(img_path)
        if not rec:
            self._set_status("No image record for this image.")
            return

        W, H = rec["width"], rec["height"]

        # Resolve image source — local file or URL-fetched PIL image
        if self.images_dir:
            full_img_path = os.path.join(self.images_dir, img_path)
        elif img_path in self.url_images:
            full_img_path = self.url_images[img_path]  # PIL Image
        elif self.pil_image is not None:
            full_img_path = self.pil_image              # current displayed image
        else:
            self._set_status("Image not loaded yet — navigate to it first.")
            return

        # Step 1 — YOLO inference
        self._set_status("YOLO detecting objects... please wait")
        self.root.update_idletasks()
        try:
            from ultralytics import YOLO
            model = YOLO(best_pt)
            results = model.predict(full_img_path, conf=conf_threshold, verbose=False)[0]
        except Exception as e:
            self._set_status(f"YOLO error: {e}")
            return

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self._set_status("Model found no objects above confidence threshold.")
            return

        # Step 2 — embed image into SAM once
        self._set_status(f"SAM embedding image for {len(boxes)} detections... please wait")
        self.root.update_idletasks()
        try:
            predictor.set_image(np.array(self._get_display_image()))
        except Exception as e:
            self._set_status(f"SAM embed error: {e}")
            return

        cat_name_to_id = {c["name"]: c["id"] for c in self.coco["categories"]}
        model_names    = results.names
        existing_anns  = self._get_annotations(rec["id"])

        added = skipped = failed = 0

        for i in range(len(boxes)):
            cls_idx  = int(boxes.cls[i].item())
            conf     = float(boxes.conf[i].item())
            cls_name = model_names.get(cls_idx, f"cls_{cls_idx}")

            if cls_name not in cat_name_to_id:
                skipped += 1
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            bw, bh  = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                skipped += 1
                continue

            # Duplicate check (>50% IoU with existing)
            duplicate = False
            for existing in existing_anns:
                ex, ey, ew, eh = existing["bbox"]
                ix1e, iy1e = max(x1, ex), max(y1, ey)
                ix2e, iy2e = min(x2, ex + ew), min(y2, ey + eh)
                if ix2e > ix1e and iy2e > iy1e:
                    inter = (ix2e - ix1e) * (iy2e - iy1e)
                    union = bw * bh + ew * eh - inter
                    if union > 0 and inter / union > 0.5:
                        duplicate = True
                        break
            if duplicate:
                skipped += 1
                continue

            # Step 3 — SAM polygon for this bbox
            try:
                box_arr = np.array([x1, y1, x2, y2])
                masks, scores, _ = predictor.predict(
                    box=box_arr[None, :], multimask_output=True
                )
                best_mask = masks[np.argmax(scores)]

                mask_uint8 = (best_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    failed += 1
                    continue
                contour = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx  = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    failed += 1
                    continue

                pts_flat = []
                for pt in approx:
                    px, py = float(pt[0][0]), float(pt[0][1])
                    pts_flat.extend([
                        max(0.0, min(float(W), px)),
                        max(0.0, min(float(H), py)),
                    ])

                ys, xs = np.where(best_mask)
                seg_bx = float(xs.min()); seg_by = float(ys.min())
                seg_bw = float(xs.max() - xs.min())
                seg_bh = float(ys.max() - ys.min())

            except Exception:
                failed += 1
                continue

            ann_id = max((a["id"] for a in self.coco["annotations"]), default=0) + 1
            self.coco["annotations"].append({
                "id":          ann_id,
                "image_id":    rec["id"],
                "category_id": cat_name_to_id[cls_name],
                "segmentation": [pts_flat],
                "bbox":        [seg_bx, seg_by, seg_bw, seg_bh],
                "area":        float(seg_bw * seg_bh),
                "iscrowd":     0,
                "score":       round(conf, 3),
            })
            existing_anns = self._get_annotations(rec["id"])  # refresh for next dup check
            added += 1

        self._autosave()
        self._refresh_ann_list()
        self._render()
        msg = f"YOLO+SAM: {added} polygons added"
        if skipped: msg += f", {skipped} skipped"
        if failed:  msg += f", {failed} SAM failed"
        self._set_status(msg)

    def _batch_pre_annotate(self, conf_threshold=0.35):
        """Run YOLO+SAM pre-annotation across all unannotated images in the dataset."""
        import cv2

        best_pt = _find_best_pt()
        if not best_pt:
            self._set_status("best.pt not found — train a model first.")
            return

        predictor = _get_sam_predictor()
        if predictor is None:
            self._set_status("SAM model not found — check models/sam_vit_b.pth.")
            return

        # Find all unannotated images
        unannotated = []
        for fname in self.image_files:
            rec = self._get_image_record(fname)
            if not rec or not self._get_annotations(rec["id"]):
                unannotated.append(fname)

        if not unannotated:
            self._set_status("All images already have annotations.")
            return

        self._set_status(f"Batch: loading model for {len(unannotated)} unannotated images...")
        self.root.update_idletasks()

        try:
            from ultralytics import YOLO
            model = YOLO(best_pt)
        except ImportError:
            self._set_status("ultralytics not installed.")
            return
        except Exception as e:
            self._set_status(f"Model load error: {e}")
            return

        cat_name_to_id = {c["name"]: c["id"] for c in self.coco["categories"]}
        total_added = 0
        total_failed = 0

        for i, fname in enumerate(unannotated):
            self._set_status(
                f"Batch: {i+1}/{len(unannotated)} — {fname}  |  {total_added} polygons added so far"
            )
            self.root.update_idletasks()

            # Resolve image source — local file or URL-fetched PIL image
            if self.images_dir:
                full_path = os.path.join(self.images_dir, fname)
                if not os.path.exists(full_path):
                    continue
                pil_src = full_path
            elif fname in self.url_images:
                pil_src = self.url_images[fname]
            elif self.lazy_url_map and fname in self.lazy_url_map:
                # Fetch on demand for batch mode
                try:
                    u = self.lazy_url_map[fname]
                    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    fetched = Image.open(io.BytesIO(data))
                    fetched = ImageOps.exif_transpose(fetched).convert("RGB")
                    self.url_images[fname] = fetched
                    pil_src = fetched
                except Exception as _fe:
                    print(f"[batch] Failed to fetch {fname}: {_fe}")
                    total_failed += 1
                    continue
            else:
                continue

            rec = self._ensure_image_record(fname, 0, 0)
            # Refresh actual image dimensions
            try:
                im = Image.open(pil_src) if isinstance(pil_src, str) else pil_src
                W, H = im.size
                rec["width"]  = W
                rec["height"] = H
            except Exception:
                continue

            # YOLO inference
            try:
                results = model.predict(pil_src, conf=conf_threshold, verbose=False)[0]
            except Exception:
                total_failed += 1
                continue

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                continue

            model_names = results.names

            # SAM embed this image
            try:
                img_arr = np.array(Image.open(full_path).convert("RGB"))
                predictor.set_image(img_arr)
            except Exception:
                total_failed += 1
                continue

            for j in range(len(boxes)):
                cls_idx  = int(boxes.cls[j].item())
                conf     = float(boxes.conf[j].item())
                cls_name = model_names.get(cls_idx, "")
                if cls_name not in cat_name_to_id:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                bw, bh  = x2 - x1, y2 - y1
                if bw < 4 or bh < 4:
                    continue

                try:
                    box_arr = np.array([x1, y1, x2, y2])
                    masks, scores, _ = predictor.predict(
                        box=box_arr[None, :], multimask_output=True
                    )
                    best_mask = masks[np.argmax(scores)]
                    mask_uint8 = (best_mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        continue
                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx  = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) < 3:
                        continue
                    pts_flat = []
                    for pt in approx:
                        px, py = float(pt[0][0]), float(pt[0][1])
                        pts_flat.extend([
                            max(0.0, min(float(W), px)),
                            max(0.0, min(float(H), py)),
                        ])
                    ys, xs = np.where(best_mask)
                    seg_bx = float(xs.min()); seg_by = float(ys.min())
                    seg_bw = float(xs.max() - xs.min())
                    seg_bh = float(ys.max() - ys.min())
                except Exception:
                    continue

                ann_id = max((a["id"] for a in self.coco["annotations"]), default=0) + 1
                self.coco["annotations"].append({
                    "id":           ann_id,
                    "image_id":     rec["id"],
                    "category_id":  cat_name_to_id[cls_name],
                    "segmentation": [pts_flat],
                    "bbox":         [seg_bx, seg_by, seg_bw, seg_bh],
                    "area":         float(seg_bw * seg_bh),
                    "iscrowd":      0,
                    "score":        round(conf, 3),
                })
                total_added += 1

            # Save every 10 images so progress isn't lost if cancelled
            if (i + 1) % 10 == 0:
                self._autosave()

        self._autosave()
        self._refresh_ann_list()
        self._refresh_image_browser()
        self._render()
        self._set_status(
            f"Batch complete: {total_added} polygons across {len(unannotated)} images"
            + (f"  ({total_failed} failed)" if total_failed else "")
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_annotated_image(self):
        """Render current image with annotation overlays at full resolution and save to reports/."""
        if not self.image_files:
            return
        img_path = self.image_files[self.current_idx]
        rec = self._get_image_record(img_path)
        if not rec:
            self._set_status("No record for this image.")
            return

        base = self._get_display_image().convert("RGBA")
        W, H = base.size

        # Try to load a reasonably sized font
        font_size = max(14, W // 60)
        font = None
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        # Two overlay layers: fill (semi-transparent) and lines (opaque)
        fill_layer    = Image.new("RGBA", base.size, (0, 0, 0, 0))
        outline_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        fill_draw    = ImageDraw.Draw(fill_layer)
        outline_draw = ImageDraw.Draw(outline_layer)
        OUTLINE_W = max(3, W // 400)  # scales with image resolution

        anns = self._get_annotations(rec["id"])
        label_jobs = []  # deferred so labels always render on top

        for ann in anns:
            name = self._cat_name(ann["category_id"])
            hex_c = self.color_map.get(name, "#ffffff")
            r = int(hex_c[1:3], 16)
            g = int(hex_c[3:5], 16)
            b = int(hex_c[5:7], 16)
            fill_rgba = (r, g, b, 90)

            seg = ann.get("segmentation")
            if seg and seg[0] and len(seg[0]) >= 6:
                pts = seg[0]
                poly = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]

                # Semi-transparent fill
                fill_draw.polygon(poly, fill=fill_rgba)

                # Thick outline: dark shadow pass then color pass
                closed = poly + [poly[0]]
                outline_draw.line(closed, fill=(0, 0, 0, 200), width=OUTLINE_W + 2)
                outline_draw.line(closed, fill=(r, g, b, 255), width=OUTLINE_W)

                lx, ly = int(pts[0]), max(0, int(pts[1]) - font_size - 6)
            else:
                x, y, w, h = ann["bbox"]
                fill_draw.rectangle([x, y, x + w, y + h], fill=fill_rgba)

                # Dark shadow rect then color rect
                outline_draw.rectangle([x, y, x + w, y + h],
                                       outline=(0, 0, 0, 200), width=OUTLINE_W + 2)
                outline_draw.rectangle([x, y, x + w, y + h],
                                       outline=(r, g, b, 255), width=OUTLINE_W)
                lx, ly = int(x), max(0, int(y) - font_size - 6)

            label_jobs.append((lx, ly, name, r, g, b))

        # Composite layers: base → fill → outlines
        composite = Image.alpha_composite(base, fill_layer)
        composite = Image.alpha_composite(composite, outline_layer)

        # Draw labels last so they're always on top
        label_draw = ImageDraw.Draw(composite)
        pad = 4
        for lx, ly, name, r, g, b in label_jobs:
            text = f"  {name}  "
            tb = label_draw.textbbox((lx, ly), text, font=font)
            # Solid colored background pill
            label_draw.rectangle(
                [tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad],
                fill=(r, g, b, 230)
            )
            # Dark shadow text then white text
            label_draw.text((lx + 1, ly + 1), text, font=font, fill=(0, 0, 0, 180))
            label_draw.text((lx, ly), text, font=font, fill=(255, 255, 255, 255))

        out_img = composite.convert("RGB")

        # Save to projects/<project>/reports/annotated/
        reports_dir = os.path.join(os.path.dirname(self.json_path), "..", "..", "reports", "annotated")
        os.makedirs(reports_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(reports_dir, f"{stem}_annotated.png")
        out_img.save(out_path)
        self._set_status(f"Saved → {os.path.relpath(out_path)}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def _fetch_url_images(urls):
    """
    Fetch a list of image URLs into memory.
    Returns (url_images, url_map):
        url_images : {filename: PIL.Image}
        url_map    : {filename: original_url}
    Images are never written to disk.
    """
    url_images = {}
    url_map    = {}
    for url in urls:
        try:
            print(f"[annotate] Fetching {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "onesvs-annotator/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            from urllib.parse import urlparse, unquote
            filename = os.path.basename(unquote(urlparse(url).path)) or "image.jpg"
            # Ensure unique filename if multiple URLs share the same basename
            base, ext = os.path.splitext(filename)
            counter = 1
            while filename in url_images:
                filename = f"{base}_{counter}{ext}"
                counter += 1
            pil = ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(raw)).convert("RGB"))
            url_images[filename] = pil
            url_map[filename]    = url
            print(f"[annotate] Loaded {filename} ({pil.size[0]}x{pil.size[1]})")
        except Exception as e:
            print(f"[annotate] WARNING: Could not fetch {url}: {e}")
    return url_images, url_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None,
                        help="Project name under projects/")
    parser.add_argument("--images", default=None,
                        help="Direct path to images directory")
    parser.add_argument("--json", default=None,
                        help="Direct path to COCO JSON file")
    parser.add_argument("--url", action="append", default=[],
                        metavar="URL",
                        help="Cloud image URL to annotate (can be repeated for multiple images). "
                             "Requires --json. Images are never written to disk.")
    parser.add_argument("--index", default=None,
                        metavar="INDEX_CSV",
                        help="Path to index.csv — loads ALL URLs from the index as the dataset. "
                             "Requires --json. Use --status to filter by status.")
    parser.add_argument("--status", default="all",
                        metavar="STATUS",
                        help="Filter index by status when using --index. "
                             "Values: all | pending | approved | rejected (default: all)")
    args = parser.parse_args()

    url_images = {}
    url_map    = {}

    if args.project:
        project_dir = os.path.join("projects", args.project)
        images_dir = os.path.join(project_dir, "datasets", "raw")
        json_path = os.path.join(project_dir, "datasets", "raw", "labels.json")

        # Use existing COCO JSON if present (e.g. labels_svs_combined.json)
        if not os.path.exists(json_path):
            for f in os.listdir(images_dir) if os.path.exists(images_dir) else []:
                if f.endswith(".json"):
                    json_path = os.path.join(images_dir, f)
                    break

        # Load classes from COCO JSON (source of truth)
        class_names = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                coco_tmp = json.load(f)
            class_names = [c["name"] for c in
                           sorted(coco_tmp.get("categories", []), key=lambda x: x["id"])]

    elif args.index and args.json:
        # Index mode — read all URLs from index.csv, optionally filtered by status
        import csv as _csv
        json_path  = args.json
        images_dir = None
        urls_from_index = []
        status_filter = args.status.strip().lower()
        try:
            with open(args.index, newline="", encoding="utf-8") as _f:
                for _row in _csv.DictReader(_f):
                    url = (_row.get("url") or "").strip()
                    status = (_row.get("status") or "pending").strip().lower()
                    if not url:
                        continue
                    if status_filter == "all" or status == status_filter:
                        urls_from_index.append(url)
        except FileNotFoundError:
            print(f"Index file not found: {args.index}")
            return
        if not urls_from_index:
            print(f"No URLs found in index (status filter: {status_filter}).")
            return
        print(f"[index] {len(urls_from_index)} image(s) from index (status={status_filter}) — lazy load")
        # Build url_map and url_images lazily: filenames derived from URLs,
        # actual PIL images fetched on demand in _load_image
        from urllib.parse import urlparse as _urlparse, unquote as _unquote
        url_map    = {}
        url_images = {}   # starts empty — populated on demand
        _lazy_url_map = {}  # filename → url, for on-demand fetching
        for _u in urls_from_index:
            _fname = os.path.basename(_unquote(_urlparse(_u).path)) or "image.jpg"
            _base, _ext = os.path.splitext(_fname)
            _counter = 1
            while _fname in _lazy_url_map:
                _fname = f"{_base}_{_counter}{_ext}"
                _counter += 1
            url_map[_fname]      = _u
            _lazy_url_map[_fname] = _u
        class_names = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                coco = json.load(f)
            class_names = [c["name"] for c in
                           sorted(coco.get("categories", []), key=lambda x: x["id"])]

    elif args.url and args.json:
        # URL mode — fetch images into memory, no local images_dir needed
        json_path  = args.json
        images_dir = None
        url_images, url_map = _fetch_url_images(args.url)
        if not url_images:
            print("No images could be fetched from the provided URLs.")
            return
        class_names = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                coco = json.load(f)
            class_names = [c["name"] for c in
                           sorted(coco.get("categories", []), key=lambda x: x["id"])]

    elif args.images and args.json:
        images_dir = args.images
        json_path = args.json
        class_names = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                coco = json.load(f)
            class_names = [c["name"] for c in
                           sorted(coco.get("categories", []), key=lambda x: x["id"])]
    else:
        print("Usage:")
        print("  python scripts/annotate.py --project svs_plumbing")
        print("  python scripts/annotate.py --images /path/imgs --json /path/labels.json")
        print("  python scripts/annotate.py --url https://example.com/photo.jpg --json /path/labels.json")
        print("  python scripts/annotate.py --index /path/index.csv --json /path/labels.json")
        print("  python scripts/annotate.py --index /path/index.csv --json /path/labels.json --status pending")
        return

    # If class_names is empty or only contains a placeholder, load from model
    _placeholder = len(class_names) <= 1 and (not class_names or class_names[0] in ("object", ""))
    if not class_names or _placeholder:
        best_pt = _find_best_pt()
        if best_pt:
            try:
                from ultralytics import YOLO as _YOLO
                _m = _YOLO(best_pt)
                class_names = [_m.names[i] for i in sorted(_m.names.keys())]
                print(f"[classes] Loaded {len(class_names)} classes from model: {best_pt}")
            except Exception as _e:
                print(f"[classes] Could not load from model: {_e}")
        if not class_names:
            class_names = ["object"]

    # In lazy-index mode url_map is populated but url_images starts empty
    _is_lazy = bool(url_map and not url_images)
    if not url_images and not _is_lazy and (not images_dir or not os.path.exists(images_dir)):
        print(f"Images directory not found: {images_dir}")
        return

    root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(900, 600)
    root.deiconify()
    root.lift()
    root.focus_force()
    app = AnnotationTool(root, images_dir, json_path, class_names,
                         project=args.project,
                         url_images=url_images,
                         url_map=url_map,
                         lazy_url_map=_lazy_url_map if _is_lazy else None)
    root.mainloop()


if __name__ == "__main__":
    main()
