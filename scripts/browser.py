"""
ML Data Browser — onesvs Continuous Learning
Reads index.csv + annotations.json, displays cloud images by URL.
Images are never downloaded to disk — thumbnails are fetched into memory on demand.

Usage:
    python scripts/scripts/browser.py
    python scripts/scripts/browser.py --index /path/to/index.csv --json /path/to/annotations.json

index.csv columns:
    image_id, company_id, company_name, job_id, url, status, annotated_by, date, notes

status values: pending | approved | rejected
"""

import argparse
import csv
import io
import json
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from urllib.parse import urlparse, unquote
import urllib.request

from PIL import Image, ImageOps, ImageTk

# ── Defaults ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.path.join(_SCRIPT_DIR, "..", "..", "data", "index.csv")
DEFAULT_JSON  = os.path.join(_SCRIPT_DIR, "..", "..", "data", "annotations.json")
ANNOTATOR_PY  = os.path.join(_SCRIPT_DIR, "annotate.py")
PYTHON_EXE    = sys.executable

# ── Colors (match annotate.py palette) ───────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_MID    = "#16213e"
BG_PANEL  = "#0f3460"
ACCENT    = "#00b4d8"
GREEN     = "#06d6a0"
YELLOW    = "#f39c12"
RED       = "#ff4444"
WHITE     = "#ffffff"
GRAY      = "#aaaaaa"
RUN_COLOR = "#a855f7"   # purple — model inference

STATUS_COLORS = {
    "approved": GREEN,
    "rejected": RED,
    "pending":  YELLOW,
}

RESULT_COLORS = {
    "BAD":       "#ff4444",
    "UNCERTAIN": "#f39c12",
    "CONFLICT":  "#ff8c00",
    "GOOD":      "#06d6a0",
    "CLEANING":  "#00b4d8",
    "NOISE":     "#666688",
    "MISSING":   "#444466",
}

RESULT_TO_STATUS = {
    "BAD":      "pending",
    "UNCERTAIN":"pending",
    "CONFLICT": "pending",
    "GOOD":     "approved",
    "CLEANING": "rejected",
    "NOISE":    "rejected",
    "MISSING":  "rejected",
}

THUMB_SIZE = (280, 210)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_index(path):
    """Load index.csv → list of dicts. Creates the file with headers if missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_id", "company_id", "company_name",
                "job_id", "url", "status", "annotated_by", "date", "notes"
            ])
            writer.writeheader()
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_index(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_annotations(path):
    """Load annotations.json (COCO-style). Returns dict keyed by image_id (str)."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)
    ann_map = {}
    for img in coco.get("images", []):
        iid = str(img["id"])
        anns = [a for a in coco.get("annotations", [])
                if str(a["image_id"]) == iid]
        ann_map[iid] = {"image": img, "annotations": anns}
    return ann_map


def export_dataset(rows, ann_map, out_path, status_filter="approved"):
    """
    Export a COCO JSON with URL references for selected rows.
    Images are referenced by URL — Colab fetches them at train time.
    """
    filtered = [r for r in rows if r.get("status") == status_filter]
    images, annotations, cat_set = [], [], {}

    for row in filtered:
        iid = str(row["image_id"])
        entry = ann_map.get(iid, {})
        img_rec = entry.get("image", {
            "id": int(row["image_id"]),
            "file_name": os.path.basename(unquote(urlparse(row["url"]).path)),
            "url": row["url"],
            "width": 0, "height": 0,
        })
        images.append(img_rec)
        for ann in entry.get("annotations", []):
            annotations.append(ann)
            cat_set[ann["category_id"]] = ann.get("category_name", str(ann["category_id"]))

    categories = [{"id": k, "name": v} for k, v in sorted(cat_set.items())]
    coco = {
        "info": {
            "description": "onesvs ML Dataset Export",
            "date_created": datetime.utcnow().isoformat() + "Z",
            "status_filter": status_filter,
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return len(images)


# ── Thumbnail fetcher (background thread) ─────────────────────────────────────

def fetch_thumbnail(url, callback):
    """Fetch image from URL in a background thread, call callback(PhotoImage | None)."""
    def _run():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "onesvs-browser/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
            pil = ImageOps.exif_transpose(
                Image.open(io.BytesIO(raw)).convert("RGB")
            )
            pil.thumbnail(THUMB_SIZE, Image.LANCZOS)
            callback(pil)
        except Exception as e:
            print(f"[browser] Thumbnail fetch failed: {e}")
            callback(None)
    threading.Thread(target=_run, daemon=True).start()


# ── Main App ──────────────────────────────────────────────────────────────────

class DataBrowser:
    def __init__(self, root, index_path, json_path):
        self.root       = root
        self.index_path = index_path
        self.json_path  = json_path

        self.root.title("ML Data Browser — onesvs Continuous Learning")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x760")
        self.root.minsize(960, 600)

        self.all_rows    = load_index(index_path)
        self.ann_map     = load_annotations(json_path)
        self.filtered    = []
        self._thumb_img  = None   # keep PhotoImage reference alive
        self._fetch_url  = None   # URL currently being fetched

        self._build_ui()
        self._apply_filters()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, minsize=220)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, minsize=300)
        self.root.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        # Scrollable left panel
        outer = tk.Frame(self.root, bg=BG_MID, width=232)
        outer.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        outer.grid_propagate(False)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        _canvas = tk.Canvas(outer, bg=BG_MID, highlightthickness=0, width=220)
        _vsb    = ttk.Scrollbar(outer, orient="vertical", command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _canvas.grid(row=0, column=0, sticky="nsew")
        _vsb.grid(row=0, column=1, sticky="ns")

        left = tk.Frame(_canvas, bg=BG_MID, width=220)
        _win_id = _canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_frame_configure(e):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
        def _on_canvas_configure(e):
            _canvas.itemconfig(_win_id, width=e.width)
        left.bind("<Configure>", _on_frame_configure)
        _canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        left.bind_all("<MouseWheel>", _on_mousewheel)

        tk.Label(left, text="ML Data Browser", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 13, "bold")).pack(pady=(14, 2), padx=10, anchor="w")
        tk.Label(left, text="onesvs Continuous Learning", bg=BG_MID, fg=GRAY,
                 font=("Helvetica", 9)).pack(padx=10, anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10, padx=8)

        # ── Filters ──
        tk.Label(left, text="FILTERS", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(padx=10, anchor="w")

        # Company filter
        tk.Label(left, text="Company", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10)).pack(padx=10, anchor="w", pady=(8, 2))
        self.company_var = tk.StringVar(value="All")
        self.company_cb  = ttk.Combobox(left, textvariable=self.company_var,
                                        state="readonly", width=22)
        self.company_cb.pack(padx=10, fill=tk.X)
        self.company_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Status filter
        tk.Label(left, text="Status", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10)).pack(padx=10, anchor="w", pady=(10, 2))
        self.status_var = tk.StringVar(value="All")
        status_cb = ttk.Combobox(left, textvariable=self.status_var, state="readonly",
                                 values=["All", "pending", "approved", "rejected"], width=22)
        status_cb.pack(padx=10, fill=tk.X)
        status_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Search by job ID
        tk.Label(left, text="Job ID", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10)).pack(padx=10, anchor="w", pady=(10, 2))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filters())
        tk.Entry(left, textvariable=self.search_var, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat",
                 font=("Helvetica", 10)).pack(padx=10, fill=tk.X)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=12, padx=8)

        # ── Dataset Builder ──
        tk.Label(left, text="DATASET BUILDER", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(padx=10, anchor="w")

        tk.Button(left, text="🔌  Query Database",
                  bg=GREEN, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2", pady=8,
                  command=self._open_db_query).pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Button(left, text="🔍  Search CSV",
                  bg=ACCENT, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2", pady=8,
                  command=self._open_search).pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Button(left, text="🤖  Run Model Report",
                  bg=RUN_COLOR, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2", pady=8,
                  command=self._open_model_report).pack(fill=tk.X, padx=10, pady=(0, 4))

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=12, padx=8)

        # ── Actions ──
        tk.Label(left, text="ACTIONS", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(padx=10, anchor="w")

        btn_cfg = dict(relief="flat", font=("Helvetica", 10, "bold"),
                       cursor="hand2", bd=0, pady=8)

        tk.Button(left, text="👁  Review Photos",
                  bg="#e040fb", fg="#000", command=self._open_review,
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Button(left, text="✎  Open in Annotator",
                  bg=YELLOW, fg="#000", command=self._open_in_annotator,
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Button(left, text="📋  Open Index in Annotator",
                  bg="#e67e22", fg="#000", command=self._open_index_in_annotator,
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Button(left, text="✓  Approve",
                  bg=GREEN, fg="#000", command=lambda: self._set_status("approved"),
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=4)

        tk.Button(left, text="✗  Reject",
                  bg=RED, fg=WHITE, command=lambda: self._set_status("rejected"),
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=4)

        tk.Button(left, text="○  Mark Pending",
                  bg=BG_PANEL, fg=WHITE, command=lambda: self._set_status("pending"),
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=12, padx=8)

        tk.Button(left, text="⬇  Export Annotations (JSON)",
                  bg=ACCENT, fg="#000", command=self._export_dataset,
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=4)

        tk.Button(left, text="⬇  Export Index (CSV)",
                  bg=BG_PANEL, fg=WHITE, command=self._export_csv,
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Button(left, text="＋  Add URL",
                  bg=BG_PANEL, fg=WHITE, command=self._add_url_dialog,
                  **btn_cfg).pack(fill=tk.X, padx=10, pady=(4, 0))

        # ── Count label ──
        self.count_label = tk.Label(left, text="", bg=BG_MID, fg=GRAY,
                                    font=("Helvetica", 9))
        self.count_label.pack(side=tk.BOTTOM, pady=8)

    def _build_center_panel(self):
        center = tk.Frame(self.root, bg=BG_DARK)
        center.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        # Header
        tk.Label(center, text="Image Index", bg=BG_DARK, fg=WHITE,
                 font=("Helvetica", 12, "bold")).grid(row=0, column=0,
                                                       sticky="w", padx=4, pady=(0, 6))

        # Treeview
        cols = ("image_id", "company", "job_id", "date", "status", "annotations", "notes")
        self.tree = ttk.Treeview(center, columns=cols, show="headings",
                                 selectmode="extended")

        col_cfg = [
            ("image_id",    "ID",          60,  tk.CENTER),
            ("company",     "Company",     130, tk.W),
            ("job_id",      "Job ID",      110, tk.W),
            ("date",        "Date",        100, tk.CENTER),
            ("status",      "Status",      90,  tk.CENTER),
            ("annotations", "Annotations", 90,  tk.CENTER),
            ("notes",       "Notes",       160, tk.W),
        ]
        for cid, heading, width, anchor in col_cfg:
            self.tree.heading(cid, text=heading,
                              command=lambda c=cid: self._sort_by(c))
            self.tree.column(cid, width=width, anchor=anchor, stretch=(cid == "notes"))

        # Scrollbars
        vsb = ttk.Scrollbar(center, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(center, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self._open_in_annotator())

        # Style tag colors — status
        self.tree.tag_configure("approved", foreground=GREEN)
        self.tree.tag_configure("rejected", foreground=RED)
        self.tree.tag_configure("pending",  foreground=YELLOW)
        # Style tag colors — model results
        for result, color in RESULT_COLORS.items():
            self.tree.tag_configure(f"result_{result}", foreground=color)

    def _build_right_panel(self):
        right = tk.Frame(self.root, bg=BG_MID, width=300)
        right.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_propagate(False)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="Preview", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 11, "bold")).grid(
            row=0, column=0, pady=(12, 6), padx=10, sticky="w")

        # Thumbnail canvas
        self.thumb_canvas = tk.Canvas(right, bg=BG_PANEL, width=280, height=210,
                                      highlightthickness=0)
        self.thumb_canvas.grid(row=1, column=0, padx=10, pady=4, sticky="n")
        self._thumb_placeholder()

        # Detail labels
        detail_frame = tk.Frame(right, bg=BG_MID)
        detail_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(8, 4))
        detail_frame.columnconfigure(1, weight=1)

        self.detail_vars = {}
        detail_fields = [
            ("image_id",    "ID"),
            ("company",     "Company"),
            ("job_id",      "Job ID"),
            ("date",        "Date"),
            ("status",      "Status"),
            ("annotations", "Annotations"),
            ("url",         "URL"),
        ]
        for i, (key, label) in enumerate(detail_fields):
            tk.Label(detail_frame, text=label + ":", bg=BG_MID, fg=GRAY,
                     font=("Helvetica", 9, "bold"), anchor="w").grid(
                row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            self.detail_vars[key] = var
            color = ACCENT if key == "url" else WHITE
            tk.Label(detail_frame, textvariable=var, bg=BG_MID, fg=color,
                     font=("Helvetica", 9), anchor="w", wraplength=180,
                     justify=tk.LEFT).grid(row=i, column=1, sticky="w", pady=2, padx=(6, 0))

        # URL copy button
        tk.Button(right, text="Copy URL", bg=BG_PANEL, fg=WHITE,
                  relief="flat", font=("Helvetica", 9), cursor="hand2",
                  command=self._copy_url).grid(
            row=3, column=0, padx=10, pady=(0, 4), sticky="w")

        # Status bar
        self.status_bar = tk.Label(right, text="", bg=BG_MID, fg=GRAY,
                                   font=("Helvetica", 9), wraplength=280, justify=tk.LEFT)
        self.status_bar.grid(row=4, column=0, padx=10, pady=(4, 8), sticky="w")

    # ── Filter & Display ─────────────────────────────────────────────────────

    def _apply_filters(self):
        company = self.company_var.get()
        status  = self.status_var.get()
        search  = self.search_var.get().strip().lower()

        self.filtered = [
            r for r in self.all_rows
            if (company == "All" or r.get("company_name", "") == company)
            and (status == "All" or r.get("status", "pending") == status)
            and (not search or search in str(r.get("job_id", "")).lower())
        ]

        # Refresh company dropdown options
        companies = sorted(set(r.get("company_name", "") for r in self.all_rows if r.get("company_name")))
        self.company_cb["values"] = ["All"] + companies

        self._refresh_tree()
        self.count_label.config(
            text=f"{len(self.filtered)} of {len(self.all_rows)} images")

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for row in self.filtered:
            iid    = str(row.get("image_id", ""))
            anns   = len(self.ann_map.get(iid, {}).get("annotations", []))
            status = row.get("status", "pending")
            notes  = row.get("notes", "")
            # Approved/rejected rows always use status color; pending rows use model result color
            if status in ("approved", "rejected"):
                result_tag = status
            else:
                result_tag = next(
                    (f"result_{r}" for r in RESULT_COLORS if notes.upper().startswith(r)),
                    status
                )
            self.tree.insert("", tk.END, iid=iid, tags=(result_tag,), values=(
                iid,
                row.get("company_name", ""),
                row.get("job_id", ""),
                row.get("date", ""),
                status,
                anns if anns else "—",
                notes,
            ))

    def _sort_by(self, col):
        reverse = getattr(self, "_sort_reverse", False)
        self.filtered.sort(key=lambda r: str(r.get(
            {"image_id": "image_id", "company": "company_name",
             "job_id": "job_id", "date": "date", "status": "status"}.get(col, col), ""
        )).lower(), reverse=reverse)
        self._sort_reverse = not reverse
        self._refresh_tree()

    # ── Selection & Preview ──────────────────────────────────────────────────

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        row = next((r for r in self.filtered if str(r.get("image_id")) == iid), None)
        if not row:
            return

        anns = len(self.ann_map.get(iid, {}).get("annotations", []))
        self.detail_vars["image_id"].set(row.get("image_id", ""))
        self.detail_vars["company"].set(row.get("company_name", ""))
        self.detail_vars["job_id"].set(row.get("job_id", ""))
        self.detail_vars["date"].set(row.get("date", ""))
        self.detail_vars["status"].set(row.get("status", "pending"))
        self.detail_vars["annotations"].set(str(anns) if anns else "None")
        url = row.get("url", "")
        self.detail_vars["url"].set(url[:60] + "..." if len(url) > 60 else url)

        # Fetch thumbnail in background
        if url and url != self._fetch_url:
            self._fetch_url = url
            self._thumb_placeholder("Loading...")
            fetch_thumbnail(url, self._on_thumbnail_ready)

    def _on_thumbnail_ready(self, pil_image):
        """Called from background thread — must schedule UI update on main thread."""
        self.root.after(0, lambda: self._show_thumbnail(pil_image))

    def _show_thumbnail(self, pil_image):
        self.thumb_canvas.delete("all")
        if pil_image is None:
            self._thumb_placeholder("Could not load image")
            return
        self._thumb_img = ImageTk.PhotoImage(pil_image)
        cw = self.thumb_canvas.winfo_width()  or THUMB_SIZE[0]
        ch = self.thumb_canvas.winfo_height() or THUMB_SIZE[1]
        x = (cw - pil_image.width)  // 2
        y = (ch - pil_image.height) // 2
        self.thumb_canvas.create_image(x, y, anchor=tk.NW, image=self._thumb_img)

    def _thumb_placeholder(self, msg="Select an image"):
        self.thumb_canvas.delete("all")
        self._thumb_img = None
        self.thumb_canvas.create_text(
            THUMB_SIZE[0] // 2, THUMB_SIZE[1] // 2,
            text=msg, fill=GRAY, font=("Helvetica", 11), justify=tk.CENTER
        )

    # ── Actions ──────────────────────────────────────────────────────────────

    def _get_selected_rows(self):
        return [r for r in self.filtered
                if str(r.get("image_id")) in self.tree.selection()]

    def _open_in_annotator(self):
        rows = self._get_selected_rows()
        if not rows:
            messagebox.showinfo("No selection", "Select one or more images first.")
            return
        urls = [r["url"] for r in rows if r.get("url")]
        if not urls:
            messagebox.showerror("No URL", "Selected images have no URL.")
            return
        cmd = [PYTHON_EXE, ANNOTATOR_PY] + [arg for u in urls for arg in ("--url", u)] + \
              ["--json", self.json_path]
        try:
            subprocess.Popen(cmd, start_new_session=True)
            self._set_status_bar(f"Opened {len(urls)} image(s) in annotator.")
        except Exception as e:
            messagebox.showerror("Launch error", str(e))

    def _open_index_in_annotator(self):
        if not self.all_rows:
            messagebox.showinfo("Empty index", "No images in the index yet.")
            return
        # Ask which status to load
        win = tk.Toplevel(self.root)
        win.title("Open Index in Annotator")
        win.configure(bg=BG_DARK)
        win.geometry("320x180")
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text="Which images to open?", bg=BG_DARK, fg=WHITE,
                 font=("Helvetica", 11, "bold")).pack(pady=(18, 8))

        status_var = tk.StringVar(value="all")
        for label, val in [("All images", "all"), ("Pending only", "pending"),
                           ("Approved only", "approved"), ("Rejected only", "rejected")]:
            tk.Radiobutton(win, text=label, variable=status_var, value=val,
                           bg=BG_DARK, fg=WHITE, selectcolor=BG_PANEL,
                           activebackground=BG_DARK,
                           font=("Helvetica", 10)).pack(anchor="w", padx=30)

        def _launch():
            win.destroy()
            cmd = [PYTHON_EXE, ANNOTATOR_PY,
                   "--index", self.index_path,
                   "--json",  self.json_path,
                   "--status", status_var.get()]
            try:
                subprocess.Popen(cmd, start_new_session=True)
                count = sum(1 for r in self.all_rows
                            if status_var.get() == "all"
                            or r.get("status") == status_var.get())
                self._set_status_bar(f"Opening {count} image(s) from index in annotator…")
            except Exception as e:
                messagebox.showerror("Launch error", str(e))

        tk.Button(win, text="Open in Annotator", bg=YELLOW, fg="#000",
                  relief="flat", font=("Helvetica", 10, "bold"),
                  cursor="hand2", command=_launch).pack(pady=14)

    def _set_status(self, status):
        rows = self._get_selected_rows()
        if not rows:
            messagebox.showinfo("No selection", "Select one or more images first.")
            return
        for row in rows:
            row["status"] = status
        try:
            save_index(self.index_path, self.all_rows)
        except PermissionError:
            messagebox.showerror("File Locked",
                                 "Cannot save index.csv — close it in Excel or any other "
                                 "program, then try again.")
            return
        self._apply_filters()
        self._set_status_bar(f"{len(rows)} image(s) marked as {status}.")

    def _copy_url(self):
        sel = self.tree.selection()
        if not sel:
            return
        row = next((r for r in self.filtered if str(r.get("image_id")) == sel[0]), None)
        if row and row.get("url"):
            self.root.clipboard_clear()
            self.root.clipboard_append(row["url"])
            self._set_status_bar("URL copied to clipboard.")

    def _export_dataset(self):
        approved = [r for r in self.all_rows if r.get("status") == "approved"]
        if not approved:
            messagebox.showinfo("Nothing to export",
                                "No approved images found. Approve images before exporting.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="dataset_export.json",
            title="Save Dataset Export",
        )
        if not out:
            return
        count = export_dataset(self.all_rows, self.ann_map, out)
        messagebox.showinfo("Export complete",
                            f"Exported {count} approved images to:\n{out}")
        self._set_status_bar(f"Dataset exported — {count} images.")

    def _export_csv(self):
        """Export the current filtered index view to a CSV — no annotations included."""
        rows = self.filtered if self.filtered else self.all_rows
        if not rows:
            messagebox.showinfo("Nothing to export", "No images in the current view.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="index_export.csv",
            title="Export Index to CSV",
        )
        if not out:
            return
        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            self._set_status_bar(f"Exported {len(rows)} row(s) to {os.path.basename(out)}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _add_url_dialog(self):
        """Dialog to manually add a new image URL to the index."""
        win = tk.Toplevel(self.root)
        win.title("Add Image URL")
        win.configure(bg=BG_DARK)
        win.geometry("520x340")
        win.grab_set()

        fields = [
            ("URL",          "url",          "https://"),
            ("Company ID",   "company_id",   ""),
            ("Company Name", "company_name", ""),
            ("Job ID",       "job_id",       ""),
            ("Notes",        "notes",        ""),
        ]
        entries = {}
        for i, (label, key, default) in enumerate(fields):
            tk.Label(win, text=label, bg=BG_DARK, fg=WHITE,
                     font=("Helvetica", 10)).grid(row=i, column=0,
                                                   padx=16, pady=6, sticky="w")
            var = tk.StringVar(value=default)
            tk.Entry(win, textvariable=var, bg=BG_MID, fg=WHITE,
                     insertbackground=WHITE, relief="flat", width=40,
                     font=("Helvetica", 10)).grid(row=i, column=1,
                                                   padx=8, pady=6, sticky="ew")
            entries[key] = var
        win.columnconfigure(1, weight=1)

        def _save():
            url = entries["url"].get().strip()
            if not url or not url.startswith("http"):
                messagebox.showerror("Invalid URL", "Please enter a valid http(s) URL.", parent=win)
                return
            new_id = max((int(r.get("image_id", 0)) for r in self.all_rows), default=0) + 1
            self.all_rows.append({
                "image_id":    new_id,
                "company_id":  entries["company_id"].get().strip(),
                "company_name": entries["company_name"].get().strip(),
                "job_id":      entries["job_id"].get().strip(),
                "url":         url,
                "status":      "pending",
                "annotated_by": "",
                "date":        datetime.utcnow().strftime("%Y-%m-%d"),
                "notes":       entries["notes"].get().strip(),
            })
            save_index(self.index_path, self.all_rows)
            self._apply_filters()
            self._set_status_bar(f"Added image {new_id}.")
            win.destroy()

        tk.Button(win, text="Add to Index", bg=ACCENT, fg="#000",
                  relief="flat", font=("Helvetica", 10, "bold"),
                  cursor="hand2", command=_save).grid(
            row=len(fields), column=0, columnspan=2, pady=14, padx=16, sticky="ew")

    def _set_status_bar(self, msg):
        self.status_bar.config(text=msg)
        self.root.after(5000, lambda: self.status_bar.config(text=""))

    # ── CSV Search Engine ─────────────────────────────────────────────────────

    def _open_review(self):
        rows = self.filtered if self.filtered else self.all_rows
        if not rows:
            messagebox.showinfo("No images", "No images to review.")
            return
        # Start from selected row if any, else row 0
        sel = self.tree.selection()
        start = 0
        if sel:
            ids = [str(r.get("image_id")) for r in rows]
            if sel[0] in ids:
                start = ids.index(sel[0])
        PhotoReviewWindow(self.root, rows, start, self._on_review_save)

    def _on_review_save(self, updated_rows):
        # updated_rows are the same dict objects — just save and refresh
        try:
            save_index(self.index_path, self.all_rows)
        except PermissionError:
            messagebox.showerror("File Locked",
                                 "Cannot save — close index.csv in Excel and try again.")
        self._apply_filters()

    def _open_model_report(self):
        """Open the model report window."""
        if not self.all_rows:
            messagebox.showinfo("No images",
                                "Add images to the index first using Query Database or Search CSV.")
            return
        ModelReportWindow(self.root, self.index_path, self.all_rows,
                          on_complete=self._on_report_complete)

    def _on_report_complete(self, updated_rows, counts):
        self.all_rows = updated_rows
        try:
            save_index(self.index_path, self.all_rows)
        except PermissionError:
            messagebox.showwarning("File Locked",
                                   "Report results are loaded but index.csv could not be saved "
                                   "— close it in Excel, then use Export Index to save.")
        self._apply_filters()
        summary = "  ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
        self._set_status_bar(f"Report complete — {summary}")

    def _open_db_query(self):
        """Open the database query window."""
        DatabaseQueryWindow(self.root, self.index_path, self.all_rows,
                            on_added=self._on_search_added)

    def _open_search(self):
        """Open the dataset builder search window."""
        SearchWindow(self.root, self.index_path, self.all_rows,
                     on_added=self._on_search_added)

    def _on_search_added(self, count):
        """Called when search results are added to the index."""
        self.all_rows = load_index(self.index_path)
        self._apply_filters()
        self._set_status_bar(f"Added {count} image(s) to index from search.")


# ── Photo Review Window ───────────────────────────────────────────────────────

class PhotoReviewWindow:
    """
    Lightbox-style photo review.
    Scroll through all images in the current filtered list, see the actual photo,
    and approve / reject / mark pending with one click or keyboard shortcut.

    Keyboard shortcuts:
        ← / →       previous / next
        A           approve
        R           reject
        P           pending
        Escape      close
    """

    IMG_W = 900
    IMG_H = 620

    def __init__(self, parent, rows, start_index, on_save):
        self.rows     = rows          # shared dicts from DataBrowser
        self.idx      = start_index
        self.on_save  = on_save
        self._tk_img  = None
        self._thread  = None

        self.win = tk.Toplevel(parent)
        self.win.title("👁  Review Photos")
        self.win.configure(bg=BG_DARK)
        self.win.geometry(f"{self.IMG_W + 40}x{self.IMG_H + 180}")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()
        self.win.bind("<Left>",   lambda e: self._step(-1))
        self.win.bind("<Right>",  lambda e: self._step(1))
        self.win.bind("<a>",      lambda e: self._set("approved"))
        self.win.bind("<A>",      lambda e: self._set("approved"))
        self.win.bind("<r>",      lambda e: self._set("rejected"))
        self.win.bind("<R>",      lambda e: self._set("rejected"))
        self.win.bind("<p>",      lambda e: self._set("pending"))
        self.win.bind("<P>",      lambda e: self._set("pending"))
        self.win.bind("<Escape>", lambda e: self._close())

        self._load_current()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=1)

        # ── Top bar: counter + meta ──
        top = tk.Frame(self.win, bg=BG_MID, pady=6)
        top.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        top.columnconfigure(1, weight=1)

        self._counter_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self._counter_var,
                 bg=BG_MID, fg=ACCENT, font=("Helvetica", 11, "bold")).grid(
            row=0, column=0, padx=14, sticky="w")

        self._meta_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self._meta_var,
                 bg=BG_MID, fg=WHITE, font=("Helvetica", 10),
                 anchor="w").grid(row=0, column=1, padx=6, sticky="ew")

        self._result_var = tk.StringVar(value="")
        self._result_lbl = tk.Label(top, textvariable=self._result_var,
                                    bg=BG_MID, font=("Helvetica", 11, "bold"))
        self._result_lbl.grid(row=0, column=2, padx=14, sticky="e")

        # ── Image canvas ──
        self._canvas = tk.Canvas(self.win, bg="#0a0a1a", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)

        self._loading_text = self._canvas.create_text(
            self.IMG_W // 2, self.IMG_H // 2,
            text="Loading…", fill=GRAY, font=("Helvetica", 14))

        # ── Bottom bar: nav + status buttons ──
        bot = tk.Frame(self.win, bg=BG_MID, pady=8)
        bot.grid(row=2, column=0, sticky="ew")

        nav_cfg  = dict(relief="flat", font=("Helvetica", 12, "bold"), cursor="hand2",
                        bg=BG_PANEL, fg=WHITE, padx=18, pady=6)
        stat_cfg = dict(relief="flat", font=("Helvetica", 11, "bold"), cursor="hand2",
                        padx=22, pady=6)

        tk.Button(bot, text="◀  Prev", command=lambda: self._step(-1),
                  **nav_cfg).pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(bot, text="Next  ▶", command=lambda: self._step(1),
                  **nav_cfg).pack(side=tk.LEFT, padx=(4, 20))

        tk.Button(bot, text="✓  Approve  [A]",
                  bg=GREEN, fg="#000", command=lambda: self._set("approved"),
                  **stat_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bot, text="✗  Reject  [R]",
                  bg=RED, fg=WHITE, command=lambda: self._set("rejected"),
                  **stat_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bot, text="○  Pending  [P]",
                  bg=BG_PANEL, fg=WHITE, command=lambda: self._set("pending"),
                  **stat_cfg).pack(side=tk.LEFT, padx=4)

        self._status_lbl = tk.Label(bot, text="", bg=BG_MID, fg=GRAY,
                                    font=("Helvetica", 10))
        self._status_lbl.pack(side=tk.RIGHT, padx=14)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _step(self, delta):
        new = self.idx + delta
        if 0 <= new < len(self.rows):
            self.idx = new
            self._load_current()

    def _load_current(self):
        row = self.rows[self.idx]
        total = len(self.rows)

        # Counter
        self._counter_var.set(f"{self.idx + 1} / {total}")

        # Meta
        parts = [row.get("company_name", ""), row.get("job_id", ""), row.get("date", "")]
        self._meta_var.set("  |  ".join(p for p in parts if p))

        # Result / status badge
        notes  = row.get("notes", "")
        status = row.get("status", "pending")
        result = next((r for r in RESULT_COLORS if notes.upper().startswith(r)), None)
        if result:
            color = RESULT_COLORS[result]
            self._result_var.set(result)
        else:
            color = {
                "approved": GREEN, "rejected": RED
            }.get(status, GRAY)
            self._result_var.set(status.upper())
        self._result_lbl.config(fg=color)

        # Image
        self._canvas.delete("all")
        self._canvas.create_text(
            self._canvas.winfo_width() // 2 or self.IMG_W // 2,
            self._canvas.winfo_height() // 2 or self.IMG_H // 2,
            text="Loading…", fill=GRAY, font=("Helvetica", 14), tags="loading")

        url = row.get("url", "")
        if url:
            t = threading.Thread(target=self._fetch_and_show,
                                 args=(url, self.idx), daemon=True)
            t.start()

    def _fetch_and_show(self, url, expected_idx):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            # Fit to canvas
            cw = self._canvas.winfo_width()  or self.IMG_W
            ch = self._canvas.winfo_height() or self.IMG_H
            img.thumbnail((cw, ch), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(img)
            self.win.after(0, lambda: self._show_image(tk_img, expected_idx))
        except Exception as e:
            self.win.after(0, lambda: self._show_error(str(e), expected_idx))

    def _show_image(self, tk_img, expected_idx):
        if expected_idx != self.idx:
            return  # user navigated away
        self._tk_img = tk_img
        cw = self._canvas.winfo_width()  or self.IMG_W
        ch = self._canvas.winfo_height() or self.IMG_H
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=tk_img)

    def _show_error(self, msg, expected_idx):
        if expected_idx != self.idx:
            return
        self._canvas.delete("all")
        self._canvas.create_text(
            self._canvas.winfo_width() // 2 or self.IMG_W // 2,
            self._canvas.winfo_height() // 2 or self.IMG_H // 2,
            text=f"Could not load image\n{msg}", fill=RED,
            font=("Helvetica", 11), justify=tk.CENTER)

    # ── Status ────────────────────────────────────────────────────────────────

    def _set(self, status):
        row = self.rows[self.idx]
        row["status"] = status
        self.on_save(self.rows)
        color = {
            "approved": GREEN, "rejected": RED
        }.get(status, GRAY)
        self._status_lbl.config(text=f"Marked {status}", fg=color)
        self.win.after(1200, lambda: self._status_lbl.config(text=""))
        # Refresh result badge
        notes  = row.get("notes", "")
        result = next((r for r in RESULT_COLORS if notes.upper().startswith(r)), None)
        if not result:
            badge_color = color
            self._result_var.set(status.upper())
            self._result_lbl.config(fg=badge_color)
        # Auto-advance to next pending
        if status in ("approved", "rejected") and self.idx + 1 < len(self.rows):
            self.win.after(600, lambda: self._step(1))

    def _close(self):
        self.on_save(self.rows)
        self.win.destroy()


# ── Model Report Window ───────────────────────────────────────────────────────

_DEFAULT_DETECTOR = os.path.join(
    "/mnt/c/Users/salbe/Documents/Cube_Seperator-Project",
    "cube_separator_detector.pt"
)
_DEFAULT_QUALITY = os.path.join(
    "/mnt/c/Users/salbe/Documents/Cube_Seperator-Project",
    "cube_separator_quality.pt"
)

SKIP_DESCRIPTIONS = {
    "qr code","qr","qr codes","data plate","data sticker","nameplate","label",
    "serial number","model number","last error code log",
    "last error code log / no applicable","error code","error codes","error log",
    "disconnect","disconnect tag","secured unit",
    "ice thickness probe position after adjustment","ice thickness probe",
    "thickness probe","water line","ice machine hose","hose",
    "before cleaning bin","empty bin after cleaning","bin control","bin",
    "ice in bin","ice level","drop zone wiped down","before cleaning drop zone",
    "unit on","unit off","unit exterior","outside unit","unit front",
    "no head clearance","air filter rinsing","air filter","filter",
    "photo","before cleaning (dirty photo)",
}


class ModelReportWindow:
    """
    Runs the two-stage cube separator report against the current index URLs.
    Images are fetched in memory — never downloaded to disk.
    Results are written back to index.csv (notes = result + confidence).
    """

    def __init__(self, parent, index_path, all_rows, on_complete):
        self.index_path  = index_path
        self.all_rows    = list(all_rows)
        self.on_complete = on_complete
        self._running    = False
        self._stop_flag  = threading.Event()

        self.win = tk.Toplevel(parent)
        self.win.title("🤖 Run Model Report")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("780x620")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._detector_var      = tk.StringVar(value=_DEFAULT_DETECTOR)
        self._quality_var       = tk.StringVar(value=_DEFAULT_QUALITY)
        self._det_conf_var      = tk.StringVar(value="0.85")
        self._min_conf_var      = tk.StringVar(value="0.635")
        self._tta_var           = tk.BooleanVar(value=False)
        self._skip_approved_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _on_close(self):
        self._stop_flag.set()
        self.win.destroy()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=1)

        self._build_config_panel()
        self._build_log_panel()
        self._build_bottom_bar()

    def _build_config_panel(self):
        cfg = tk.Frame(self.win, bg=BG_MID, pady=10)
        cfg.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        cfg.columnconfigure(1, weight=1)

        tk.Label(cfg, text="🤖  Model Report Configuration",
                 bg=BG_MID, fg=RUN_COLOR,
                 font=("Helvetica", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(4, 10))

        fields = [
            ("Stage 1 — Detector (.pt):", self._detector_var, _DEFAULT_DETECTOR),
            ("Stage 2 — Quality (.pt):",  self._quality_var,  _DEFAULT_QUALITY),
        ]
        for i, (label, var, default) in enumerate(fields, 1):
            tk.Label(cfg, text=label, bg=BG_MID, fg=WHITE,
                     font=("Helvetica", 10)).grid(row=i, column=0,
                                                   sticky="w", padx=(12, 6), pady=4)
            tk.Entry(cfg, textvariable=var, bg=BG_PANEL, fg=WHITE,
                     insertbackground=WHITE, relief="flat",
                     font=("Helvetica", 10)).grid(row=i, column=1,
                                                   sticky="ew", padx=4, pady=4)
            tk.Button(cfg, text="Browse", bg=BG_PANEL, fg=WHITE,
                      relief="flat", font=("Helvetica", 9), cursor="hand2",
                      command=lambda v=var: self._browse_model(v)).grid(
                row=i, column=2, padx=(4, 12), pady=4)

        # Thresholds row
        thresh = tk.Frame(cfg, bg=BG_MID)
        thresh.grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=(6, 4))

        tk.Label(thresh, text="Detect confidence:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        tk.Entry(thresh, textvariable=self._det_conf_var, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat", width=6,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 20))

        tk.Label(thresh, text="Min quality confidence:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        tk.Entry(thresh, textvariable=self._min_conf_var, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat", width=6,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 20))

        tk.Checkbutton(thresh, text="TTA (5-crop avg, slower)",
                       variable=self._tta_var, bg=BG_MID, fg=WHITE,
                       selectcolor=BG_PANEL, activebackground=BG_MID,
                       font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 16))

        tk.Checkbutton(thresh, text="Skip already-approved",
                       variable=self._skip_approved_var, bg=BG_MID, fg=WHITE,
                       selectcolor=BG_PANEL, activebackground=BG_MID,
                       font=("Helvetica", 10)).pack(side=tk.LEFT)

    def _build_log_panel(self):
        log_frame = tk.Frame(self.win, bg=BG_DARK)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, bg=BG_PANEL, fg=WHITE,
                           font=("Courier", 10), relief="flat",
                           state="disabled", wrap="word")
        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=vsb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Color tags for log
        self.log.tag_configure("BAD",      foreground=RESULT_COLORS["BAD"])
        self.log.tag_configure("UNCERTAIN",foreground=RESULT_COLORS["UNCERTAIN"])
        self.log.tag_configure("CONFLICT", foreground=RESULT_COLORS["CONFLICT"])
        self.log.tag_configure("GOOD",     foreground=RESULT_COLORS["GOOD"])
        self.log.tag_configure("CLEANING", foreground=RESULT_COLORS["CLEANING"])
        self.log.tag_configure("NOISE",    foreground=RESULT_COLORS["NOISE"])
        self.log.tag_configure("MISSING",  foreground=RESULT_COLORS["MISSING"])
        self.log.tag_configure("INFO",     foreground=ACCENT)
        self.log.tag_configure("ERROR",    foreground=RED)

    def _build_bottom_bar(self):
        bar = tk.Frame(self.win, bg=BG_MID, pady=8)
        bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))

        self._progress_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._progress_var,
                 bg=BG_MID, fg=GRAY, font=("Helvetica", 10)).pack(
            side=tk.LEFT, padx=12)

        self._stop_btn = tk.Button(bar, text="Stop", bg=RED, fg=WHITE,
                                   relief="flat", font=("Helvetica", 10, "bold"),
                                   cursor="hand2", command=self._stop,
                                   state="disabled")
        self._stop_btn.pack(side=tk.RIGHT, padx=(4, 12))

        self._run_btn = tk.Button(bar, text="▶  Run Report",
                                  bg=RUN_COLOR, fg="#000", relief="flat",
                                  font=("Helvetica", 10, "bold"),
                                  cursor="hand2", command=self._start)
        self._run_btn.pack(side=tk.RIGHT, padx=4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse_model(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            parent=self.win
        )
        if path:
            var.set(path)

    def _log(self, msg, tag=""):
        def _do():
            self.log.configure(state="normal")
            self.log.insert(tk.END, msg + "\n", tag)
            self.log.see(tk.END)
            self.log.configure(state="disabled")
        self.win.after(0, _do)

    def _set_progress(self, msg):
        self.win.after(0, lambda: self._progress_var.set(msg))

    # ── Run ───────────────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return
        self._stop_flag.clear()
        self._running = True
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        threading.Thread(target=self._run_report, daemon=True).start()

    def _stop(self):
        self._stop_flag.set()
        self._log("⚠ Stop requested — finishing current image...", "ERROR")

    def _run_report(self):
        try:
            self._do_run()
        except Exception as e:
            self._log(f"ERROR: {e}", "ERROR")
            import traceback
            self._log(traceback.format_exc(), "ERROR")
        finally:
            self._running = False
            self.win.after(0, lambda: self._run_btn.config(state="normal"))
            self.win.after(0, lambda: self._stop_btn.config(state="disabled"))

    def _do_run(self):
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision.models import efficientnet_b0, efficientnet_b2

        det_path  = self._detector_var.get().strip()
        qual_path = self._quality_var.get().strip()

        try:
            det_conf  = float(self._det_conf_var.get())
            min_conf  = float(self._min_conf_var.get())
        except ValueError:
            self._log("Invalid confidence values.", "ERROR")
            return

        use_tta       = self._tta_var.get()
        skip_approved = self._skip_approved_var.get()

        # ── Load models ──
        _ARCH_MAP = {"efficientnet_b0": efficientnet_b0,
                     "efficientnet_b2": efficientnet_b2}

        def make_net(n, arch="efficientnet_b0"):
            builder = _ARCH_MAP.get(arch, efficientnet_b0)
            net = builder(weights=None)
            net.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(net.classifier[1].in_features, n)
            )
            return net

        def build_transform(sz):
            return T.Compose([
                T.Resize(sz + 32), T.CenterCrop(sz), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

        def build_tta(sz):
            norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            return [
                T.Compose([T.Resize(sz+32), T.CenterCrop(sz),  T.ToTensor(), norm]),
                T.Compose([T.Resize(sz+32), T.RandomCrop(sz),  T.ToTensor(), norm]),
                T.Compose([T.Resize(sz+32), T.RandomCrop(sz),  T.ToTensor(), norm]),
                T.Compose([T.Resize(sz+32), T.RandomCrop(sz),  T.ToTensor(), norm]),
                T.Compose([T.Resize(sz+32), T.RandomCrop(sz),  T.ToTensor(), norm]),
            ]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log(f"Device: {device}", "INFO")

        # Stage 1
        detector = cs_idx = det_tf = det_tta_tfs = None
        if os.path.exists(det_path):
            bundle   = torch.load(det_path, map_location="cpu", weights_only=False)
            det_arch = bundle.get("architecture", "efficientnet_b0")
            detector = make_net(2, det_arch)
            detector.load_state_dict(bundle["model_state_dict"])
            detector.eval().to(device)
            cs_idx     = bundle["cs_class_idx"]
            det_sz     = bundle.get("input_size", 224)
            det_tf     = build_transform(det_sz)
            det_tta_tfs= build_tta(det_sz)
            self._log(f"Stage 1 loaded  | {det_arch} | input={det_sz}px | cs_idx={cs_idx} | "
                      f"val_acc={bundle.get('val_acc','?')}", "INFO")
        else:
            self._log("Stage 1 detector not found — single-stage mode.", "INFO")

        # Stage 2
        if not os.path.exists(qual_path):
            self._log(f"Quality model not found: {qual_path}", "ERROR")
            return

        bundle2      = torch.load(qual_path, map_location="cpu", weights_only=False)
        idx_to_class = bundle2["idx_to_class"]
        qual_arch    = bundle2.get("architecture", "efficientnet_b2")
        num_classes  = len(idx_to_class)
        qual_sz      = bundle2.get("input_size", 224)
        qual_tf      = build_transform(qual_sz)
        qual_tta_tfs = build_tta(qual_sz)
        # use stored threshold if available, else fall back to UI value
        stored_thresh = bundle2.get("threshold")
        if stored_thresh is not None:
            min_conf = stored_thresh

        if "fold_state_dicts" in bundle2:
            nets = []
            for sd in bundle2["fold_state_dicts"]:
                n = make_net(num_classes, qual_arch); n.load_state_dict(sd)
                n.eval().to(device); nets.append(n)
            self._log(f"Stage 2 loaded  | {qual_arch} | {len(nets)}-fold ensemble | input={qual_sz}px | "
                      f"threshold={min_conf:.3f} | mean_val_acc={bundle2.get('mean_val_acc','?')}", "INFO")
        else:
            net2 = make_net(num_classes, qual_arch)
            net2.load_state_dict(bundle2["model_state_dict"])
            net2.eval().to(device)
            nets = [net2]
            self._log(f"Stage 2 loaded  | single model | input={qual_sz}px | "
                      f"val_acc={bundle2.get('val_acc','?')}", "INFO")

        # ── Run over index rows ──
        rows_to_run = [
            r for r in self.all_rows
            if r.get("url") and
            not (skip_approved and r.get("status") == "approved")
        ]
        total  = len(rows_to_run)
        counts = {k: 0 for k in ["BAD","UNCERTAIN","CONFLICT","GOOD",
                                  "CLEANING","NOISE","MISSING","SKIPPED"]}

        self._log(f"\nRunning on {total} images "
                  f"({'skip approved' if skip_approved else 'all'})...\n", "INFO")

        url_to_row = {r.get("url"): r for r in self.all_rows}

        for i, row in enumerate(rows_to_run, 1):
            if self._stop_flag.is_set():
                self._log("Stopped by user.", "INFO")
                break

            url   = row.get("url", "").strip()
            notes = (row.get("notes") or "").strip().lower()

            # Skip known non-target descriptions
            if notes in SKIP_DESCRIPTIONS:
                counts["NOISE"] += 1
                url_to_row[url]["notes"]  = "NOISE | skipped description"
                url_to_row[url]["status"] = "rejected"
                self._set_progress(f"{i}/{total}  noise={counts['NOISE']}")
                continue

            # Fetch image
            try:
                req = urllib.request.Request(
                    url.split("?")[0], headers={"User-Agent": "onesvs-browser/1.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    raw = r.read()
                img = ImageOps.exif_transpose(
                    Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                counts["MISSING"] += 1
                url_to_row[url]["notes"]  = "MISSING"
                url_to_row[url]["status"] = "rejected"
                self._set_progress(f"{i}/{total}  missing={counts['MISSING']}")
                continue

            result      = None
            detect_conf = None

            # Stage 1
            if detector is not None:
                with torch.no_grad():
                    tf    = det_tf(img).unsqueeze(0).to(device)
                    probs = torch.softmax(detector(tf), dim=1).squeeze()
                detect_conf = round(probs[cs_idx].item(), 4)
                if detect_conf < det_conf:
                    result = "NOISE"

            if result == "NOISE":
                counts["NOISE"] += 1
                url_to_row[url]["notes"]  = f"NOISE | det={detect_conf}"
                url_to_row[url]["status"] = "rejected"
                self._set_progress(f"{i}/{total}  noise={counts['NOISE']}")
                continue

            # Stage 2
            all_probs = []
            with torch.no_grad():
                for net in nets:
                    tfs = qual_tta_tfs if use_tta else [qual_tf]
                    for tf in tfs:
                        t = tf(img).unsqueeze(0).to(device)
                        all_probs.append(torch.softmax(net(t), dim=1).squeeze())

            probs2       = torch.stack(all_probs).mean(0)
            sorted_p, _  = torch.sort(probs2, descending=True)
            class_idx    = probs2.argmax().item()
            result       = idx_to_class[class_idx].upper()
            qual_conf    = round(sorted_p[0].item(), 4)
            qual_margin  = round((sorted_p[0] - sorted_p[1]).item(), 4)

            if qual_conf < min_conf:
                result = "UNCERTAIN"

            det_str = f" | det={detect_conf}" if detect_conf is not None else ""
            note    = f"{result} | conf={qual_conf} | margin={qual_margin}{det_str}"
            url_to_row[url]["notes"]  = note
            url_to_row[url]["status"] = RESULT_TO_STATUS.get(result, "pending")

            counts[result] = counts.get(result, 0) + 1
            tag = result if result in RESULT_COLORS else ""
            self._log(
                f"[{i:>5}/{total}] {result:<10} conf={qual_conf}  "
                f"{url.split('/')[-1][:40]}", tag
            )
            self._set_progress(
                f"{i}/{total}  bad={counts['BAD']}  "
                f"uncertain={counts['UNCERTAIN']}  good={counts['GOOD']}  "
                f"noise={counts['NOISE']}"
            )

        # ── Conflict detection ──
        self._log("\nChecking for conflicts...", "INFO")
        unit_results: dict = {}
        for r in self.all_rows:
            notes = r.get("notes", "")
            result = next((k for k in ["BAD","GOOD"] if notes.upper().startswith(k)), None)
            job_id = r.get("job_id", "")
            if result and job_id:
                unit_results.setdefault(job_id, set()).add(result)

        conflicts = {j for j, rs in unit_results.items()
                     if "BAD" in rs and "GOOD" in rs}
        n_conflict = 0
        for r in self.all_rows:
            if r.get("job_id") in conflicts and \
               r.get("notes", "").upper().startswith("GOOD"):
                r["notes"]  = r["notes"].replace("GOOD", "CONFLICT", 1)
                r["status"] = "pending"
                n_conflict += 1

        counts["CONFLICT"] = n_conflict
        self._log(f"Conflicts flagged: {n_conflict}", "INFO")

        # ── Summary ──
        summary = "\n" + "="*50 + "\n"
        for k, v in counts.items():
            if v > 0:
                summary += f"  {k:<12}: {v}\n"
        self._log(summary, "INFO")

        updated = list(url_to_row.values())
        self.win.after(0, lambda: self.on_complete(updated, counts))


# ── Database Query Window ─────────────────────────────────────────────────────

# Default connection — reads from .env if present
def _read_env_db():
    """Try to read DB credentials from Laravel .env file."""
    # LARAVEL_ENV_PATH can be set in the FastAPI .env to point to the Laravel app's .env
    laravel_env = os.getenv("LARAVEL_ENV_PATH", "")
    env_paths = [p for p in [
        laravel_env,
        "/mnt/c/xampp/htdocs/website_rebuild/.env",
        os.path.join(os.path.dirname(_SCRIPT_DIR), ".env"),
    ] if p]
    vals = {"host": "127.0.0.1", "port": "3306", "db": "staging_db",
            "user": "mylfs", "password": ""}
    for path in env_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DB_HOST="):     vals["host"]     = line.split("=",1)[1].strip()
                    elif line.startswith("DB_PORT="):   vals["port"]     = line.split("=",1)[1].strip()
                    elif line.startswith("DB_DATABASE="):vals["db"]      = line.split("=",1)[1].strip()
                    elif line.startswith("DB_USERNAME="):vals["user"]    = line.split("=",1)[1].strip()
                    elif line.startswith("DB_PASSWORD="):vals["password"]= line.split("=",1)[1].strip()
            break
    return vals


class DatabaseQueryWindow:
    """
    Query the units_visits database directly.
    Filters by company, unit type, and photo description keywords (tags).
    Results are added to index.csv as URL references — images never downloaded.
    """

    def __init__(self, parent, index_path, existing_rows, on_added):
        self.index_path    = index_path
        self.existing_rows = existing_rows
        self.on_added      = on_added
        self.results       = []
        self._thumb_img    = None
        self._conn         = None
        self._companies    = []
        self._unit_types   = []

        self.win = tk.Toplevel(parent)
        self.win.title("Query Database — Dataset Builder")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("1150x700")
        self.win.minsize(900, 550)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        env = _read_env_db()
        self._host_var     = tk.StringVar(value=env["host"])
        self._port_var     = tk.StringVar(value=env["port"])
        self._db_var       = tk.StringVar(value=env["db"])
        self._user_var     = tk.StringVar(value=env["user"])
        self._pass_var     = tk.StringVar(value=env["password"])
        self._db_type_var  = tk.StringVar(value="mysql")

        self._build_ui()
        self.win.after(200, self._auto_connect)

    def _on_close(self):
        if self._conn:
            try: self._conn.close()
            except: pass
        self.win.destroy()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=1)
        self._build_conn_bar()
        self._build_query_panel()
        self._build_bottom_bar()

    def _build_conn_bar(self):
        bar = tk.Frame(self.win, bg=BG_MID, pady=8)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        tk.Label(bar, text="DB Type:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(12, 4))
        for label, val in [("MySQL", "mysql"), ("PostgreSQL", "postgres")]:
            tk.Radiobutton(bar, text=label, variable=self._db_type_var, value=val,
                           bg=BG_MID, fg=WHITE, selectcolor=BG_PANEL,
                           activebackground=BG_MID,
                           font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)

        for label, var, w in [
            ("Host:", self._host_var, 12),
            ("Port:", self._port_var, 5),
            ("DB:",   self._db_var,   14),
            ("User:", self._user_var, 10),
            ("Pass:", self._pass_var, 10),
        ]:
            tk.Label(bar, text=label, bg=BG_MID, fg=WHITE,
                     font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(10, 3))
            show = "*" if label == "Pass:" else None
            tk.Entry(bar, textvariable=var, bg=BG_PANEL, fg=WHITE,
                     insertbackground=WHITE, relief="flat", width=w,
                     show=show, font=("Helvetica", 9)).pack(side=tk.LEFT)

        self._conn_btn = tk.Button(bar, text="Connect", bg=ACCENT, fg="#000",
                                   relief="flat", font=("Helvetica", 9, "bold"),
                                   cursor="hand2", command=self._connect)
        self._conn_btn.pack(side=tk.LEFT, padx=(12, 4))

        self._conn_status = tk.Label(bar, text="Not connected", bg=BG_MID, fg=GRAY,
                                     font=("Helvetica", 9))
        self._conn_status.pack(side=tk.LEFT, padx=8)

    def _build_query_panel(self):
        pane = tk.Frame(self.win, bg=BG_DARK)
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        pane.columnconfigure(1, weight=1)
        pane.rowconfigure(1, weight=1)

        # ── Filter bar — two rows ──
        fbar = tk.Frame(pane, bg=BG_MID, pady=8)
        fbar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        row1 = tk.Frame(fbar, bg=BG_MID)
        row1.pack(fill=tk.X, padx=10, pady=(0, 6))
        row2 = tk.Frame(fbar, bg=BG_MID)
        row2.pack(fill=tk.X, padx=10)

        # ── Row 1: Company | Unit Type | Report Type | Hoshizaki Cares toggle ──
        tk.Label(row1, text="Company:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self._company_var = tk.StringVar(value="All")
        self._company_cb  = ttk.Combobox(row1, textvariable=self._company_var,
                                          state="readonly", width=16)
        self._company_cb.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(row1, text="Unit Type:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self._unit_type_var = tk.StringVar(value="All")
        self._unit_type_cb  = ttk.Combobox(row1, textvariable=self._unit_type_var,
                                            state="readonly", width=16)
        self._unit_type_cb.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(row1, text="Report Type:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self._report_type_var = tk.StringVar(value="All")
        self._report_type_cb  = ttk.Combobox(row1, textvariable=self._report_type_var,
                                              state="readonly", width=26)
        self._report_type_cb.pack(side=tk.LEFT, padx=(0, 14))

        # Hoshizaki Cares tag toggle
        self._hoz_cares_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row1, text="Hoshizaki Cares Only",
                       variable=self._hoz_cares_var,
                       bg=BG_MID, fg=ACCENT, selectcolor=BG_PANEL,
                       activebackground=BG_MID,
                       font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 14))

        # ── Row 2: Service type | Description tags | Limit | Run ──
        tk.Label(row2, text="Service Type:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self._service_type_var = tk.StringVar(value="")
        st_entry = tk.Entry(row2, textvariable=self._service_type_var,
                            bg=BG_PANEL, fg=WHITE, insertbackground=WHITE,
                            relief="flat", width=22, font=("Helvetica", 10))
        st_entry.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(row2, text="Description Tags:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self._kw_var = tk.StringVar()
        kw_entry = tk.Entry(row2, textvariable=self._kw_var, bg=BG_PANEL, fg=WHITE,
                            insertbackground=WHITE, relief="flat", width=28,
                            font=("Helvetica", 10))
        kw_entry.pack(side=tk.LEFT, padx=(0, 4))
        kw_entry.bind("<Return>", lambda e: self._run_query())
        tk.Label(row2, text="e.g. gauge, water filter, service loop",
                 bg=BG_MID, fg=GRAY, font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(row2, text="Limit:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self._limit_var = tk.StringVar(value="500")
        tk.Entry(row2, textvariable=self._limit_var, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat", width=6,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(row2, text="🔍  Run Query", bg=GREEN, fg="#000",
                  relief="flat", font=("Helvetica", 10, "bold"),
                  cursor="hand2", command=self._run_query).pack(side=tk.LEFT, padx=4)

        # ── Results treeview ──
        tree_frame = tk.Frame(pane, bg=BG_DARK)
        tree_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        cols = ("unit_id", "company", "unit_type", "report_type",
                "description", "photo_type", "date", "url")
        self.result_tree = ttk.Treeview(tree_frame, columns=cols,
                                         show="headings", selectmode="extended")
        col_cfg = [
            ("unit_id",     "Unit ID",         75,  tk.CENTER),
            ("company",     "Company",         110,  tk.W),
            ("unit_type",   "Unit Type",       100,  tk.W),
            ("report_type", "Report Type",     130,  tk.W),
            ("description", "Tag / Description", 180, tk.W),
            ("photo_type",  "Photo Type",       85,  tk.CENTER),
            ("date",        "Date",             90,  tk.CENTER),
            ("url",         "URL",             200,  tk.W),
        ]
        for cid, heading, width, anchor in col_cfg:
            self.result_tree.heading(cid, text=heading)
            self.result_tree.column(cid, width=width, anchor=anchor,
                                    stretch=(cid in ("description", "url")))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",   command=self.result_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.result_tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── Thumbnail preview ──
        preview = tk.Frame(pane, bg=BG_MID, width=250)
        preview.grid(row=1, column=2, sticky="nsew", padx=(6, 0))
        preview.grid_propagate(False)

        tk.Label(preview, text="Preview", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(pady=(10, 6), padx=10, anchor="w")
        self.thumb_canvas = tk.Canvas(preview, bg=BG_PANEL,
                                      width=230, height=175, highlightthickness=0)
        self.thumb_canvas.pack(padx=10, pady=4)
        self._draw_placeholder("Select a result")

        self._prev_desc = tk.Label(preview, text="", bg=BG_MID, fg=WHITE,
                                   font=("Helvetica", 9), wraplength=220, justify=tk.LEFT)
        self._prev_desc.pack(padx=10, pady=4, anchor="w")
        self._prev_type = tk.Label(preview, text="", bg=BG_MID, fg=GRAY,
                                   font=("Helvetica", 8))
        self._prev_type.pack(padx=10, anchor="w")

    def _build_bottom_bar(self):
        bar = tk.Frame(self.win, bg=BG_MID, pady=8)
        bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))

        self._count_label = tk.Label(bar, text="", bg=BG_MID, fg=GRAY,
                                     font=("Helvetica", 10))
        self._count_label.pack(side=tk.LEFT, padx=12)

        tk.Button(bar, text="Add Selected to Index",
                  bg=GREEN, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2",
                  command=self._add_selected).pack(side=tk.RIGHT, padx=(4, 12))

        tk.Button(bar, text="Add All to Index",
                  bg=ACCENT, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2",
                  command=self._add_all).pack(side=tk.RIGHT, padx=4)

        tk.Button(bar, text="✎  Open Selected in Annotator",
                  bg=YELLOW, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2",
                  command=self._open_in_annotator).pack(side=tk.RIGHT, padx=4)

        tk.Button(bar, text="⬇  Export Results to CSV",
                  bg=BG_PANEL, fg=WHITE, relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2",
                  command=self._export_csv).pack(side=tk.RIGHT, padx=4)

    # ── Connection ────────────────────────────────────────────────────────────

    def _auto_connect(self):
        self._connect(silent=True)

    def _connect(self, silent=False):
        try:
            db_type = self._db_type_var.get()
            if db_type == "mysql":
                import mysql.connector
                self._conn = mysql.connector.connect(
                    host=self._host_var.get(),
                    port=int(self._port_var.get()),
                    database=self._db_var.get(),
                    user=self._user_var.get(),
                    password=self._pass_var.get(),
                    connection_timeout=10,
                )
            else:
                import psycopg2
                self._conn = psycopg2.connect(
                    host=self._host_var.get(),
                    port=int(self._port_var.get()),
                    dbname=self._db_var.get(),
                    user=self._user_var.get(),
                    password=self._pass_var.get(),
                    connect_timeout=10,
                )
            self._conn_status.config(text="✓ Connected", fg=GREEN)
            self._conn_btn.config(text="Reconnect")
            self._load_dropdowns()
        except ImportError as e:
            pkg = "mysql-connector-python" if "mysql" in str(e) else "psycopg2-binary"
            self._conn_status.config(text=f"Install {pkg} first", fg=RED)
        except Exception as e:
            self._conn_status.config(text=f"✗ {e}", fg=RED)
            if not silent:
                messagebox.showerror("Connection failed", str(e), parent=self.win)

    def _load_dropdowns(self):
        """Populate company and unit type dropdowns from the database."""
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT id, name FROM companies ORDER BY name")
            self._companies = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
            self._company_cb["values"] = ["All"] + [c["name"] for c in self._companies]
            self._company_cb.set("All")

            cur.execute("SELECT id, name FROM unit_types ORDER BY name")
            self._unit_types = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
            self._unit_type_cb["values"] = ["All"] + [u["name"] for u in self._unit_types]
            self._unit_type_cb.set("All")

            cur.execute("SELECT DISTINCT report_type FROM visits WHERE report_type IS NOT NULL ORDER BY report_type")
            report_types = [r[0] for r in cur.fetchall()]
            self._report_type_cb["values"] = ["All"] + report_types
            self._report_type_cb.set("All")
            cur.close()
        except Exception as e:
            self._conn_status.config(text=f"Dropdown error: {e}", fg=YELLOW)

    # ── Query ─────────────────────────────────────────────────────────────────

    def _run_query(self):
        if not self._conn:
            messagebox.showinfo("Not connected",
                                "Connect to the database first.", parent=self.win)
            return

        company_name    = self._company_var.get()
        unit_type_name  = self._unit_type_var.get()
        report_type     = self._report_type_var.get()
        service_type_kw = self._service_type_var.get().strip().lower()
        hoz_cares_only  = self._hoz_cares_var.get()
        keywords        = [k.strip().lower() for k in
                           self._kw_var.get().split(",") if k.strip()]
        try:
            limit = int(self._limit_var.get())
        except ValueError:
            limit = 500

        # Build WHERE clauses
        where  = ["(uv.full_picture IS NOT NULL OR (uv.photos IS NOT NULL AND uv.photos != '[]'))"]
        params = []
        joins  = [
            "LEFT JOIN companies  c  ON c.id  = uv.company_id",
            "LEFT JOIN unit_types ut ON ut.id = uv.unit_type_id",
            "LEFT JOIN visits     v  ON v.id  = uv.visit_id",
        ]

        if company_name != "All":
            cid = next((c["id"] for c in self._companies if c["name"] == company_name), None)
            if cid:
                where.append("uv.company_id = %s")
                params.append(cid)

        if unit_type_name != "All":
            uid = next((u["id"] for u in self._unit_types if u["name"] == unit_type_name), None)
            if uid:
                where.append("uv.unit_type_id = %s")
                params.append(uid)

        if report_type != "All":
            where.append("v.report_type = %s")
            params.append(report_type)

        if service_type_kw:
            where.append("LOWER(v.service_type) LIKE %s")
            params.append(f"%{service_type_kw}%")

        # Hoshizaki Cares — join tagged_items to filter only tagged sites
        if hoz_cares_only:
            joins.append(
                "INNER JOIN tagged_items ti ON ti.type = 'Site' "
                "AND ti.external_id = v.site_id AND ti.tag_id = 1 "
                "AND ti.deactivated = 0"
            )

        sql = f"""
            SELECT
                uv.id          AS uv_id,
                uv.company_id,
                c.name         AS company_name,
                uv.visit_id,
                uv.unit_id     AS unit_db_id,
                ut.name        AS unit_type,
                v.report_type,
                v.service_type,
                uv.full_picture,
                uv.photos,
                uv.problem_photos,
                uv.created_at
            FROM units_visits uv
            {chr(10).join(joins)}
            WHERE {' AND '.join(where)}
            ORDER BY uv.created_at DESC
            LIMIT %s
        """
        params.append(limit)

        existing_urls = {r.get("url", "") for r in self.existing_rows}
        self.results  = []
        skipped_dup   = 0

        try:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
        except Exception as e:
            messagebox.showerror("Query error", str(e), parent=self.win)
            return

        for row in rows:
            (uv_id, company_id, company_name_, visit_id, unit_id,
             unit_type, report_type_, service_type_,
             full_pic, photos_json, problems_json, created_at) = row

            # Parse photo JSON fields
            try:
                photos   = json.loads(photos_json   or "[]")
            except Exception:
                photos   = []
            try:
                problems = json.loads(problems_json or "[]")
            except Exception:
                problems = []

            # Build photo entries: (url, description, photo_type)
            candidates = []
            if full_pic:
                candidates.append((full_pic.strip(), "full picture", "full_picture"))
            for p in photos:
                u = (p.get("photo") or "").strip()
                d = (p.get("description") or "").strip()
                if u: candidates.append((u, d, "photo"))
            for p in problems:
                u = (p.get("photo") or "").strip()
                d = (p.get("description") or "").strip()
                if u: candidates.append((u, d, "problem_photo"))

            for url, desc, photo_type in candidates:
                if not url:
                    continue
                # Keyword filter on description
                if keywords and not any(kw in desc.lower() for kw in keywords):
                    continue
                if url in existing_urls:
                    skipped_dup += 1
                    continue
                self.results.append({
                    "url":          url,
                    "description":  desc,
                    "photo_type":   photo_type,
                    "unit_id":      unit_id or uv_id,
                    "company_id":   company_id,
                    "company":      company_name_ or "",
                    "unit_type":    unit_type or "",
                    "report_type":  report_type_ or "",
                    "service_type": service_type_ or "",
                    "visit_id":     visit_id,
                    "date":         str(created_at)[:10] if created_at else "",
                })

        self._populate_results()
        dup_note = f"  ({skipped_dup} duplicates skipped)" if skipped_dup else ""
        self._count_label.config(
            text=f"{len(self.results)} photo(s) found{dup_note}")

    def _populate_results(self):
        self.result_tree.delete(*self.result_tree.get_children())
        for i, r in enumerate(self.results):
            url_short = ("..." + r["url"][-50:]) if len(r["url"]) > 50 else r["url"]
            self.result_tree.insert("", tk.END, iid=str(i), values=(
                r["unit_id"], r["company"], r["unit_type"],
                r.get("report_type", ""), r["description"],
                r["photo_type"], r["date"], url_short,
            ))

    # ── Preview ───────────────────────────────────────────────────────────────

    def _on_select(self, event=None):
        sel = self.result_tree.selection()
        if not sel:
            return
        r = self.results[int(sel[0])]
        self._prev_desc.config(text=r.get("description") or "(no description)")
        self._prev_type.config(text=r.get("photo_type", ""))
        self._draw_placeholder("Loading...")
        fetch_thumbnail(r["url"], lambda pil: self.win.after(0, lambda: self._show_thumb(pil)))

    def _show_thumb(self, pil_image):
        self.thumb_canvas.delete("all")
        if pil_image is None:
            self._draw_placeholder("Could not load")
            return
        self._thumb_img = ImageTk.PhotoImage(pil_image)
        x = (230 - pil_image.width)  // 2
        y = (175 - pil_image.height) // 2
        self.thumb_canvas.create_image(x, y, anchor=tk.NW, image=self._thumb_img)

    def _draw_placeholder(self, msg=""):
        self.thumb_canvas.delete("all")
        self._thumb_img = None
        self.thumb_canvas.create_text(115, 87, text=msg, fill=GRAY,
                                      font=("Helvetica", 10))

    # ── Add to Index ──────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self.results:
            messagebox.showinfo("No results", "Run a query first.", parent=self.win)
            return

        out = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="query_results.csv",
            title="Export Results to CSV",
            parent=self.win,
        )
        if not out:
            return

        fields = ["unit_id", "company", "unit_type", "report_type",
                  "service_type", "description", "photo_type",
                  "visit_id", "date", "url"]
        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.results)
            self._count_label.config(
                text=f"Exported {len(self.results)} row(s) to {os.path.basename(out)}")
        except Exception as e:
            messagebox.showerror("Export error", str(e), parent=self.win)

    def _open_in_annotator(self):
        sel = self.result_tree.selection()
        if not sel:
            messagebox.showinfo("No selection",
                                "Select one or more rows first.", parent=self.win)
            return
        urls = [self.results[int(i)]["url"] for i in sel
                if self.results[int(i)].get("url")]
        if not urls:
            messagebox.showerror("No URL", "Selected rows have no URL.", parent=self.win)
            return
        cmd = ([PYTHON_EXE, ANNOTATOR_PY] +
               [arg for u in urls for arg in ("--url", u)] +
               ["--json", self.index_path.replace("index.csv", "annotations.json")])
        try:
            subprocess.Popen(cmd, start_new_session=True)
            self._count_label.config(
                text=f"Annotator opened with {len(urls)} image(s).")
        except Exception as e:
            messagebox.showerror("Launch error", str(e), parent=self.win)

    def _add_selected(self):
        sel = self.result_tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select rows first.", parent=self.win)
            return
        self._commit([self.results[int(i)] for i in sel])

    def _add_all(self):
        if not self.results:
            messagebox.showinfo("No results", "Run a query first.", parent=self.win)
            return
        self._commit(self.results)

    def _commit(self, rows_to_add):
        existing_rows = load_index(self.index_path)
        existing_urls = {r.get("url", "") for r in existing_rows}
        new_id = max((int(r.get("image_id", 0)) for r in existing_rows), default=0)
        added  = 0
        for r in rows_to_add:
            url = r.get("url", "").strip()
            if not url or url in existing_urls:
                continue
            new_id += 1
            existing_rows.append({
                "image_id":     new_id,
                "company_id":   r.get("company_id", ""),
                "company_name": r.get("company", ""),
                "job_id":       r.get("visit_id", ""),
                "url":          url,
                "status":       "pending",
                "annotated_by": "",
                "date":         r.get("date", ""),
                "notes":        r.get("description", ""),
            })
            existing_urls.add(url)
            added += 1

        save_index(self.index_path, existing_rows)
        self.existing_rows = existing_rows
        if self.on_added:
            self.on_added(added)
        messagebox.showinfo("Added", f"{added} image(s) added to index.", parent=self.win)
        if added:
            self.win.destroy()


# ── Search Window ─────────────────────────────────────────────────────────────

class SearchWindow:
    """
    Dataset Builder — searches a unit visits CSV export for photos matching
    keywords in the photo description field.
    Results are added to index.csv as URL references — images never downloaded.
    """

    RESULT_COLS = ("unit_id", "company", "model", "description", "unit_type", "date", "url")

    def __init__(self, parent, index_path, existing_rows, on_added):
        self.index_path    = index_path
        self.existing_rows = existing_rows
        self.on_added      = on_added
        self.results       = []          # list of result dicts from last search
        self._thumb_img    = None
        self._csv_path     = tk.StringVar()

        self.win = tk.Toplevel(parent)
        self.win.title("Dataset Builder — Search CSV")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("1100x680")
        self.win.minsize(800, 500)
        self.win.grab_set()

        self._build_ui()

    def _build_ui(self):
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=1)

        self._build_search_bar()
        self._build_results_panel()
        self._build_bottom_bar()

    def _build_search_bar(self):
        bar = tk.Frame(self.win, bg=BG_MID, pady=10)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        bar.columnconfigure(1, weight=1)

        # CSV file picker
        tk.Label(bar, text="CSV File:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=(12, 6), sticky="w")
        tk.Entry(bar, textvariable=self._csv_path, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat",
                 font=("Helvetica", 10), width=50).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(bar, text="Browse", bg=BG_PANEL, fg=WHITE, relief="flat",
                  font=("Helvetica", 9), cursor="hand2",
                  command=self._browse_csv).grid(row=0, column=2, padx=(4, 12))

        # Keywords
        tk.Label(bar, text="Keywords:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 10, "bold")).grid(row=1, column=0, padx=(12, 6),
                                                       pady=(8, 0), sticky="w")
        self._keywords_var = tk.StringVar(value="")
        kw_entry = tk.Entry(bar, textvariable=self._keywords_var, bg=BG_PANEL, fg=WHITE,
                            insertbackground=WHITE, relief="flat",
                            font=("Helvetica", 10))
        kw_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))
        kw_entry.bind("<Return>", lambda e: self._run_search())
        tk.Label(bar, text="comma-separated  e.g. water filter, gauge, cube",
                 bg=BG_MID, fg=GRAY, font=("Helvetica", 8)).grid(
            row=2, column=1, sticky="w", padx=4)

        # Filters row
        filter_frame = tk.Frame(bar, bg=BG_MID)
        filter_frame.grid(row=1, column=2, rowspan=2, padx=(8, 12), pady=(8, 0), sticky="w")

        tk.Label(filter_frame, text="Unit Type:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self._unit_type_var = tk.StringVar(value="")
        tk.Entry(filter_frame, textvariable=self._unit_type_var, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat", width=12,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(filter_frame, text="Company:", bg=BG_MID, fg=WHITE,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self._company_var = tk.StringVar(value="")
        tk.Entry(filter_frame, textvariable=self._company_var, bg=BG_PANEL, fg=WHITE,
                 insertbackground=WHITE, relief="flat", width=14,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(filter_frame, text="🔍  Search", bg=ACCENT, fg="#000",
                  relief="flat", font=("Helvetica", 10, "bold"),
                  cursor="hand2", command=self._run_search).pack(side=tk.LEFT, padx=4)

    def _build_results_panel(self):
        pane = tk.Frame(self.win, bg=BG_DARK)
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        pane.columnconfigure(0, weight=1)
        pane.columnconfigure(1, minsize=260)
        pane.rowconfigure(0, weight=1)

        # Results treeview
        tree_frame = tk.Frame(pane, bg=BG_DARK)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        col_cfg = [
            ("unit_id",     "Unit ID",     80,  tk.CENTER),
            ("company",     "Company",    120,  tk.W),
            ("model",       "Model",      120,  tk.W),
            ("description", "Description",200,  tk.W),
            ("unit_type",   "Unit Type",  110,  tk.W),
            ("date",        "Date",        95,  tk.CENTER),
            ("url",         "URL",        200,  tk.W),
        ]
        self.result_tree = ttk.Treeview(
            tree_frame, columns=[c[0] for c in col_cfg],
            show="headings", selectmode="extended"
        )
        for cid, heading, width, anchor in col_cfg:
            self.result_tree.heading(cid, text=heading)
            self.result_tree.column(cid, width=width, anchor=anchor,
                                    stretch=(cid == "description"))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.result_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                            command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.result_tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self.result_tree.bind("<Double-1>", lambda e: self._preview_selected())

        # Preview panel
        preview = tk.Frame(pane, bg=BG_MID, width=260)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.grid_propagate(False)

        tk.Label(preview, text="Preview", bg=BG_MID, fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(pady=(10, 6), padx=10, anchor="w")

        self.thumb_canvas = tk.Canvas(preview, bg=BG_PANEL,
                                      width=240, height=180, highlightthickness=0)
        self.thumb_canvas.pack(padx=10, pady=4)
        self._thumb_placeholder()

        self.preview_desc = tk.Label(preview, text="", bg=BG_MID, fg=WHITE,
                                     font=("Helvetica", 9), wraplength=230,
                                     justify=tk.LEFT)
        self.preview_desc.pack(padx=10, pady=4, anchor="w")

        self.preview_url = tk.Label(preview, text="", bg=BG_MID, fg=GRAY,
                                    font=("Helvetica", 8), wraplength=230,
                                    justify=tk.LEFT)
        self.preview_url.pack(padx=10, pady=2, anchor="w")

    def _build_bottom_bar(self):
        bar = tk.Frame(self.win, bg=BG_MID, pady=8)
        bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))

        self.result_count_label = tk.Label(bar, text="No results yet",
                                           bg=BG_MID, fg=GRAY,
                                           font=("Helvetica", 10))
        self.result_count_label.pack(side=tk.LEFT, padx=12)

        tk.Button(bar, text="Add Selected to Index",
                  bg=GREEN, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2",
                  command=self._add_selected).pack(side=tk.RIGHT, padx=(4, 12))

        tk.Button(bar, text="Add All to Index",
                  bg=ACCENT, fg="#000", relief="flat",
                  font=("Helvetica", 10, "bold"), cursor="hand2",
                  command=self._add_all).pack(side=tk.RIGHT, padx=4)

    # ── Search Logic ──────────────────────────────────────────────────────────

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select unit visits CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._csv_path.set(path)

    def _run_search(self):
        csv_path = self._csv_path.get().strip()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("CSV not found",
                                 "Select a valid unit visits CSV file first.",
                                 parent=self.win)
            return

        raw_keywords = [k.strip().lower() for k in
                        self._keywords_var.get().split(",") if k.strip()]
        unit_filter  = self._unit_type_var.get().strip().lower()
        company_filter = self._company_var.get().strip().lower()

        # Existing URLs already in index — skip duplicates
        existing_urls = {r.get("url", "") for r in self.existing_rows}

        self.results = []
        skipped_dup  = 0

        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Unit type filter
                    unit_type = row.get("unit_type", "")
                    if unit_filter and unit_filter not in unit_type.lower():
                        continue

                    # Company filter
                    company = (row.get("company_name") or row.get("company") or
                               row.get("Company") or "")
                    if company_filter and company_filter not in company.lower():
                        continue

                    # Parse photo JSON fields
                    try:
                        photos   = json.loads(row.get("photo json")    or
                                              row.get("photos")        or "[]")
                        problems = json.loads(row.get("problem photos json") or
                                              row.get("problem_photos") or "[]")
                    except Exception:
                        continue

                    all_photos = photos + problems

                    for p in all_photos:
                        url  = (p.get("photo") or p.get("url") or "").strip()
                        desc = (p.get("description") or "").strip()

                        if not url:
                            continue

                        # Keyword match — if no keywords, include all photos
                        if raw_keywords:
                            if not any(kw in desc.lower() for kw in raw_keywords):
                                continue

                        if url in existing_urls:
                            skipped_dup += 1
                            continue

                        self.results.append({
                            "url":         url,
                            "description": desc,
                            "unit_id":     row.get("Unit ID") or row.get("unit_id", ""),
                            "company":     company,
                            "company_id":  row.get("company_id", ""),
                            "model":       row.get("model", ""),
                            "serial":      row.get("serial", ""),
                            "unit_type":   unit_type,
                            "visit_id":    row.get("visit #") or row.get("visit_id", ""),
                            "date":        row.get("completed time") or
                                           row.get("completed_at") or
                                           row.get("created_at", ""),
                            "uar_link":    row.get("uar Link") or row.get("uar_link", ""),
                        })
        except Exception as e:
            messagebox.showerror("Read error", str(e), parent=self.win)
            return

        self._populate_results()
        dup_note = f"  ({skipped_dup} duplicates skipped)" if skipped_dup else ""
        self.result_count_label.config(
            text=f"{len(self.results)} result(s) found{dup_note}")

    def _populate_results(self):
        self.result_tree.delete(*self.result_tree.get_children())
        for i, r in enumerate(self.results):
            url_short = r["url"][-60:] if len(r["url"]) > 60 else r["url"]
            date_short = str(r["date"])[:10]
            self.result_tree.insert("", tk.END, iid=str(i), values=(
                r["unit_id"], r["company"], r["model"],
                r["description"], r["unit_type"], date_short, url_short,
            ))

    # ── Preview ───────────────────────────────────────────────────────────────

    def _on_result_select(self, event=None):
        sel = self.result_tree.selection()
        if not sel:
            return
        r = self.results[int(sel[0])]
        self.preview_desc.config(text=r.get("description", ""))
        url = r.get("url", "")
        self.preview_url.config(text=url[:80] + "..." if len(url) > 80 else url)
        self._thumb_placeholder("Loading...")
        fetch_thumbnail(url, self._on_thumb_ready)

    def _preview_selected(self):
        sel = self.result_tree.selection()
        if not sel:
            return
        url = self.results[int(sel[0])].get("url", "")
        if url:
            import webbrowser
            webbrowser.open(url)

    def _on_thumb_ready(self, pil_image):
        self.win.after(0, lambda: self._show_thumbnail(pil_image))

    def _show_thumbnail(self, pil_image):
        self.thumb_canvas.delete("all")
        if pil_image is None:
            self._thumb_placeholder("Could not load")
            return
        self._thumb_img = ImageTk.PhotoImage(pil_image)
        x = (240 - pil_image.width)  // 2
        y = (180 - pil_image.height) // 2
        self.thumb_canvas.create_image(x, y, anchor=tk.NW, image=self._thumb_img)

    def _thumb_placeholder(self, msg="Select a result"):
        self.thumb_canvas.delete("all")
        self._thumb_img = None
        self.thumb_canvas.create_text(120, 90, text=msg, fill=GRAY,
                                      font=("Helvetica", 10), justify=tk.CENTER)

    # ── Add to Index ──────────────────────────────────────────────────────────

    def _add_selected(self):
        sel = self.result_tree.selection()
        if not sel:
            messagebox.showinfo("No selection",
                                "Select one or more results first.", parent=self.win)
            return
        rows_to_add = [self.results[int(i)] for i in sel]
        self._commit_to_index(rows_to_add)

    def _add_all(self):
        if not self.results:
            messagebox.showinfo("No results", "Run a search first.", parent=self.win)
            return
        self._commit_to_index(self.results)

    def _commit_to_index(self, rows_to_add):
        existing_rows = load_index(self.index_path)
        existing_urls = {r.get("url", "") for r in existing_rows}
        new_id = max((int(r.get("image_id", 0)) for r in existing_rows), default=0)

        added = 0
        for r in rows_to_add:
            url = r.get("url", "").strip()
            if not url or url in existing_urls:
                continue
            new_id += 1
            existing_rows.append({
                "image_id":     new_id,
                "company_id":   r.get("company_id", ""),
                "company_name": r.get("company", ""),
                "job_id":       r.get("visit_id", ""),
                "url":          url,
                "status":       "pending",
                "annotated_by": "",
                "date":         str(r.get("date", ""))[:10],
                "notes":        r.get("description", ""),
            })
            existing_urls.add(url)
            added += 1

        save_index(self.index_path, existing_rows)
        self.existing_rows = existing_rows

        if self.on_added:
            self.on_added(added)

        messagebox.showinfo("Added",
                            f"{added} image(s) added to index.",
                            parent=self.win)
        if added:
            self.win.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Data Browser — onesvs")
    parser.add_argument("--index", default=DEFAULT_INDEX,
                        help="Path to index.csv")
    parser.add_argument("--json", default=DEFAULT_JSON,
                        help="Path to annotations.json")
    args = parser.parse_args()

    # Ensure data directory exists
    for path in (args.index, args.json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    root = tk.Tk()
    app  = DataBrowser(root, args.index, args.json)
    root.mainloop()


if __name__ == "__main__":
    main()
