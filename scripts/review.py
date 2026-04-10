"""
Model Review — Visual ML Workflow Builder

Loads best.pt from the latest experiment, runs validation, and displays:
  - Per-class precision / recall / mAP50 table
  - Existing confusion matrix and result plots
  - Sample validation prediction images

Run from ml_system/:
    python3 scripts/review.py --project svs_plumbing
    python3 scripts/review.py --project svs_plumbing --exp exp_001_20260318_161743
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_experiment(project):
    exp_dir = Path("projects") / project / "experiments"
    if not exp_dir.exists():
        return None, None
    candidates = [
        d for d in sorted(exp_dir.iterdir())
        if d.is_dir() and (d / "results.json").exists()
    ]
    return (candidates[-1], exp_dir) if candidates else (None, exp_dir)


def find_best_pt(exp_folder_name, project):
    """Search known locations for best.pt matching this experiment."""
    # Explicit known paths first (fast)
    candidates = [
        Path("runs") / "detect" / "projects" / project / "experiments" / exp_folder_name / "train" / "weights" / "best.pt",
        Path("/mnt/c/Users/salbe/Downloads/FiltrexAIDemo/runs/detect/projects") / project / "experiments" / exp_folder_name / "train" / "weights" / "best.pt",
        Path("/mnt/c/Users/salbe/Downloads/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def open_image(path):
    """Open an image file with the system viewer (WSL-friendly)."""
    path = str(path)
    for cmd in [["explorer.exe", path.replace("/mnt/c/", "C:\\").replace("/", "\\")],
                ["eog", path], ["display", path], ["xdg-open", path]]:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


def print_table(headers, rows, col_widths=None):
    if not col_widths:
        col_widths = [max(len(str(r[i])) for r in ([headers] + rows)) + 2
                      for i in range(len(headers))]
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    sep = "  ".join("-" * w for w in col_widths)
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="svs_plumbing",
                        help="Project folder under projects/")
    parser.add_argument("--exp", default=None, help="Specific experiment folder name")
    parser.add_argument("--open-plots", action="store_true",
                        help="Open confusion matrix and result plots in image viewer")
    parser.add_argument("--validate", action="store_true",
                        help="Run per-class validation (slow on CPU, fast on GPU)")
    args = parser.parse_args()

    # ── Find experiment ───────────────────────────────────────────────────────
    exp_dir = Path("projects") / args.project / "experiments"
    if args.exp:
        exp_path = exp_dir / args.exp
    else:
        exp_path, _ = find_latest_experiment(args.project)

    if not exp_path or not exp_path.exists():
        print(f"No experiment found for project: {args.project}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Model Review — {args.project}")
    print(f"  Experiment: {exp_path.name}")
    print(f"{'='*60}\n")

    # ── Load config + summary results ─────────────────────────────────────────
    config_path  = exp_path / "config.json"
    results_path = exp_path / "results.json"

    with open(config_path)  as f: config  = json.load(f)
    with open(results_path) as f: summary = json.load(f)

    print("── Training Config ──────────────────────────────────────")
    print(f"  Model   : {config.get('model')}")
    print(f"  Dataset : {config.get('dataset_path')}")
    print(f"  Epochs  : {config.get('epochs')}  |  Batch: {config.get('batch')}  |  imgsz: {config.get('imgsz')}")
    print(f"  Classes : {', '.join(config.get('classes', []))}")
    print()
    print("── Overall Results ──────────────────────────────────────")
    print(f"  mAP@50      : {summary.get('mAP50', 'N/A'):.4f}")
    print(f"  mAP@50-95   : {summary.get('mAP50-95', 'N/A'):.4f}")
    print(f"  Precision   : {summary.get('precision', 'N/A'):.4f}")
    print(f"  Recall      : {summary.get('recall', 'N/A'):.4f}")
    print()

    # ── Per-class metrics from results.json ───────────────────────────────────
    per_class = summary.get("per_class", [])
    if per_class:
        print("── Per-Class Metrics ────────────────────────────────────")
        rows = []
        for c in per_class:
            flag = " ← needs work" if c.get("mAP50", 1) < 0.6 else ""
            rows.append((
                c["class"],
                f"{c.get('precision', 0):.3f}",
                f"{c.get('recall', 0):.3f}",
                f"{c.get('mAP50', 0):.3f}{flag}",
            ))
        print_table(["Class", "Precision", "Recall", "mAP@50"],
                    rows, [24, 11, 9, 20])
        print()

    # ── Find best.pt ─────────────────────────────────────────────────────────
    best_pt = find_best_pt(exp_path.name, args.project)
    if best_pt:
        print(f"── Model Weights ─────────────────────────────────────────")
        print(f"  best.pt : {best_pt}")
        print()
    else:
        print("  [!] best.pt not found — per-class metrics from CSV only\n")

    # ── Per-class metrics from results.csv ───────────────────────────────────
    train_run = None
    if best_pt:
        train_run = best_pt.parent.parent  # weights/ → train/

    csv_path = train_run / "results.csv" if train_run else None
    if csv_path and csv_path.exists():
        print("── Training Progress (final epoch) ──────────────────────")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            # Strip whitespace from keys
            last = {k.strip(): v.strip() for k, v in last.items()}
            metric_keys = [
                ("metrics/precision(B)", "Precision"),
                ("metrics/recall(B)",    "Recall"),
                ("metrics/mAP50(B)",     "mAP@50"),
                ("metrics/mAP50-95(B)",  "mAP@50-95"),
                ("train/box_loss",       "Box Loss"),
                ("train/cls_loss",       "Cls Loss"),
                ("val/box_loss",         "Val Box Loss"),
                ("val/cls_loss",         "Val Cls Loss"),
            ]
            table_rows = []
            for key, label in metric_keys:
                if key in last:
                    try:
                        table_rows.append((label, f"{float(last[key]):.4f}"))
                    except ValueError:
                        table_rows.append((label, last[key]))
            print_table(["Metric", "Value"], table_rows, [20, 10])
        print()

    # ── Run per-class validation with YOLO ───────────────────────────────────
    if best_pt and not args.validate:
        print("── Per-Class Validation ─────────────────────────────────")
        print("  Skipped by default (slow on CPU).")
        print("  Run with --validate to get per-class precision/recall/mAP.\n")

    if best_pt and args.validate:
        print("── Per-Class Validation ─────────────────────────────────")
        dataset_path = Path("projects") / args.project / config["dataset_path"]
        yaml_path    = dataset_path / "data.yaml"

        if not yaml_path.exists():
            print(f"  [!] data.yaml not found at {yaml_path} — skipping per-class validation")
        else:
            try:
                from ultralytics import YOLO
                model = YOLO(str(best_pt))
                print("  Running validation... (this may take a minute)\n")
                val_results = model.val(data=str(yaml_path), verbose=False)

                class_names = config.get("classes", [])
                # val_results.box.ap_class_index, .p, .r, .ap50
                ap_idx  = val_results.box.ap_class_index.tolist() if hasattr(val_results.box, "ap_class_index") else []
                prec    = val_results.box.p.tolist()   if hasattr(val_results.box, "p")   else []
                recall  = val_results.box.r.tolist()   if hasattr(val_results.box, "r")   else []
                ap50    = val_results.box.ap50.tolist() if hasattr(val_results.box, "ap50") else []

                # Build class name lookup from data.yaml
                import yaml as _yaml
                with open(yaml_path) as yf:
                    yaml_data = _yaml.safe_load(yf)
                yaml_names = yaml_data.get("names", class_names)

                rows = []
                for i, cls_idx in enumerate(ap_idx):
                    name = yaml_names[cls_idx] if cls_idx < len(yaml_names) else f"cls_{cls_idx}"
                    p  = f"{prec[i]:.3f}"   if i < len(prec)   else "—"
                    r  = f"{recall[i]:.3f}" if i < len(recall) else "—"
                    m  = f"{ap50[i]:.3f}"   if i < len(ap50)   else "—"
                    flag = " ← low" if i < len(ap50) and ap50[i] < 0.5 else ""
                    rows.append((name, p, r, m + flag))

                print_table(["Class", "Precision", "Recall", "mAP@50"],
                            rows, [22, 11, 9, 14])
                print()

                # Highlight weak classes
                weak = [(yaml_names[ap_idx[i]], ap50[i])
                        for i in range(len(ap_idx)) if i < len(ap50) and ap50[i] < 0.5]
                if weak:
                    print("  Classes needing more annotation:")
                    for name, score in sorted(weak, key=lambda x: x[1]):
                        print(f"    ⚠  {name:25s}  mAP@50 = {score:.3f}")
                    print()
                else:
                    print("  All classes above 0.50 mAP@50 ✓\n")

            except ImportError:
                print("  ultralytics not installed — run: pip3 install ultralytics --break-system-packages")
            except Exception as e:
                print(f"  Validation error: {e}")

    # ── Surface existing plots ────────────────────────────────────────────────
    plots = []
    if train_run:
        for name in ["confusion_matrix_normalized.png", "BoxPR_curve.png",
                     "results.png", "val_batch0_pred.jpg"]:
            p = train_run / name
            if p.exists():
                plots.append(p)

    if plots:
        print("── Available Plots ──────────────────────────────────────")
        for p in plots:
            print(f"  {p.name:40s}  {p}")
        print()

        if args.open_plots:
            print("  Opening plots...")
            for p in plots:
                open_image(p)
        else:
            print("  Tip: run with --open-plots to open them in your image viewer")
            print()

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
