import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")  # no UI
import matplotlib.pyplot as plt


def load_experiments(project):
    experiments_dir = os.path.join("projects", project, "experiments")
    records = []

    if not os.path.exists(experiments_dir):
        print(f"No experiments found for project: {project}")
        return records

    for exp_folder in sorted(os.listdir(experiments_dir)):
        exp_path = os.path.join(experiments_dir, exp_folder)
        config_path = os.path.join(exp_path, "config.json")
        results_path = os.path.join(exp_path, "results.json")

        if not os.path.exists(config_path) or not os.path.exists(results_path):
            continue

        with open(config_path) as f:
            config = json.load(f)

        with open(results_path) as f:
            metrics = json.load(f)

        records.append({
            "experiment": config["experiment_name"],
            "dataset": os.path.basename(config["dataset_path"]),
            "mAP50": round(metrics.get("mAP50") or 0, 4),
            "mAP50-95": round(metrics.get("mAP50-95") or 0, 4),
        })

    return records


def plot_map_comparison(project, records):
    output_path = os.path.join("projects", project, "experiments", "map_comparison.png")

    labels = [r["experiment"] for r in records]
    scores = [r["mAP50"] for r in records]
    datasets = [r["dataset"] for r in records]

    unique_datasets = sorted(set(datasets))
    palette = ["#4C9BE8", "#E87B4C", "#6BCB77", "#FFD93D"]
    color_map = {d: palette[i % len(palette)] for i, d in enumerate(unique_datasets)}
    colors = [color_map[d] for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=colors)

    for bar, score, dataset in zip(bars, scores, datasets):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score}\n({dataset})",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP@50")
    ax.set_title(f"Experiment Comparison — {project} — mAP@50")
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, label="0.80 threshold")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[d], label=d)
        for d in unique_datasets
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, label="0.80 threshold")
    ])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="svs_plumbing")
    args = parser.parse_args()

    records = load_experiments(args.project)
    for r in records:
        print(f"{r['experiment']} → {r['dataset']} → mAP {r['mAP50']}")
    if records:
        plot_map_comparison(args.project, records)
