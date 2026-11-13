"""Візуалізація: розбиття даних, прогрес навчання, фінальне порівняння, AP по класах."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import config

COLORS = {"FedAvg": "#0072B2", "FedProx": "#E69F00", "FedAdam": "#009E73"}


def _style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
    })


def _load_runs():
    runs = []
    for p in sorted(config.RUNS_DIR.glob("*.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


def _index(runs):
    return {(r["strategy"], r["mode"]): r for r in runs}


def _save(fig, name):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- розбиття
def plot_splits():
    _style()
    splits = []
    for mode in config.MODES:
        p = config.SPLITS_DIR / f"{mode}.json"
        if p.exists():
            splits.append(json.loads(p.read_text(encoding="utf-8")))
    if not splits:
        return None
    classes = splits[0]["classes"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(splits), figsize=(3.6 * len(splits), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, splits):
        bottom = np.zeros(2)
        for ci, cname in enumerate(classes):
            vals = np.array([w["class_counts"][ci] for w in s["workers"]])
            ax.bar([0, 1], vals, bottom=bottom, color=cmap(ci % 10), label=cname, width=0.6)
            bottom += vals
        for x, tot in enumerate(bottom):
            ax.text(x, tot, str(int(tot)), ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["воркер 0", "воркер 1"])
        ax.set_title(config.MODE_TITLES[s["mode"]], fontsize=9)
    axes[0].set_ylabel("кількість зображень")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7), frameon=False)
    fig.suptitle("Розбиття train-даних між воркерами", y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return _save(fig, "splits_distribution.png")


# ---------------------------------------------------------------- прогрес
def plot_progress(runs, key, ylabel, title, fname):
    _style()
    idx = _index(runs)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, mode in zip(axes.ravel(), config.MODES):
        for s in config.STRATEGIES:
            r = idx.get((s, mode))
            if not r:
                continue
            x = [h["round"] for h in r["history"]]
            y = [h[key] for h in r["history"]]
            ax.plot(x, y, marker="o", lw=2, color=COLORS[s], label=s)
        ax.set_title(config.MODE_TITLES[mode], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes[-1]:
        ax.set_xlabel("раунд федеративного навчання")
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, l
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, fname)


# ---------------------------------------------------------------- підсумок
def plot_final(runs):
    _style()
    idx = _index(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(config.MODES))
    w = 0.26
    for ax, key, name in zip(axes, ("map50", "map5095"), ("mAP@0.5", "mAP@0.5:0.95")):
        for k, s in enumerate(config.STRATEGIES):
            vals = [idx[(s, m)]["history"][-1][key] if (s, m) in idx else np.nan for m in config.MODES]
            xs = x + (k - 1) * w
            ax.bar(xs, vals, w, color=COLORS[s], label=s)
            for xi, v in zip(xs, vals):
                if not np.isnan(v):
                    ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([config.MODE_SHORT[m] for m in config.MODES], fontsize=9)
        ax.set_title(f"Фінальний {name} на спільному тесті")
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "final_comparison.png")


def plot_per_class(runs):
    _style()
    idx = _index(runs)
    rows, labels = [], []
    classes = runs[0]["classes"]
    for m in config.MODES:
        for s in config.STRATEGIES:
            r = idx.get((s, m))
            if r:
                v = r["history"][-1]["ap50_per_class"]
                rows.append([np.nan if t is None else t for t in v])
                labels.append(f"{m.split('_')[0]} · {s}")
    if not rows:
        return None
    mat = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 3, 0.38 * len(rows) + 2))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="AP@0.5")
    ax.set_title("AP@0.5 по класах (фінальний раунд)")
    fig.tight_layout()
    return _save(fig, "per_class_ap.png")


def make_all():
    runs = _load_runs()
    plot_splits()
    if not runs:
        print("Немає результатів у results/runs")
        return
    plot_progress(runs, "map50", "mAP@0.5", "Прогрес навчання: mAP@0.5 по раундах", "progress_map50.png")
    plot_progress(runs, "map5095", "mAP@0.5:0.95", "Прогрес навчання: mAP@0.5:0.95 по раундах", "progress_map5095.png")
    plot_progress(runs, "train_loss", "train loss (середня по воркерах)", "Втрата навчання по раундах", "progress_loss.png")
    plot_final(runs)
    plot_per_class(runs)

    rows = []
    for r in runs:
        last = r["history"][-1]
        rows.append({"режим": r["mode"], "стратегія": r["strategy"], "mAP50": round(last["map50"], 4),
                     "mAP50:95": round(last["map5095"], 4), "train_loss": round(last["train_loss"], 4),
                     "час, с": round(r["time_sec"], 1)})
    df = pd.DataFrame(rows).sort_values(["режим", "стратегія"])
    df.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nГрафіки збережено в {config.PLOTS_DIR}")


if __name__ == "__main__":
    make_all()
