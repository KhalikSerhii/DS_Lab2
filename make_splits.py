"""Розбиття train-пулу на двох воркерів у 4 режимах. Результат: splits/<режим>.json

Запуск (опційно, run_experiments.py робить це сам):  python make_splits.py
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
import data


def _apportion(n, w):
    """Розподіл n одиниць за вагами w цілими числами (метод найбільших залишків)."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    raw = w * n
    q = np.floor(raw).astype(int)
    rem = int(n - q.sum())
    if rem > 0:
        q[np.argsort(-(raw - q))[:rem]] += 1
    return q


def quotas(mode, n_train, n_classes):
    uni = np.ones(n_classes)
    if mode == "A_equal_balanced":
        s0 = n_train // 2
        sizes, w = [s0, n_train - s0], [uni, uni]
    elif mode == "B_unequal_balanced":
        s0 = int(round(0.7 * n_train))
        sizes, w = [s0, n_train - s0], [uni, uni]
    elif mode == "C_unequal_imbalanced":
        s0 = int(round(0.7 * n_train))
        g = config.IMBALANCE_RATIO ** np.arange(n_classes)
        sizes, w = [s0, n_train - s0], [g, g[::-1]]
    elif mode == "D_missing_classes":
        if n_classes < 4:
            raise ValueError("Для режиму D потрібно щонайменше 4 класи")
        s0 = n_train // 2
        w0, w1 = uni.copy(), uni.copy()
        w0[[0, 1]] = 0   # воркер 0 не має класів 0 і 1
        w1[[2, 3]] = 0   # воркер 1 не має класів 2 і 3
        sizes, w = [s0, n_train - s0], [w0, w1]
    else:
        raise ValueError(mode)
    return [_apportion(s, wi) for s, wi in zip(sizes, w)]


def build_split(ds, mode, n_train, seed):
    n_classes = len(ds["classes"])
    rng = np.random.default_rng(seed)
    pool = [[] for _ in range(n_classes)]
    for i, (c, is_test) in enumerate(zip(ds["cls"], ds["is_test"])):
        if not is_test:
            pool[c].append(i)
    for c in range(n_classes):
        rng.shuffle(pool[c])

    q = quotas(mode, n_train, n_classes)
    cursor = [0] * n_classes
    workers = []
    for k in range(2):
        idxs, counts = [], []
        for c in range(n_classes):
            take = min(int(q[k][c]), len(pool[c]) - cursor[c])
            if take < q[k][c]:
                print(f"  ! режим {mode}: воркеру {k} бракує зображень класу {ds['classes'][c]}")
            idxs += pool[c][cursor[c]:cursor[c] + take]
            cursor[c] += take
            counts.append(int(take))
        workers.append({"worker": k, "indices": [int(i) for i in idxs], "class_counts": counts})
    return {"mode": mode, "classes": ds["classes"], "workers": workers}


def save_split(split, ds):
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(split)
    out["workers"] = [dict(w, files=[ds["names"][i] for i in w["indices"]]) for w in split["workers"]]
    path = config.SPLITS_DIR / f"{split['mode']}.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return path


def stats_table(splits):
    rows = []
    for s in splits:
        for w in s["workers"]:
            row = {"режим": s["mode"], "воркер": w["worker"], "зображень": len(w["indices"])}
            row.update({c: n for c, n in zip(s["classes"], w["class_counts"])})
            rows.append(row)
    return pd.DataFrame(rows)


def make_all(ds, n_train, seed=config.SEED, modes=None):
    splits = {}
    for mode in modes or config.MODES:
        split = build_split(ds, mode, n_train, seed)
        save_split(split, ds)
        splits[mode] = split
    df = stats_table(list(splits.values()))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "split_stats.csv", index=False)
    return splits, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=config.N_TRAIN)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    ds = data.load_cache()
    _, table = make_all(ds, a.n_train, a.seed)
    print(table.to_string(index=False))
    import plots
    plots.plot_splits()
    print("Графік: results/plots/splits_distribution.png")
