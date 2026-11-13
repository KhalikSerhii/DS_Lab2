"""Запускає всі експерименти: 3 стратегії x 4 режими розбиття = 12 запусків, потім будує графіки.

Приклади:
    python run_experiments.py --quick          # швидка перевірка, що все працює
    python run_experiments.py                  # повний прогін
    python run_experiments.py --strategies FedAvg --modes A_equal_balanced
"""
import argparse
import json
import time

import numpy as np
import torch

import config
import data
import make_splits
import plots
from fl import run_experiment


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=config.ROUNDS)
    ap.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    ap.add_argument("--n-train", type=int, default=config.N_TRAIN)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--test-limit", type=int, default=0, help="обмежити кількість тест-зображень (0 = усі)")
    ap.add_argument("--strategies", nargs="+", default=config.STRATEGIES, choices=config.STRATEGIES)
    ap.add_argument("--modes", nargs="+", default=config.MODES, choices=config.MODES)
    ap.add_argument("--quick", action="store_true", help="мінімальний прогін для перевірки")
    ap.add_argument("--force", action="store_true", help="перезапустити, навіть якщо результат уже є")
    return ap.parse_args()


def main():
    a = parse_args()
    if a.quick:
        a.rounds, a.n_train, a.test_limit = 2, 64, 35

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("УВАГА: CUDA недоступна, навчання на CPU буде дуже повільним. "
              "Перевірте встановлення PyTorch з підтримкою CUDA.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    ds = data.load_cache()
    test_idx = [i for i, t in enumerate(ds["is_test"]) if t]
    if a.test_limit and a.test_limit < len(test_idx):
        rng = np.random.default_rng(config.SEED)
        test_idx = sorted(rng.choice(test_idx, size=a.test_limit, replace=False).tolist())
    test_set = data.Subset(ds, test_idx)
    print(f"Класи: {ds['classes']}\nТест: {len(test_set)} зображень")

    splits, table = make_splits.make_all(ds, a.n_train, config.SEED)
    print("\nРозбиття між воркерами:")
    print(table.to_string(index=False))
    plots.plot_splits()

    cfg = {
        "rounds": a.rounds, "local_epochs": a.local_epochs, "batch_size": a.batch_size,
        "n_train": a.n_train, "test_limit": a.test_limit, "lr": config.LR,
        "mu": config.FEDPROX_MU, "fedadam": config.FEDADAM, "img_size": config.IMG_SIZE,
        "seed": config.SEED,
    }
    cfg_json = json.loads(json.dumps(cfg))
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    for mode in a.modes:
        for strategy in a.strategies:
            out = config.RUNS_DIR / f"{strategy}__{mode}.json"
            if out.exists() and not a.force:
                old = json.loads(out.read_text(encoding="utf-8"))
                if old.get("config") == cfg_json:
                    print(f"\n== {strategy} / {mode}: уже є результат, пропускаю (--force щоб перезапустити)")
                    continue
            print(f"\n== {strategy} / {mode}")
            t0 = time.time()
            history = run_experiment(strategy, mode, splits[mode], ds, test_set, cfg, device)
            out.write_text(json.dumps({
                "strategy": strategy, "mode": mode, "config": cfg_json, "classes": ds["classes"],
                "history": history, "time_sec": time.time() - t0,
            }, ensure_ascii=False), encoding="utf-8")
            print(f"   готово за {time.time() - t0:.0f} с")

    print(f"\nЗагальний час: {(time.time() - t_all) / 60:.1f} хв")
    plots.make_all()


if __name__ == "__main__":
    main()
