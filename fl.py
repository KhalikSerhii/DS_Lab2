"""Федеративне навчання: клієнти (воркери), стратегії FedAvg / FedProx / FedAdam, цикл раундів.

Це симуляція в одному процесі: воркери навчаються по черзі на одній GPU, кожен бачить
лише свій Subset. Сервер бачить тільки ваги, що повертають воркери.
"""
import random
import time

import numpy as np
import torch

import config
from data import Subset
from evaluate import evaluate_model
from model import build_model, get_params, set_params, trainable_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- клієнт
def local_train(model, subset, global_params, epochs, batch_size, mu, device, rng):
    set_params(model, global_params)
    model.train()
    params = trainable_params(model)
    ref = [g.to(device) for g in global_params] if mu > 0 else None
    opt = torch.optim.SGD(params, lr=config.LR, momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)
    total, steps = 0.0, 0
    for _ in range(epochs):
        for imgs, tgts in subset.batches(batch_size, shuffle=True, augment=True, rng=rng):
            imgs = [i.to(device).float().div_(255.0) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ref is not None:  # FedProx: градієнт доданка mu/2 * ||w - w_global||^2
                for p, g in zip(params, ref):
                    if p.grad is not None:
                        p.grad.add_(p.detach() - g, alpha=mu)
            torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP)
            opt.step()
            total += loss.item()
            steps += 1
    return get_params(model), total / max(steps, 1)


class Client:
    def __init__(self, cid, subset):
        self.cid = cid
        self.data = subset

    def fit(self, model, global_params, epochs, batch_size, mu, device, round_idx):
        rng = np.random.default_rng(config.SEED + 1000 * round_idx + self.cid)
        params, loss = local_train(model, self.data, global_params, epochs, batch_size, mu, device, rng)
        return {"params": params, "n": len(self.data), "loss": loss}


# ---------------------------------------------------------------- стратегії
def weighted_average(results):
    total = sum(r["n"] for r in results)
    out = [torch.zeros_like(p) for p in results[0]["params"]]
    for r in results:
        w = r["n"] / total
        for o, p in zip(out, r["params"]):
            o.add_(p, alpha=w)
    return out


class FedAvg:
    name = "FedAvg"
    mu = 0.0

    def aggregate(self, global_params, results):
        return weighted_average(results)


class FedProx(FedAvg):
    """Агрегація як у FedAvg; відмінність у локальній втраті (proximal term) на клієнті."""
    name = "FedProx"

    def __init__(self, mu):
        self.mu = mu


class FedAdam:
    """Adaptive Federated Optimization (Reddi et al., 2021): Adam на псевдоградієнті."""
    name = "FedAdam"
    mu = 0.0

    def __init__(self, server_lr, beta1, beta2, tau):
        self.lr, self.b1, self.b2, self.tau = server_lr, beta1, beta2, tau
        self.m = None
        self.v = None

    def aggregate(self, global_params, results):
        avg = weighted_average(results)
        if self.m is None:
            self.m = [torch.zeros_like(p) for p in global_params]
            self.v = [torch.full_like(p, self.tau ** 2) for p in global_params]
        new = []
        for i, (g, a) in enumerate(zip(global_params, avg)):
            delta = a - g
            self.m[i].mul_(self.b1).add_(delta, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(delta, delta, value=1 - self.b2)
            new.append(g + self.lr * self.m[i] / (self.v[i].sqrt() + self.tau))
        return new


def make_strategy(name):
    if name == "FedAvg":
        return FedAvg()
    if name == "FedProx":
        return FedProx(config.FEDPROX_MU)
    if name == "FedAdam":
        return FedAdam(**config.FEDADAM)
    raise ValueError(name)


# ---------------------------------------------------------------- один експеримент
def run_experiment(strategy_name, mode_name, split, ds, test_set, cfg, device):
    set_seed(config.SEED)
    num_classes = len(ds["classes"])
    model = build_model(num_classes, config.IMG_SIZE).to(device)
    global_params = get_params(model)
    clients = [Client(i, Subset(ds, w["indices"])) for i, w in enumerate(split["workers"])]
    strategy = make_strategy(strategy_name)

    history = []
    for r in range(1, cfg["rounds"] + 1):
        t0 = time.time()
        results = [
            c.fit(model, global_params, cfg["local_epochs"], cfg["batch_size"], strategy.mu, device, r)
            for c in clients
        ]
        global_params = strategy.aggregate(global_params, results)
        set_params(model, global_params)
        metrics = evaluate_model(model, test_set, device, num_classes)
        n_all = sum(x["n"] for x in results)
        loss = sum(x["loss"] * x["n"] for x in results) / n_all
        history.append({"round": r, "train_loss": loss, **metrics})
        print(f"  [{strategy_name:7s} | {mode_name}] раунд {r}/{cfg['rounds']}  "
              f"loss={loss:.3f}  mAP50={metrics['map50']:.3f}  "
              f"mAP50:95={metrics['map5095']:.3f}  ({time.time() - t0:.1f}с)", flush=True)

    del model, clients
    if device == "cuda":
        torch.cuda.empty_cache()
    return history
