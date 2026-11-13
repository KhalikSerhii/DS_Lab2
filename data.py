"""Доступ до кешованого датасету. Subset — це "локальні дані" одного воркера."""
import numpy as np
import torch

import config


def load_cache():
    path = config.CACHE_DIR / "dataset.pt"
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено {path}. Спочатку запустіть prepare_data.py")
    return torch.load(path, weights_only=False)


class Subset:
    """Копіює посилання лише на свої зображення. Воркер не має доступу до решти датасету."""

    def __init__(self, ds, indices):
        self.indices = list(indices)
        self.images = [ds["images"][i] for i in self.indices]
        self.boxes = [ds["boxes"][i] for i in self.indices]
        self.labels = [ds["labels"][i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def gts(self):
        return [{"boxes": b.numpy(), "labels": l.numpy()} for b, l in zip(self.boxes, self.labels)]

    def batches(self, batch_size, shuffle=False, augment=False, rng=None):
        order = np.arange(len(self))
        if shuffle:
            if rng is None:
                rng = np.random.default_rng(0)
            rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            imgs, tgts = [], []
            for k in order[s:s + batch_size]:
                img = self.images[k]
                boxes = self.boxes[k].clone()
                if augment and rng is not None and rng.random() < 0.5:
                    w = img.shape[2]
                    img = img.flip(-1)
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                imgs.append(img)
                tgts.append({"boxes": boxes, "labels": self.labels[k]})
            yield imgs, tgts
