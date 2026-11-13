"""Власна оцінка детекції: AP по класах для IoU 0.5:0.05:0.95 (без pycocotools)."""
import numpy as np
import torch


def _iou_one_to_many(b, g):
    ix1 = np.maximum(b[0], g[:, 0])
    iy1 = np.maximum(b[1], g[:, 1])
    ix2 = np.minimum(b[2], g[:, 2])
    iy2 = np.minimum(b[3], g[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (b[2] - b[0]) * (b[3] - b[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def _average_precision(tp, n_gt):
    if len(tp) == 0:
        return 0.0
    fp = 1.0 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detections(preds, gts, num_classes):
    thrs = np.arange(0.5, 0.96, 0.05)
    ap = np.full((len(thrs), num_classes), np.nan)
    for c in range(1, num_classes + 1):
        gt_c = [g["boxes"][g["labels"] == c] for g in gts]
        n_gt = sum(len(b) for b in gt_c)
        if n_gt == 0:
            continue
        dets = []
        for i, p in enumerate(preds):
            m = p["labels"] == c
            for box, s in zip(p["boxes"][m], p["scores"][m]):
                dets.append((float(s), i, box))
        dets.sort(key=lambda d: -d[0])
        best_iou = np.zeros(len(dets))
        best_j = np.full(len(dets), -1)
        for k, (_, i, box) in enumerate(dets):
            if len(gt_c[i]):
                ious = _iou_one_to_many(box, gt_c[i])
                j = int(ious.argmax())
                best_iou[k], best_j[k] = ious[j], j
        for ti, thr in enumerate(thrs):
            used = [np.zeros(len(b), bool) for b in gt_c]
            tp = np.zeros(len(dets))
            for k, (_, i, _) in enumerate(dets):
                j = best_j[k]
                if j >= 0 and best_iou[k] >= thr and not used[i][j]:
                    used[i][j] = True
                    tp[k] = 1.0
            ap[ti, c - 1] = _average_precision(tp, n_gt)
    map50 = float(np.nanmean(ap[0])) if not np.all(np.isnan(ap[0])) else 0.0
    map5095 = float(np.nanmean(ap)) if not np.all(np.isnan(ap)) else 0.0
    return {
        "map50": map50,
        "map5095": map5095,
        "ap50_per_class": [None if np.isnan(v) else float(v) for v in ap[0]],
    }


@torch.no_grad()
def evaluate_model(model, test_set, device, num_classes, batch_size=16):
    model.eval()
    preds = []
    for imgs, _ in test_set.batches(batch_size, shuffle=False):
        imgs = [i.to(device).float().div_(255.0) for i in imgs]
        for o in model(imgs):
            preds.append({k: o[k].detach().cpu().numpy() for k in ("boxes", "scores", "labels")})
    return evaluate_detections(preds, test_set.gts(), num_classes)
