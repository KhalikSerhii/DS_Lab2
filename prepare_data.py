"""Читає розпакований датасет, автоматично визначає формат анотацій, зменшує зображення
і зберігає компактний кеш cache/dataset.pt (його потім використовують усі експерименти).

Підтримувані формати анотацій: Pascal VOC (xml), COCO (один json), YOLO (txt),
окремі json на кожне зображення (з полями на кшталт name/label + xmin,ymin,xmax,ymax
або x1,y1,x2,y2 або bbox).

Для простоти розбиття на класи беруться лише зображення, де всі об'єкти одного класу.

Запуск:  python prepare_data.py --root "шлях/до/датасету"
"""
import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOX_KEYSETS = [("xmin", "ymin", "xmax", "ymax"), ("x1", "y1", "x2", "y2"),
               ("left", "top", "right", "bottom")]
NAME_KEYS = ("name", "label", "class", "class_name", "category", "category_name", "tag", "order")


# ---------------------------------------------------------------- пошук файлів
def collect_files(root):
    by_ext = defaultdict(list)
    for p in Path(root).rglob("*"):
        if p.is_file():
            by_ext[p.suffix.lower()].append(p)
    return by_ext


def index_images(by_ext):
    by_dir_stem, by_stem = {}, {}
    for ext in IMG_EXT:
        for p in by_ext.get(ext, []):
            by_dir_stem[(p.parent, p.stem)] = p
            by_stem.setdefault(p.stem, p)
    return by_dir_stem, by_stem


def find_image(ann_path, stem, idx):
    by_dir_stem, by_stem = idx
    return by_dir_stem.get((ann_path.parent, stem)) or by_stem.get(stem)


# ---------------------------------------------------------------- Pascal VOC
def load_voc(xmls, idx):
    recs = []
    for p in xmls:
        try:
            r = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        objs = []
        for o in r.iter("object"):
            name, bb = o.findtext("name"), o.find("bndbox")
            if not name or bb is None:
                continue
            try:
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError):
                continue
            objs.append((name.strip(), x1, y1, x2, y2))
        if not objs:
            continue
        img = find_image(p, p.stem, idx)
        if img is None and r.findtext("filename"):
            img = find_image(p, Path(r.findtext("filename")).stem, idx)
        if img is not None:
            recs.append({"img": img, "objs": objs, "norm": False})
    return recs


# ---------------------------------------------------------------- JSON
def _box_from_dict(d):
    for ks in BOX_KEYSETS:
        if all(k in d for k in ks):
            try:
                return [float(d[k]) for k in ks]
            except (TypeError, ValueError):
                return None
    for k in ("bbox", "box", "bounding_box", "rect"):
        v = d.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(t, (int, float)) for t in v):
            x, y, w, h = map(float, v)  # припущення: [x, y, ширина, висота] як у COCO
            return [x, y, x + w, y + h]
    for kw, kh in (("width", "height"), ("w", "h")):
        if all(k in d for k in ("x", kw, "y", kh)):
            try:
                x, y, w, h = float(d["x"]), float(d["y"]), float(d[kw]), float(d[kh])
                return [x, y, x + w, y + h]
            except (TypeError, ValueError):
                return None
    pts = d.get("points")
    if isinstance(pts, list) and len(pts) >= 2 and all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pts):
        xs, ys = [float(q[0]) for q in pts], [float(q[1]) for q in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _collect(o, inherited, out, depth):
    if isinstance(o, dict):
        name = inherited
        if depth > 0:
            for k in NAME_KEYS:
                v = o.get(k)
                if isinstance(v, (str, int)) and not isinstance(v, bool):
                    name = str(v)
                    break
        box = _box_from_dict(o)
        if box is not None:
            out.append((name, box))
            return
        for v in o.values():
            _collect(v, name, out, depth + 1)
    elif isinstance(o, list):
        for v in o:
            _collect(v, inherited, out, depth + 1)


def load_coco(data, jpath, idx):
    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {i["id"]: i for i in data["images"]}
    per_img = defaultdict(list)
    for a in data["annotations"]:
        if a.get("bbox") and a["category_id"] in cats:
            x, y, w, h = a["bbox"]
            per_img[a["image_id"]].append((cats[a["category_id"]], x, y, x + w, y + h))
    recs = []
    for iid, objs in per_img.items():
        fn = imgs.get(iid, {}).get("file_name")
        img = find_image(jpath, Path(fn).stem, idx) if fn else None
        if img is not None:
            recs.append({"img": img, "objs": objs, "norm": False})
    return recs


def load_json_files(jsons, idx):
    # 1) COCO? (такий json один-два великих, тому дивимось лише на найбільші файли)
    big = sorted(jsons, key=lambda p: p.stat().st_size, reverse=True)[:3]
    coco_recs = []
    for p in big:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= set(data):
            coco_recs += load_coco(data, p, idx)
    if coco_recs:
        return coco_recs, "COCO (json)"

    # 2) окремий json на зображення
    recs = []
    for p in jsons:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out = []
        _collect(data, None, out, 0)
        if not out:
            continue
        img = find_image(p, p.stem, idx)
        if img is None and isinstance(data, dict):
            for k in ("image", "filename", "file_name", "imagePath", "imagename"):
                if isinstance(data.get(k), str):
                    img = find_image(p, Path(data[k]).stem, idx)
                    if img is not None:
                        break
        if img is None:
            continue
        norm = max(max(b) for _, b in out) <= 1.0
        objs = [(n if n is not None else p.parent.name, *b) for n, b in out]
        recs.append({"img": img, "objs": objs, "norm": norm})
    return recs, "JSON на кожне зображення"


# ---------------------------------------------------------------- VoTT JSON (ArTaxOr)
def load_vott(jsons, idx):
    """<Order>/annotations/<id>-asset.json: asset.name, regions[].tags, regions[].boundingBox."""
    recs = []
    for p in jsons:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "asset" not in d or "regions" not in d:
            continue
        objs = []
        for rg in d["regions"]:
            bb, tags = rg.get("boundingBox"), rg.get("tags") or []
            if not bb or not tags:
                continue
            try:
                x1, y1 = float(bb["left"]), float(bb["top"])
                x2, y2 = x1 + float(bb["width"]), y1 + float(bb["height"])
            except (KeyError, TypeError, ValueError):
                continue
            objs.append((str(tags[0]).strip(), x1, y1, x2, y2))
        name = d["asset"].get("name")
        if not objs or not name:
            continue
        stem = Path(name).stem
        # картинка лежить на рівень вище папки annotations
        img = idx[0].get((p.parent.parent, stem)) or idx[1].get(stem)
        if img is not None:
            recs.append({"img": img, "objs": objs, "norm": False})
    return recs


# ---------------------------------------------------------------- YOLO
def _yolo_names(by_ext):
    for p in by_ext.get(".txt", []) + by_ext.get(".names", []):
        if p.name.lower() in ("classes.txt", "obj.names", "labels.txt"):
            return [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    for p in by_ext.get(".yaml", []) + by_ext.get(".yml", []):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        m = re.search(r"names\s*:\s*\n((?:[ \t]+\d+\s*:\s*.+\n?)+)", txt)
        if m:
            items = re.findall(r"(\d+)\s*:\s*(.+)", m.group(1))
            return [v.strip().strip("'\"") for _, v in sorted(items, key=lambda t: int(t[0]))]
    return None


def load_yolo(txts, by_ext, idx):
    names = _yolo_names(by_ext)
    recs = []
    for p in txts:
        if p.name.lower() in ("classes.txt", "obj.names", "labels.txt", "readme.txt"):
            continue
        img = find_image(p, p.stem, idx)
        if img is None:
            continue
        objs = []
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                objs = []
                break
            name = names[c] if names and c < len(names) else str(c)
            objs.append((name, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if objs:
            recs.append({"img": img, "objs": objs, "norm": True})
    return recs


def load_annotations(root):
    by_ext = collect_files(root)
    idx = index_images(by_ext)
    if by_ext.get(".xml"):
        recs = load_voc(by_ext[".xml"], idx)
        if recs:
            return recs, "Pascal VOC (xml)"
    if by_ext.get(".json"):
        recs = load_vott(by_ext[".json"], idx)
        if recs:
            return recs, "VoTT (json)"
        recs, fmt = load_json_files(by_ext[".json"], idx)
        if recs:
            return recs, fmt
    if by_ext.get(".txt"):
        recs = load_yolo(by_ext[".txt"], by_ext, idx)
        if recs:
            return recs, "YOLO (txt)"
    return [], "невідомий"


# ---------------------------------------------------------------- кеш
def load_item(r, size):
    try:
        with Image.open(r["img"]) as im0:
            w0, h0 = im0.size
            im0.draft("RGB", (size * 2, size * 2))  # швидше декодування великих JPEG
            im = im0.convert("RGB")
    except Exception:
        return None
    s = size / max(w0, h0)
    nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    boxes = []
    for _, x1, y1, x2, y2 in r["objs"]:
        if r["norm"]:
            x1, x2, y1, y2 = x1 * w0, x2 * w0, y1 * h0, y2 * h0
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        b = [min(max(x1 * s, 0), nw), min(max(y1 * s, 0), nh),
             min(max(x2 * s, 0), nw), min(max(y2 * s, 0), nh)]
        if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
            boxes.append(b)
    if not boxes:
        return None
    img = torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous()
    return img, torch.tensor(boxes, dtype=torch.float32)


def build_cache(root, per_class, test_per_class, img_size, seed):
    records, fmt = load_annotations(root)
    print(f"Формат анотацій: {fmt}. Зображень з анотаціями: {len(records)}")
    if not records:
        sys.exit("Анотації не знайдено. Запустіть inspect_dataset.py і надішліть його вивід.")

    single = [r for r in records if len({o[0] for o in r["objs"]}) == 1]
    print(f"Зображень, де всі об'єкти одного класу: {len(single)} з {len(records)}")
    by_class = defaultdict(list)
    for r in single:
        by_class[r["objs"][0][0]].append(r)

    min_needed = test_per_class + 10
    classes = sorted(c for c, l in by_class.items() if len(l) >= min_needed)
    dropped = sorted(set(by_class) - set(classes))
    if dropped:
        print(f"Відкинуто класи з замалою кількістю зображень: {dropped}")
    print("Класи:", classes)
    if len(classes) < 4:
        sys.exit("Потрібно щонайменше 4 класи (для режиму D). Перевірте вивід inspect_dataset.py.")

    rng = random.Random(seed)
    ds = {"classes": classes, "images": [], "boxes": [], "labels": [], "cls": [],
          "is_test": [], "names": [], "img_size": img_size}
    for ci, cname in enumerate(classes):
        recs = by_class[cname][:]
        rng.shuffle(recs)
        loaded = 0
        pbar = tqdm(total=per_class, desc=f"{cname:15s}")
        for r in recs:
            item = load_item(r, img_size)
            if item is None:
                continue
            img, boxes = item
            ds["images"].append(img)
            ds["boxes"].append(boxes)
            ds["labels"].append(torch.full((len(boxes),), ci + 1, dtype=torch.int64))
            ds["cls"].append(ci)
            ds["is_test"].append(loaded < test_per_class)
            ds["names"].append(r["img"].stem)
            loaded += 1
            pbar.update(1)
            if loaded >= per_class:
                break
        pbar.close()

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ds, config.CACHE_DIR / "dataset.pt")
    n_test = sum(ds["is_test"])
    print(f"\nЗбережено cache/dataset.pt: {len(ds['images'])} зображень "
          f"(train-пул {len(ds['images']) - n_test}, спільний тест {n_test})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="папка з розпакованим датасетом")
    ap.add_argument("--per-class", type=int, default=config.PER_CLASS)
    ap.add_argument("--test-per-class", type=int, default=config.TEST_PER_CLASS)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    if not Path(a.root).exists():
        sys.exit(f"Папку {a.root} не знайдено")
    build_cache(a.root, a.per_class, a.test_per_class, a.img_size, a.seed)


if __name__ == "__main__":
    main()
