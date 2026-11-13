"""Модель: Faster R-CNN з MobileNetV3-Large FPN (320), ваги COCO, нова голова під наші класи.

Backbone заморожений (економія часу). BatchNorm у ньому FrozenBatchNorm, тому буферів
статистики немає, і між воркерами та сервером обмінюються лише навчувані параметри.
"""
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, img_size):
    try:
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W

        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=W.COCO_V1,
            min_size=img_size,
            max_size=img_size,           # зображення вже зменшені, повторного масштабування не буде
            box_detections_per_img=20,
            box_score_thresh=0.02,
            rpn_pre_nms_top_n_train=500,
            rpn_post_nms_top_n_train=300,
            box_batch_size_per_image=128,
        )
    except Exception as e:  # найімовірніше немає інтернету для завантаження ваг
        raise RuntimeError(
            "Не вдалося створити модель / завантажити ваги COCO. "
            f"Перевірте інтернет і версію torchvision (потрібна >=0.13). Помилка: {e}"
        ) from e

    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)  # +1 фон
    for p in model.backbone.body.parameters():
        p.requires_grad_(False)
    return model


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def get_params(model):
    return [p.detach().cpu().clone() for p in trainable_params(model)]


def set_params(model, params):
    with torch.no_grad():
        for p, v in zip(trainable_params(model), params):
            p.copy_(v.to(p.device))
