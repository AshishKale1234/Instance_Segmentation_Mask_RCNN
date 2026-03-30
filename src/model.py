
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes, pretrained=True, trainable_backbone_layers=3):
    """
    Loads Mask R-CNN with ResNet-50 + FPN backbone pretrained on COCO.

    Args:
        num_classes              : number of object classes + 1 (background)
                                   COCO has 80 classes → pass 81
        pretrained               : load COCO pretrained weights
        trainable_backbone_layers: how many ResNet layers to unfreeze
                                   0 = freeze entire backbone (fastest)
                                   5 = unfreeze entire backbone (slowest)
                                   3 = good default balance

    Returns:
        model : Mask R-CNN ready for training or inference
    """

    # ── Load pretrained backbone ─────────────────────────────────────────────
    # weights='DEFAULT' loads the best available COCO pretrained weights
    model = maskrcnn_resnet50_fpn(
        weights                  = 'DEFAULT' if pretrained else None,
        trainable_backbone_layers = trainable_backbone_layers,
    )

    # ── Replace the box predictor head ───────────────────────────────────────
    # The pretrained head outputs 91 classes (COCO uses 1-indexed 1..90
    # plus background=0). We replace it to match our num_classes.
    # Even if num_classes=81 (full COCO) we still replace to be explicit.
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box, num_classes
    )

    # ── Replace the mask predictor head ──────────────────────────────────────
    # Same reason — replace to match num_classes exactly
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_model_info(model):
    """Print a clean summary of model parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {frozen:,}")

    print(f"\nKey components:")
    components = {
        'Backbone (ResNet50+FPN)' : model.backbone,
        'RPN'                     : model.rpn,
        'RoI heads'               : model.roi_heads,
    }
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<25}: {n:>10,} total  {t:>10,} trainable")
