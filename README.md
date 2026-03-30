# Instance Segmentation using Mask R-CNN

Per-instance object detection, classification, and pixel-level mask prediction using Mask R-CNN with a ResNet-50 FPN backbone, trained on the COCO 2017 dataset.

![Inference results](outputs/visualizations/inference_grid.png)

---

## Overview

This project implements an instance segmentation pipeline using Mask R-CNN. Unlike semantic segmentation which assigns a single class label per pixel, instance segmentation detects every individual object separately — each instance gets its own bounding box, class label, confidence score, and pixel-level mask simultaneously.

The notebook `Instance_segmentation_Mask_Rcnn.ipynb` contains the full development walkthrough on Google Colab demonstrating the complete pipeline end to end.

**Results after 10 epochs fine-tuning on 400 COCO images:**

| Metric | Value |
|--------|-------|
| Best val loss | 1.437 |
| Mask loss | 0.415 |
| Classifier loss | 0.266 |

---

## How to replicate

### 1. Clone the repository

    git clone https://github.com/AshishKale1234/Instance_Segmentation_Mask_RCNN.git
    cd Instance_Segmentation_Mask_RCNN

### 2. Set up the environment

    conda create -n maskrcnn python=3.10 -y
    conda activate maskrcnn

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt

### 3. Download COCO 2017 dataset

    mkdir -p data/coco/images data/coco/annotations

    wget http://images.cocodataset.org/zips/val2017.zip -O data/coco/val2017.zip
    unzip data/coco/val2017.zip -d data/coco/images/

    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations.zip
    unzip data/coco/annotations.zip -d data/coco/

### 4. Train

    python run.py --mode train --data_root ./data/coco --epochs 10

All available arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| --mode | train | train or infer |
| --data_root | ./data/coco | Path to COCO data folder |
| --epochs | 10 | Number of training epochs |
| --batch_size | 2 | Images per batch |
| --lr | 0.005 | Initial learning rate |
| --max_images | 500 | Images to use from val set |

Checkpoints saved to outputs/checkpoints/best_model.pth whenever val loss improves.

### 5. Run inference

    python run.py --mode infer --checkpoint outputs/checkpoints/best_model.pth

Results saved to outputs/visualizations/ — each image shows original and predictions side by side with colored per-instance masks, bounding boxes, and confidence labels.

### 6. Running on Google Colab

This project was developed on Google Colab with a free T4 GPU. Open Instance_segmentation_Mask_Rcnn.ipynb directly in Colab for the full step-by-step walkthrough including environment setup, dataset download, training, and inference.

1. Go to colab.research.google.com
2. File -> Upload notebook -> select Instance_segmentation_Mask_Rcnn.ipynb
3. Runtime -> Change runtime type -> T4 GPU
4. Run cells top to bottom

---

## Architecture

Mask R-CNN with ResNet-50 + FPN backbone:

    Input image
        -> ResNet-50 backbone (pretrained on ImageNet via torchvision)
        -> Feature Pyramid Network (FPN) — multi-scale feature maps
        -> Region Proposal Network (RPN) — proposes ~2000 candidate regions
        -> RoI Align — extracts fixed-size features per proposal
        -> Three parallel heads:
            Box head    -> refined bounding box coordinates
            Class head  -> object category (80 COCO classes + background)
            Mask head   -> 28x28 binary mask upsampled to full resolution

Total parameters: ~44.3M | Trainable: ~44.1M

---

## Loss functions

Five losses computed and backpropagated simultaneously:

| Loss | What it optimizes |
|------|-------------------|
| loss_objectness | RPN — is there an object at this anchor? |
| loss_rpn_box_reg | RPN — rough bounding box estimation |
| loss_classifier | Detection head — which of 80 categories? |
| loss_box_reg | Detection head — refined box coordinates |
| loss_mask | Mask head — pixel-level instance mask |

---

## Transfer learning

ResNet-50 backbone loaded with COCO pretrained weights via torchvision. The box predictor and mask predictor heads are replaced with fresh heads for num_classes=91. LR warmup over the first epoch prevents the randomly initialized heads from destabilizing pretrained backbone weights.

---

## Project structure

    Instance_segmentation_Mask_Rcnn.ipynb   <- full Colab walkthrough
    src/
        dataset.py      # COCO Dataset class — loads images and per-instance targets
        model.py        # Mask R-CNN with pretrained ResNet-50 FPN backbone
        train.py        # Training loop with warmup, LR decay, checkpointing
        inference.py    # Inference and OpenCV per-instance visualization
    run.py              # Entry point for training and inference
    requirements.txt
    README.md

Note: data/ and outputs/checkpoints/ are excluded. Download COCO separately and checkpoints are generated after training.

---

## Dependencies

| Package | Version |
|---------|---------|
| torch | >=2.0.0 |
| torchvision | >=0.15.0 |
| opencv-python | >=4.7.0 |
| pycocotools | >=2.0.6 |
| matplotlib | >=3.7.0 |
| numpy | >=1.24.0 |

---

## References

- Mask R-CNN — He et al., 2017 — https://arxiv.org/abs/1703.06870
- COCO Dataset — Lin et al., 2014 — https://cocodataset.org
- torchvision Mask R-CNN — https://pytorch.org/vision/stable/models.html
