
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


class COCOInstanceDataset(Dataset):
    """
    Loads COCO images and returns per-instance targets in the exact
    format torchvision Mask R-CNN expects.

    Each item returns:
        image  : FloatTensor [3, H, W]  normalized to [0, 1]
        target : dict with keys —
            boxes    [N, 4]   xyxy format
            labels   [N]      class ids (1-indexed, 0 = background)
            masks    [N,H,W]  binary per-instance masks
            image_id [1]
            area     [N]
            iscrowd  [N]
    """

    def __init__(self, img_dir, annot_file, transforms=None,
                 max_images=None, min_area=100):
        self.img_dir    = Path(img_dir)
        self.transforms = transforms
        self.min_area   = min_area   # skip tiny annotations

        print(f"Loading COCO annotations from {annot_file}...")
        self.coco = COCO(annot_file)

        # Only keep images that have at least one annotation
        all_img_ids = self.coco.getImgIds()
        self.img_ids = [
            img_id for img_id in all_img_ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
        ]

        if max_images:
            self.img_ids = self.img_ids[:max_images]

        print(f"Dataset ready: {len(self.img_ids)} images with annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # ── Load image ───────────────────────────────────────────────────────
        img_path = self.img_dir / img_info['file_name']
        image    = cv2.imread(str(img_path))
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W     = image.shape[:2]

        # ── Load annotations ─────────────────────────────────────────────────
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        # Filter out tiny annotations (often noise)
        anns = [a for a in anns if a['area'] >= self.min_area]

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            # COCO bbox is [x, y, w, h] — convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

            # Convert COCO polygon/RLE mask to binary array [H, W]
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # ── Handle images where all annotations were filtered out ────────────
        if len(boxes) == 0:
            boxes   = torch.zeros((0, 4),  dtype=torch.float32)
            labels  = torch.zeros((0,),    dtype=torch.int64)
            masks   = torch.zeros((0, H, W), dtype=torch.uint8)
            areas   = torch.zeros((0,),    dtype=torch.float32)
            iscrowd = torch.zeros((0,),    dtype=torch.int64)
        else:
            boxes   = torch.tensor(boxes,   dtype=torch.float32)
            labels  = torch.tensor(labels,  dtype=torch.int64)
            masks   = torch.tensor(np.stack(masks), dtype=torch.uint8)
            areas   = torch.tensor(areas,   dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes'   : boxes,
            'labels'  : labels,
            'masks'   : masks,
            'image_id': torch.tensor([img_id], dtype=torch.int64),
            'area'    : areas,
            'iscrowd' : iscrowd,
        }

        # ── Convert image to tensor ──────────────────────────────────────────
        # ToTensor normalizes uint8 [0,255] → float32 [0,1]
        image = F.to_tensor(image)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


# ── Collate function ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Default collate stacks tensors into a single batch tensor.
    That breaks here because each image has a different number of
    objects (N varies per image). We return a plain list instead —
    torchvision detection models expect exactly this format.
    """
    return tuple(zip(*batch))


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(img_dir, annot_file, batch_size=2,
                    num_workers=2, max_images=None):
    """
    Returns (dataset, dataloader).
    We use a single val split here since COCO val is our full dataset.
    Training uses a subset, evaluation uses the rest.
    """
    dataset = COCOInstanceDataset(
        img_dir    = img_dir,
        annot_file = annot_file,
        max_images = max_images,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,   # critical — must use custom collate
        pin_memory  = True,
    )

    return dataset, loader
