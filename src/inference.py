
import cv2
import torch
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
import torchvision.transforms.functional as F


# ── COCO category colors — one fixed color per class ─────────────────────────

def get_color_map():
    """Fixed color per COCO category for consistent visualization."""
    np.random.seed(42)
    colors = {}
    for i in range(91):
        colors[i] = tuple(int(x) for x in np.random.randint(50, 230, 3))
    return colors

COLOR_MAP = get_color_map()


def load_model(checkpoint_path, num_classes, device):
    """Load trained Mask R-CNN from checkpoint."""
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from model import get_model

    model = get_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model = model.to(device)
    print(f"✓ Loaded checkpoint (epoch {checkpoint['epoch']}, "
          f"val_loss {checkpoint['val_losses']['total']:.3f})")
    return model


@torch.no_grad()
def predict(model, image_path, device, score_threshold=0.5):
    """
    Run inference on a single image.

    Args:
        score_threshold : only keep predictions above this confidence
                          0.5 is a good default — lower = more detections
                          but more false positives

    Returns:
        original  : np.array [H, W, 3]  original BGR image
        boxes     : np.array [N, 4]     xyxy boxes above threshold
        labels    : np.array [N]        class ids
        scores    : np.array [N]        confidence scores
        masks     : np.array [N, H, W]  binary masks
    """
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor [3, H, W] normalized to [0, 1]
    tensor = F.to_tensor(image_rgb).to(device)

    # Model expects a list of images
    predictions = model([tensor])[0]

    # Filter by score threshold
    keep    = predictions['scores'] >= score_threshold
    boxes   = predictions['boxes'][keep].cpu().numpy()
    labels  = predictions['labels'][keep].cpu().numpy()
    scores  = predictions['scores'][keep].cpu().numpy()

    # Masks come as [N, 1, H, W] soft probabilities — threshold to binary
    masks = predictions['masks'][keep].cpu().numpy()
    masks = (masks[:, 0] > 0.5).astype(np.uint8)  # [N, H, W] binary

    return image_bgr, boxes, labels, scores, masks


def draw_instances(image_bgr, boxes, labels, scores, masks,
                   id_to_name, alpha=0.45):
    """
    Draws per-instance colored masks, bounding boxes, and labels
    on the image using OpenCV.

    Each instance gets a unique color from the COLOR_MAP.
    Returns a new image — original is not modified.
    """
    output = image_bgr.copy()
    H, W   = output.shape[:2]

    for i, (box, label, score, mask) in enumerate(
            zip(boxes, labels, scores, masks)):

        color    = COLOR_MAP[int(label)]       # BGR
        color_bgr = color[::-1] if len(color) == 3 else color

        # ── Filled mask overlay ───────────────────────────────────────────────
        color_layer = np.zeros_like(output)
        color_layer[mask == 1] = color_bgr
        output = cv2.addWeighted(output, 1.0, color_layer, alpha, 0)

        # ── Mask contour — sharp boundary ────────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color_bgr, 2)

        # ── Bounding box ──────────────────────────────────────────────────────
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), color_bgr, 2)

        # ── Label with confidence score ───────────────────────────────────────
        class_name = id_to_name.get(int(label), str(label))
        text       = f"{class_name} {score:.2f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Background rectangle behind text
        cv2.rectangle(output,
                      (x1, y1 - th - 6),
                      (x1 + tw + 4, y1),
                      color_bgr, -1)

        # White text on colored background
        cv2.putText(output, text,
                    (x1 + 2, y1 - 4),
                    font, font_scale,
                    (255, 255, 255), thickness,
                    cv2.LINE_AA)

    return output


def run_inference_batch(model, img_dir, annot_file, output_dir,
                        device, n_samples=8, score_threshold=0.5):
    """
    Runs inference on n_samples images and saves side-by-side
    visualizations: original | predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO for category names
    coco       = COCO(annot_file)
    id_to_name = {c['id']: c['name']
                  for c in coco.loadCats(coco.getCatIds())}

    # Grab sample images
    img_dir   = Path(img_dir)
    img_paths = sorted(img_dir.glob('*.jpg'))[:n_samples]

    print(f"Running inference on {len(img_paths)} images...")
    saved = 0

    for img_path in img_paths:
        image_bgr, boxes, labels, scores, masks = predict(
            model, img_path, device, score_threshold
        )

        # Skip images with no detections
        if len(boxes) == 0:
            continue

        # Draw predictions
        pred_vis = draw_instances(
            image_bgr, boxes, labels, scores, masks, id_to_name
        )

        # Add instance count label
        H, W = image_bgr.shape[:2]
        info  = f"{len(boxes)} instances detected"
        cv2.putText(pred_vis, info, (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Add column headers
        orig_labeled = image_bgr.copy()
        cv2.putText(orig_labeled, "Original",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pred_vis, "Predictions",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Stack side by side
        combined = np.hstack([orig_labeled, pred_vis])

        save_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(save_path), combined)
        saved += 1

    print(f"✓ Saved {saved} visualizations to {output_dir}")
    return output_dir
