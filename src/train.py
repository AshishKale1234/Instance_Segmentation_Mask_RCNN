
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import sys, os
import math


# ── Warmup LR scheduler ──────────────────────────────────────────────────────

class WarmupScheduler:
    """
    Linearly increases LR from warmup_factor * base_lr to base_lr
    over warmup_iters steps.

    Why warmup? The replaced classification and mask heads start with
    random weights. A full learning rate on step 1 would produce huge
    gradients that destabilize the pretrained backbone weights we want
    to preserve. Warmup gives the new heads time to reach reasonable
    weights before full-strength updates kick in.
    """
    def __init__(self, optimizer, warmup_iters, warmup_factor=0.001):
        self.optimizer     = optimizer
        self.warmup_iters  = warmup_iters
        self.warmup_factor = warmup_factor
        self.last_step     = 0

    def step(self):
        self.last_step += 1
        if self.last_step <= self.warmup_iters:
            alpha  = self.last_step / self.warmup_iters
            factor = self.warmup_factor * (1 - alpha) + alpha
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * factor

    def finished(self):
        return self.last_step >= self.warmup_iters


def train_one_epoch(model, loader, optimizer, device,
                    epoch, warmup_scheduler=None, print_freq=20):
    """
    One full pass over the training data.
    Returns dict of average losses for the epoch.
    """
    model.train()

    loss_accum = {
        'loss_classifier'    : 0,
        'loss_box_reg'       : 0,
        'loss_mask'          : 0,
        'loss_objectness'    : 0,
        'loss_rpn_box_reg'   : 0,
        'total'              : 0,
    }
    n_batches = 0

    loop = tqdm(loader, desc=f"  Epoch {epoch}", leave=False)

    for batch_idx, (images, targets) in enumerate(loop):

        # Move to device
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        # ── Forward pass — returns loss dict in train mode ───────────────────
        loss_dict  = model(images, targets)
        total_loss = sum(loss_dict.values())

        # ── Backward pass ────────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping — prevents exploding gradients
        # especially important in early warmup steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Warmup step ──────────────────────────────────────────────────────
        if warmup_scheduler and not warmup_scheduler.finished():
            warmup_scheduler.step()

        # ── Accumulate losses ─────────────────────────────────────────────
        for k, v in loss_dict.items():
            loss_accum[k] += v.item()
        loss_accum['total'] += total_loss.item()
        n_batches += 1

        # Update tqdm
        loop.set_postfix(
            loss=f"{total_loss.item():.3f}",
            cls=f"{loss_dict['loss_classifier'].item():.3f}",
            mask=f"{loss_dict['loss_mask'].item():.3f}",
        )

    # Average over batches
    return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Runs inference on validation data and computes average losses.
    Note: for proper COCO mAP evaluation we would use
    pycocotools — we keep this simple for now and use loss as proxy.
    """
    model.train()   # losses only available in train mode
    loss_accum = {
        'loss_classifier'  : 0,
        'loss_box_reg'     : 0,
        'loss_mask'        : 0,
        'loss_objectness'  : 0,
        'loss_rpn_box_reg' : 0,
        'total'            : 0,
    }
    n_batches = 0

    for images, targets in tqdm(loader, desc="  Eval ", leave=False):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        loss_dict  = model(images, targets)
        total_loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_accum[k] += v.item()
        loss_accum['total'] += total_loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


def train(model, train_loader, val_loader, config):
    """
    Full training loop with:
    - LR warmup for first epoch
    - StepLR decay after warmup
    - Best checkpoint saving based on total val loss
    - Per-epoch logging of all five losses
    """
    device    = config['device']
    epochs    = config['epochs']
    ckpt_path = config['checkpoint_path']
    lr        = config['lr']

    # ── Optimizer ────────────────────────────────────────────────────────────
    # SGD with momentum is standard for detection models
    # Adam works too but SGD generalizes better here
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Store initial LR for warmup scheduler
    for pg in optimizer.param_groups:
        pg['initial_lr'] = lr

    # ── Warmup — runs for first epoch's worth of steps ───────────────────────
    warmup_iters      = len(train_loader)
    warmup_scheduler  = WarmupScheduler(optimizer, warmup_iters)

    # ── LR decay — halve LR every step_size epochs after warmup ─────────────
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    best_loss = float('inf')
    history   = {
        'train_total': [], 'val_total': [],
        'train_mask' : [], 'val_mask' : [],
        'train_cls'  : [], 'val_cls'  : [],
    }

    print(f"\nTraining on {device} | {epochs} epochs | "
          f"lr={lr} | {len(train_loader)} batches/epoch")
    print(f"{'Epoch':>6}  {'T-Total':>8}  {'V-Total':>8}  "
          f"{'T-Mask':>8}  {'V-Mask':>8}  {'T-Cls':>7}  LR")
    print("─" * 66)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        t_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            warmup_scheduler = warmup_scheduler if epoch == 1 else None,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        v_losses = evaluate(model, val_loader, device)

        # ── LR step (after warmup epoch) ──────────────────────────────────────
        if epoch > 1:
            lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Log ───────────────────────────────────────────────────────────────
        history['train_total'].append(t_losses['total'])
        history['val_total'].append(v_losses['total'])
        history['train_mask'].append(t_losses['loss_mask'])
        history['val_mask'].append(v_losses['loss_mask'])
        history['train_cls'].append(t_losses['loss_classifier'])
        history['val_cls'].append(v_losses['loss_classifier'])

        print(f"{epoch:>6}  "
              f"{t_losses['total']:>8.3f}  {v_losses['total']:>8.3f}  "
              f"{t_losses['loss_mask']:>8.3f}  {v_losses['loss_mask']:>8.3f}  "
              f"{t_losses['loss_classifier']:>7.3f}  {current_lr:.1e}")

        # ── Save best checkpoint ──────────────────────────────────────────────
        if v_losses['total'] < best_loss:
            best_loss = v_losses['total']
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_losses' : v_losses,
            }, ckpt_path)
            print(f"         ✓ checkpoint saved  "
                  f"(val_loss={best_loss:.3f})")

    print(f"\nBest val loss: {best_loss:.3f}")
    return history
