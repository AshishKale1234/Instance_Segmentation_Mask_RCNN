"""
Instance Segmentation using Mask R-CNN
Runs training and inference on COCO dataset.

Usage:
    # Train
    python run.py --mode train --data_root ./data/coco --epochs 10

    # Inference
    python run.py --mode infer --data_root ./data/coco --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import torch
import sys
sys.path.append('./src')

from dataset import COCOInstanceDataset, collate_fn
from model   import get_model
from train   import train
from inference import load_model, run_inference_batch
from torch.utils.data import DataLoader, Subset


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    IMG_DIR    = f'{args.data_root}/images/val2017'
    ANNOT_FILE = f'{args.data_root}/annotations/instances_val2017.json'

    if args.mode == 'train':
        full_dataset = COCOInstanceDataset(
            img_dir    = IMG_DIR,
            annot_file = ANNOT_FILE,
            max_images = args.max_images,
        )

        split       = int(len(full_dataset) * 0.8)
        train_ds    = Subset(full_dataset, range(0, split))
        val_ds      = Subset(full_dataset, range(split, len(full_dataset)))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=2,
                                  collate_fn=collate_fn, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=2,
                                  collate_fn=collate_fn, pin_memory=True)

        model  = get_model(num_classes=91, pretrained=True)
        model  = model.to(device)

        config = {
            'device'         : device,
            'epochs'         : args.epochs,
            'lr'             : args.lr,
            'checkpoint_path': 'outputs/checkpoints/best_model.pth',
        }

        train(model, train_loader, val_loader, config)

    elif args.mode == 'infer':
        model = load_model(args.checkpoint, num_classes=91, device=device)
        run_inference_batch(
            model           = model,
            img_dir         = IMG_DIR,
            annot_file      = ANNOT_FILE,
            output_dir      = 'outputs/visualizations',
            device          = device,
            n_samples       = args.n_samples,
            score_threshold = args.threshold,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       default='train',
                        choices=['train', 'infer'])
    parser.add_argument('--data_root',  default='./data/coco')
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=2)
    parser.add_argument('--lr',         type=float, default=0.005)
    parser.add_argument('--max_images', type=int,   default=500)
    parser.add_argument('--checkpoint', default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--n_samples',  type=int,   default=8)
    parser.add_argument('--threshold',  type=float, default=0.5)
    args = parser.parse_args()
    main(args)
