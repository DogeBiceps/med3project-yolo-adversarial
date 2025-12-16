#!/usr/bin/env python3
"""
Simple evaluation script for one adversarial patch.

Computes:
 - ASR (Attack Success Rate) w.r.t. ground-truth YOLO labels
 - PSNR between clean and patched images
 - Confidence distributions (clean vs patched) for the target class

Usage example (from repo root):

  python test_patch_metrics.py \
    --cfg adv_patch_gen/configs/target_person_coco.json \
    --patchfile runs/patch_person/20251112-120409_person_yolov5s/patches/e_127.png \
    --imgdir data/person_val/images \
    --labeldir data/person_val/labels \
    --conf-thresh 0.4 \
    --iou-thresh 0.5 \
    --max-images 200

"""

import argparse
import json
import math
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from adv_patch_gen.utils.common import pad_to_square
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device


# ---------- Helpers ----------

def load_cfg(path: str) -> edict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return edict(d)


def load_yolo_labels(label_path: str, img_w: int, img_h: int, cls_filter: int = None):
    """
    Load YOLO-format labels: 'cls x_center y_center w h' (normalized).
    Returns:
        - labels_norm: tensor (N,5) -> [cls, x, y, w, h] normalized (0..1)
        - boxes_xyxy: tensor (N,4) -> [x1,y1,x2,y2] in pixels
    """
    if not osp.exists(label_path):
        return (
            torch.zeros((0, 5), dtype=torch.float32),
            torch.zeros((0, 4), dtype=torch.float32),
        )

    lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            if cls_filter is not None and c != cls_filter:
                continue
            lines.append([c, x, y, w, h])

    if not lines:
        return (
            torch.zeros((0, 5), dtype=torch.float32),
            torch.zeros((0, 4), dtype=torch.float32),
        )

    labels = torch.tensor(lines, dtype=torch.float32)  # (N,5)

    # Convert to pixel xyxy
    x_c = labels[:, 1] * img_w
    y_c = labels[:, 2] * img_h
    bw = labels[:, 3] * img_w
    bh = labels[:, 4] * img_h
    x1 = x_c - bw / 2.0
    y1 = y_c - bh / 2.0
    x2 = x_c + bw / 2.0
    y2 = y_c + bh / 2.0
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    return labels, boxes_xyxy


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU between two sets of boxes (xyxy).
    boxes1: (N,4), boxes2: (M,4)
    returns: (N,M)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    # intersection
    tl = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    br = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    # union
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6

    return inter / union


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    PSNR between two images in [0,1].
    img1, img2: (3,H,W) tensors
    """
    mse = F.mse_loss(img1, img2, reduction="mean").item()
    if mse <= 1e-10:
        return float("inf")
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


# ---------- Main evaluation ----------

def main():
    parser = argparse.ArgumentParser("Simple patch evaluation (ASR, PSNR, conf dist)")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config JSON (same as training).")
    parser.add_argument("--patchfile", type=str, required=True, help="Path to patch image (e.g. e_127.png).")
    parser.add_argument("--imgdir", type=str, required=True, help="Directory with val images.")
    parser.add_argument("--labeldir", type=str, required=True, help="Directory with YOLO-format labels.")
    parser.add_argument("--conf-thresh", type=float, default=0.4, help="Confidence threshold for detection.")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching GT to detections.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all).")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    dev = select_device(cfg.device)

    # Load detector
    model = DetectMultiBackend(cfg.weights_file, device=dev, dnn=False, data=None, fp16=False)
    model.model.eval().to(dev)

    # Patch transformer & applier (same params as training)
    patch_transformer = PatchTransformer(
        cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, dev
    ).to(dev)
    patch_applier = PatchApplier(cfg.patch_alpha).to(dev)

    # Load patch
    patch_img = Image.open(args.patchfile).convert(cfg.patch_img_mode)
    patch_img = T.Resize(cfg.patch_size)(patch_img)
    adv_patch_cpu = T.ToTensor()(patch_img)  # (C,H,W) in [0,1]
    adv_patch = adv_patch_cpu.to(dev)

    m_h, m_w = cfg.model_in_sz
    target_cls = cfg.objective_class_id if hasattr(cfg, "objective_class_id") else 0

    # Transforms
    resize_to_model = T.Resize((m_h, m_w))
    to_tensor = T.ToTensor()

    img_paths = sorted(
        [
            osp.join(args.imgdir, f)
            for f in os.listdir(args.imgdir)
            if osp.splitext(f)[-1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )

    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    print(f"Found {len(img_paths)} images for evaluation.")

    total_gt_detected_clean = 0
    total_gt_success_attack = 0
    psnr_values = []

    clean_confs = []
    patched_confs = []

    for img_path in tqdm(img_paths, desc="Evaluating"):
        img_name = osp.splitext(osp.basename(img_path))[0]

        # --- Load & pad image ---
        img_pil = Image.open(img_path).convert("RGB")
        padded = pad_to_square(img_pil)
        padded = resize_to_model(padded)
        img_tensor = to_tensor(padded).unsqueeze(0).to(dev)  # (1,3,H,W)

        # --- Load YOLO GT labels for this image ---
        label_path = osp.join(args.labeldir, img_name + ".txt")
        gt_labels_norm, gt_boxes_xyxy = load_yolo_labels(label_path, m_w, m_h, cls_filter=target_cls)

        if gt_boxes_xyxy.numel() == 0:
            # No GT person boxes; skip from ASR stats but still can count PSNR/conf if you want,
            # but typically you care about images where the target exists.
            continue

        # ------------- CLEAN DETECTIONS -------------
        with torch.no_grad():
            pred = model(img_tensor)
            pred_boxes = non_max_suppression(pred, args.conf_thresh, 0.4)[0]  # (N,6): x1,y1,x2,y2,conf,cls

        if pred_boxes is None or pred_boxes.numel() == 0:
            pred_boxes = torch.zeros((0, 6), device=dev)

        # Filter by target class
        cls_mask = (pred_boxes[:, 5].int() == int(target_cls)) if pred_boxes.numel() else torch.zeros(0, dtype=torch.bool, device=dev)
        pred_tgt_clean = pred_boxes[cls_mask]

        # Confidences distribution (clean)
        if pred_tgt_clean.numel():
            clean_confs.extend(pred_tgt_clean[:, 4].detach().cpu().tolist())

        # --- Match GT -> clean detections (handle "no detections" case safely) ---
        if pred_tgt_clean.numel() == 0:
            # no person detections at all
            detected_clean = torch.zeros(gt_boxes_xyxy.shape[0], dtype=torch.bool, device=dev)
        else:
            ious_clean = box_iou_xyxy(gt_boxes_xyxy.to(dev), pred_tgt_clean[:, :4])  # (N_gt, N_det)
            max_iou, best_idx = ious_clean.max(dim=1)  # best detection per GT
            best_confs = pred_tgt_clean[best_idx, 4]   # confidences of those detections
            detected_clean = (max_iou >= args.iou_thresh) & (best_confs >= args.conf_thresh)


        # We will only count GT boxes that are actually detected in the clean image
        gt_indices_of_interest = torch.nonzero(detected_clean).flatten()
        n_clean = int(gt_indices_of_interest.numel())
        if n_clean == 0:
            # This image has GT but YOLOv5 doesn't detect them at this conf; skip for ASR
            continue

        total_gt_detected_clean += n_clean

        # Build lab_batch for patch placement (like training): (1,N,5) [cls,x,y,w,h] normalized
        labels_for_patch = gt_labels_norm.to(dev).unsqueeze(0)  # (1,N,5)

        adv_batch_t = patch_transformer(
            adv_patch,
            labels_for_patch,
            cfg.model_in_sz,
            use_mul_add_gau=cfg.use_mul_add_gau,
            do_transforms=cfg.transform_patches,
            do_rotate=cfg.rotate_patches,
            rand_loc=cfg.random_patch_loc,
        )

        patched_tensor = patch_applier(img_tensor, adv_batch_t)
        # PSNR (per-image)
        psnr_val = psnr(img_tensor.squeeze(0).cpu(), patched_tensor.squeeze(0).detach().cpu())
        psnr_values.append(psnr_val)

        # ------------- PATCHED DETECTIONS -------------
        with torch.no_grad():
            pred_p = model(patched_tensor)
            pred_boxes_p = non_max_suppression(pred_p, args.conf_thresh, 0.4)[0]

        if pred_boxes_p is None or pred_boxes_p.numel() == 0:
            pred_boxes_p = torch.zeros((0, 6), device=dev)

        cls_mask_p = (pred_boxes_p[:, 5].int() == int(target_cls)) if pred_boxes_p.numel() else torch.zeros(0, dtype=torch.bool, device=dev)
        pred_tgt_patched = pred_boxes_p[cls_mask_p]

        if pred_tgt_patched.numel():
            patched_confs.extend(pred_tgt_patched[:, 4].detach().cpu().tolist())

        if pred_tgt_patched.numel() == 0:
            detected_patched = torch.zeros(gt_boxes_xyxy.shape[0], dtype=torch.bool, device=dev)
        else:
            ious_patched = box_iou_xyxy(gt_boxes_xyxy.to(dev), pred_tgt_patched[:, :4])  # (N_gt, N_det)
            max_iou_p, best_idx_p = ious_patched.max(dim=1)
            best_confs_p = pred_tgt_patched[best_idx_p, 4]
            detected_patched = (max_iou_p >= args.iou_thresh) & (best_confs_p >= args.conf_thresh)


        # Count successes only among GT that were detected clean
        # Success = was detected clean, but NOT detected patched
        success_mask = detected_clean & (~detected_patched)
        total_gt_success_attack += int(success_mask[gt_indices_of_interest].sum().item())

    # ---------- Final stats ----------
    if total_gt_detected_clean == 0:
        print("No GT boxes were detected in clean images at the given thresholds. ASR undefined.")
    else:
        asr = total_gt_success_attack / total_gt_detected_clean
        print(f"\n=== Attack Success Rate (ASR) ===")
        print(f"Detected GT (clean): {total_gt_detected_clean}")
        print(f"Successful attacks : {total_gt_success_attack}")
        print(f"ASR                 : {asr:.3f}")

    if psnr_values:
        print("\n=== PSNR ===")
        print(f"Mean PSNR over images: {np.mean(psnr_values):.2f} dB")
        print(f"Min  PSNR            : {np.min(psnr_values):.2f} dB")
        print(f"Max  PSNR            : {np.max(psnr_values):.2f} dB")

    def summarize_confs(name, confs):
        if not confs:
            print(f"{name}: no detections.")
            return
        arr = np.array(confs)
        print(f"{name}: n={arr.size}, mean={arr.mean():.3f}, std={arr.std():.3f}, "
              f"25%={np.percentile(arr,25):.3f}, 50%={np.percentile(arr,50):.3f}, 75%={np.percentile(arr,75):.3f}")

    print("\n=== Confidence distributions (target class) ===")
    summarize_confs("Clean detections  ", clean_confs)
    summarize_confs("Patched detections", patched_confs)


if __name__ == "__main__":
    main()
