"""
Training code for Adversarial patch training.

Usage:
    python train_patch.py --cfg config_json_file
"""

import glob
import json
import os
import os.path as osp
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image
from tensorboard import program
from torch import autograd, optim
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from adv_patch_gen.utils.common import IMG_EXTNS, is_port_in_use, pad_to_square
from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.loss import NPSLoss, SaliencyLoss, TotalVariationLoss
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from models.common import DetectMultiBackend
from test_patch import PatchTester
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device

SEED = None
if SEED is not None:
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)

torch.backends.cudnn.benchmark = False


class PatchTrainer:
    """Module for training on dataset to generate adv patches."""

    def _ensure_labels_for_batch(self, lab_batch: torch.Tensor) -> torch.Tensor:
        B = lab_batch.shape[0]
        device = lab_batch.device
        cls_id = 0 if self.cfg.objective_class_id is None else self.cfg.objective_class_id

    # width/height of the synthetic box ~ target_size_frac (clamped)
    # PatchTransformer expects normalized xywh in [0,1]
        frac = self.patch_transformer.t_size_frac
        if isinstance(frac, (list, tuple)):
        # if provided as [wfrac, hfrac]; fall back to scalar if needed
            wfrac = float(frac[0])
            hfrac = float(frac[1]) if len(frac) > 1 else float(frac[0])
        else:
            wfrac = hfrac = float(frac)
        wfrac = max(0.05, min(0.8, wfrac))
        hfrac = max(0.05, min(0.8, hfrac))

        new_list = []
        for b in range(B):
            lb = lab_batch[b]  # (N, 5)
        # valid if any row is not all zeros
            is_all_zero = (lb.abs().sum(dim=1) == 0).all()
            if lb.numel() == 0 or is_all_zero:
                fake = torch.tensor([[cls_id, 0.5, 0.5, wfrac, hfrac]], dtype=lb.dtype, device=device)
            # keep the shape (N,5). If original N>=1, replace first row and keep rest zeros.
                if lb.numel() == 0:
                    new_list.append(fake)
                else:
                    lb = lb.clone()
                    lb[0] = fake[0]
                    new_list.append(lb)
            else:
                new_list.append(lb)
        return torch.stack(new_list, dim=0)

    def __init__(self, cfg: edict):
        self.cfg = cfg
        self.dev = select_device(cfg.device)

        self.detector = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.detector.model.eval()

        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.dev
        ).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)

                # --- OPTIONAL: load a spatial mask in patch space (white=keep, black=cut) ---
        self.mask = None
        if getattr(cfg, "use_mask", False) and getattr(cfg, "mask_src", None):
            m = Image.open(cfg.mask_src).convert("L")  # single channel
            m = T.Resize(cfg.patch_size)(m)
            m = T.ToTensor()(m).clamp(0, 1)            # (1,H,W), range [0,1]
            if getattr(cfg, "mask_invert", False):
                m = 1.0 - m
            # expand to 3 channels to match RGB patch
            self.mask = m.expand(3, -1, -1).to(self.dev)


        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.dev)
        self.tv_loss  = TotalVariationLoss().to(self.dev)

        for p in self.detector.model.parameters():
            p.requires_grad = False

        cfg.log_dir = osp.join(cfg.log_dir, f'{time.strftime("%Y%m%d-%H%M%S")}_{cfg.patch_name}')
        self.writer = self.init_tensorboard(cfg.log_dir, cfg.tensorboard_port)
        for k, v in cfg.items():
            self.writer.add_text(k, str(v))

        transforms = None
        if cfg.augment_image:
            transforms = T.Compose([
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                T.ColorJitter(brightness=0.2, hue=0.04, contrast=0.1),
                T.RandomAdjustSharpness(sharpness_factor=2),
            ])

        self.train_loader = torch.utils.data.DataLoader(
            YOLODataset(
                image_dir=cfg.image_dir,
                label_dir=cfg.label_dir,
                max_labels=cfg.max_labels,
                model_in_sz=cfg.model_in_sz,
                use_even_odd_images=cfg.use_even_odd_images,
                transform=transforms,
                filter_class_ids=cfg.objective_class_id,
                min_pixel_area=cfg.min_pixel_area,
                shuffle=True,
            ),
            batch_size=cfg.batch_size, shuffle=True, num_workers=4,
            pin_memory=True if self.dev.type == "cuda" else False,
        )
        self.epoch_length = len(self.train_loader)



    def init_tensorboard(self, log_dir: str = None, port: int = 6006, run_tb=True):
        if run_tb:
            while is_port_in_use(port) and port < 65535:
                port += 1
                print(f"Port {port - 1} in use, switching to {port}")
            tboard = program.TensorBoard()
            tboard.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
            url = tboard.launch()
            print(f"Tensorboard started on {url}")
        return SummaryWriter(log_dir) if log_dir else SummaryWriter()

    def generate_patch(self, patch_type: str, pil_img_mode: str = "RGB") -> torch.Tensor:
        p_c = 1 if pil_img_mode in {"L"} else 3
        p_w, p_h = self.cfg.patch_size
        if patch_type == "gray":
            adv_patch_cpu = torch.full((p_c, p_h, p_w), 0.5)
        elif patch_type == "random":
            adv_patch_cpu = torch.rand((p_c, p_h, p_w))
        return adv_patch_cpu

    def read_image(self, path, pil_img_mode: str = "RGB") -> torch.Tensor:
        patch_img = Image.open(path).convert(pil_img_mode)
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        return adv_patch_cpu

    def _yolo_confidence(self, out: torch.Tensor) -> torch.Tensor:
        obj = out[..., 4].sigmoid()
        if self.cfg.objective_class_id is not None:
            cls = out[..., 5 + self.cfg.objective_class_id].sigmoid()
        else:
            cls = out[..., 5:].sigmoid().amax(dim=-1)
        conf = obj * cls
        return conf.amax(dim=1)

    def train(self) -> None:
        patch_dir = osp.join(self.cfg.log_dir, "patches")
        os.makedirs(patch_dir, exist_ok=True)
        with open(osp.join(self.cfg.log_dir, "cfg.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=4)

        lt = self.cfg.loss_target
        if lt == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif lt == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif lt in {"obj * cls", "obj*cls"}:
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(f"Loss target {lt} not implemented")

        if self.cfg.patch_src == "gray":
            adv_patch_cpu = self.generate_patch("gray", self.cfg.patch_img_mode)
        elif self.cfg.patch_src == "random":
            adv_patch_cpu = self.generate_patch("random", self.cfg.patch_img_mode)
        else:
            adv_patch_cpu = self.read_image(self.cfg.patch_src, self.cfg.patch_img_mode)

        adv_patch_cpu.requires_grad = True

        optimizer = optim.Adam([adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

        prev_patch = adv_patch_cpu.detach().clone()
        start_time = time.time()

        for epoch in range(1, self.cfg.n_epochs + 1):
            out_patch_path = osp.join(patch_dir, f"e_{epoch}.png")
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.dev)
            zero_tensor = torch.tensor([0], device=self.dev)

            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(self.train_loader), desc=f"Running train epoch {epoch}", total=self.epoch_length
            ):
                img_batch = img_batch.to(self.dev, non_blocking=True)
                lab_batch = lab_batch.to(self.dev, non_blocking=True)
                adv_patch = adv_patch_cpu.to(self.dev, non_blocking=True)
                lab_batch = self._ensure_labels_for_batch(lab_batch)

                if self.mask is not None:
                    adv_patch_in = adv_patch * self.mask
                else:
                    adv_patch_in = adv_patch

                adv_batch_t = self.patch_transformer(
                    adv_patch_in, lab_batch, self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc,
                )
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))

                with autocast() if self.cfg.use_amp else nullcontext():
                    det_out_raw = self.detector.model(p_img_batch)
                    if isinstance(det_out_raw, (list, tuple)):
                         det_out_raw = det_out_raw[0]
                    det_out = det_out_raw
                    max_prob = self._yolo_confidence(det_out)
                    sal = self.sal_loss(adv_patch) if self.cfg.sal_mult != 0 else zero_tensor
                    nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else zero_tensor
                    tv  = self.tv_loss(adv_patch)  if self.cfg.tv_mult  != 0 else zero_tensor

                det_loss = torch.mean(max_prob)
                sal_loss = sal * self.cfg.sal_mult
                nps_loss = nps * self.cfg.nps_mult
                tv_loss  = torch.max(tv * self.cfg.tv_mult, min_tv_loss)

                loss = det_loss + sal_loss + nps_loss + tv_loss
                ep_loss += loss
                loss.backward()

                if i_batch == 0 and adv_patch_cpu.grad is not None:
                    gmean = adv_patch_cpu.grad.abs().mean().item()
                    print(f"[epoch {epoch}] grad mean on patch = {gmean:.6e}")

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pl, ph = self.cfg.patch_pixel_range
                adv_patch_cpu.data.clamp_(pl, ph)

            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            with torch.no_grad():
                delta = (adv_patch_cpu - prev_patch).abs().mean().item()
                print(f"[epoch {epoch}] mean|Δpatch| = {delta:.6e}")
                prev_patch = adv_patch_cpu.detach().clone()

            img = T.ToPILImage(self.cfg.patch_img_mode)(adv_patch_cpu.clamp(0, 1))
            img.save(out_patch_path)

        print(f"Total training time {time.time() - start_time:.2f}s")


def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)
    trainer = PatchTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
