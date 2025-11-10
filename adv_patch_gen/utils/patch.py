"""Modules for creating adversarial object patch."""
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_patch_gen.utils.median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    """
    Transforms a patch per target box (brightness/contrast/noise, resize, rotate, translate),
    pads to image size, and returns a 5D tensor (B, K, C, H, W) with zeros where no patch.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        x_off_loc: Tuple[float, float] = (-0.25, 0.25),
        y_off_loc: Tuple[float, float] = (-0.25, 0.25),
        dev: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        # normalize ranges to (low, high)
        self.t_size_frac = (t_size_frac, t_size_frac) if isinstance(t_size_frac, float) else tuple(t_size_frac)
        self.m_gau_mean = (mul_gau_mean, mul_gau_mean) if isinstance(mul_gau_mean, float) else tuple(mul_gau_mean)
        self.m_gau_std  = (mul_gau_std,  mul_gau_std)  if isinstance(mul_gau_std,  float) else tuple(mul_gau_std)
        assert len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2, "Each range must have 2 values"

        self.x_off_loc = tuple(x_off_loc)
        self.y_off_loc = tuple(y_off_loc)
        self.dev = dev

        # photometric ranges
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10

        # geometric ranges (radians)
        self.minangle = -20 / 180 * math.pi
        self.maxangle =  20 / 180 * math.pi

        self.medianpooler = MedianPool2d(kernel_size=7, same=True)

    def forward(
        self,
        adv_patch: torch.Tensor,          # (C, Hp, Wp) in [0,1]
        lab_batch: torch.Tensor,          # (B, K, 5)  [cls, x, y, w, h] normalized
        model_in_sz,                      # (H, W)
        use_mul_add_gau=True,
        do_transforms=True,
        do_rotate=True,
        rand_loc=True,
    ) -> torch.Tensor:
        B, K = lab_batch.shape[:2]
        C, Hp, Wp = adv_patch.shape
        H, W = model_in_sz

        # --- photometric jitter on the base patch (per-sample later we add more) ---
        P = adv_patch
        if use_mul_add_gau:
            # draw scalars safely on device
            m_mean = float(np.random.uniform(*self.m_gau_mean))
            m_std  = float(np.random.uniform(*self.m_gau_std))
            mul_gau = torch.normal(mean=m_mean, std=m_std, size=(C, Hp, Wp), device=self.dev)
            add_gau = torch.normal(mean=0.0, std=0.001, size=(C, Hp, Wp), device=self.dev)
            P = P * mul_gau + add_gau

        # denoise a bit
        P = self.medianpooler(P.unsqueeze(0)).squeeze(0)
        P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # pad amount so patch can be placed anywhere inside (H, W)
        pad_w = max((W - P.size(-1)) / 2.0, 0.0)
        pad_h = max((H - P.size(-2)) / 2.0, 0.0)

        # expand to (B, K, C, Hp, Wp)
        P = P.unsqueeze(0).unsqueeze(0).expand(B, K, C, P.size(-2), P.size(-1)).contiguous()

        # per-box photometric transforms
        if do_transforms:
            # contrast in [0.8, 1.2], brightness in [-0.1, 0.1], noise in [-1,1]*factor
            contrast = torch.empty((B, K, 1, 1, 1), device=self.dev).uniform_(self.min_contrast, self.max_contrast)
            brightness = torch.empty((B, K, 1, 1, 1), device=self.dev).uniform_(self.min_brightness, self.max_brightness)
            noise = torch.empty((B, K, C, P.size(-2), P.size(-1)), device=self.dev).uniform_(-1.0, 1.0) * self.noise_factor
            P = P * contrast + brightness + noise
            P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # mask = ones (we’ll zero out of bounds via grid_sample anyway)
        M = torch.ones_like(P, device=self.dev)

        # pad P and M to image dims
        pad = (int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h))  # left, right, top, bottom
        padder = nn.ConstantPad2d(pad, 0.0)
        P = padder(P.view(B*K, C, P.size(-2), P.size(-1)))
        M = padder(M.view(B*K, C, M.size(-2), M.size(-1)))

        # target geometry
        # lab_batch: [cls, x, y, w, h] normalized
        xywh = lab_batch[..., 1:5].to(self.dev)
        x = xywh[..., 0].reshape(B*K)
        y = xywh[..., 1].reshape(B*K)
        w_n = xywh[..., 2].reshape(B*K)
        h_n = xywh[..., 3].reshape(B*K)

        # random offset
        if rand_loc:
            off_x = w_n * torch.empty_like(x, device=self.dev).uniform_(*self.x_off_loc)
            off_y = h_n * torch.empty_like(y, device=self.dev).uniform_(*self.y_off_loc)
            x = x + off_x
            y = y + off_y

        # choose target scale as a fraction of box size (robustly > 0)
        tmin, tmax = self.t_size_frac
        tsize = float(np.random.uniform(tmin, tmax))
        tsize = max(min(tsize, 0.99), 0.05)

        # convert normalized box wh to pixels
        w_px = w_n * W
        h_px = h_n * W  # NOTE: original code used m_w for all; keep consistent for now
        target_size = torch.sqrt((w_px * tsize) ** 2 + (h_px * tsize) ** 2)

        current_patch_size = P.size(-1)  # after padding above
        scale = (target_size / max(current_patch_size, 1.0)).reshape(B*K)
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-3, max=100.0)

        # rotation
        if do_rotate:
            angle = torch.empty(B*K, device=self.dev).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.zeros(B*K, device=self.dev)

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # translation to grid coords [-1,1]
        tx = (-x + 0.5) * 2.0
        ty = (-y + 0.5) * 2.0

        # build affine theta (B*K, 2, 3) for grid_sample
        theta = torch.zeros((B*K, 2, 3), device=self.dev, dtype=P.dtype)
        inv_s = 1.0 / scale
        theta[:, 0, 0] =  cos * inv_s
        theta[:, 0, 1] =  sin * inv_s
        theta[:, 0, 2] =  tx * cos * inv_s + ty * sin * inv_s
        theta[:, 1, 0] = -sin * inv_s
        theta[:, 1, 1] =  cos * inv_s
        theta[:, 1, 2] = -tx * sin * inv_s + ty * cos * inv_s

        # sanitize theta
        theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)

        # warp into (H, W)
        P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        M = torch.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        grid = F.affine_grid(theta, size=(B*K, P.size(1), H, W), align_corners=False)
        P_w = F.grid_sample(P, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        M_w = F.grid_sample(M, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        P_w = torch.nan_to_num(P_w, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        M_w = torch.nan_to_num(M_w, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        P_w = P_w.view(B, K, C, H, W)
        M_w = M_w.view(B, K, C, H, W)

        # return patch overlays multiplied by their masks
        return P_w * M_w


class PatchApplier(nn.Module):
    """
    Vectorized patch applier supporting:
      - adv_batch: (B, 3, H, W)  single composite overlay, OR
      - adv_batch: (B, K, 3, H, W) multiple overlays per image

    Sequential alpha compositing (order = K dimension):
      out0 = I
      out_{k+1} = out_k * (1 - alpha*M_k) + alpha*M_k * P_k
    """

    def __init__(self, patch_alpha: float = 1.0):
        super().__init__()
        self.patch_alpha = float(patch_alpha)

    @staticmethod
    def _mask_from_patch(P: torch.Tensor) -> torch.Tensor:
        # True where any channel is non-zero
        return (P.abs().sum(dim=1, keepdim=True) > 1e-6).to(P.dtype)

    def forward(self, images: torch.Tensor, adv_batch: torch.Tensor) -> torch.Tensor:
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if adv_batch.dim() == 4:
            # (B, 3, H, W)
            P = torch.nan_to_num(adv_batch, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            M = self._mask_from_patch(P)  # (B,1,H,W)
            out = images + M * self.patch_alpha * (P - images)
            return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if adv_batch.dim() == 5:
            # (B, K, 3, H, W)
            B, K, C, H, W = adv_batch.shape
            P = torch.nan_to_num(adv_batch, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            M = (P.abs().sum(dim=2, keepdim=True) > 1e-6).to(P.dtype)  # (B,K,1,H,W)

            A = self.patch_alpha * M
            one_minus_A = 1.0 - A

            # prefix products of (1-A)
            prefix = torch.cumprod(one_minus_A, dim=1)  # (B,K,1,H,W)
            prefix_excl = torch.cat([torch.ones_like(prefix[:, :1]), prefix[:, :-1]], dim=1)

            coeff_img = prefix[:, -1]  # (B,1,H,W)

            A3 = A.expand(-1, -1, C, -1, -1)
            pref3 = prefix_excl.expand(-1, -1, C, -1, -1)
            contrib = (A3 * pref3) * P
            contrib_sum = contrib.sum(dim=1)  # (B,3,H,W)

            out = images * coeff_img.expand_as(images) + contrib_sum
            return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        raise ValueError(f"adv_batch must be 4D or 5D, got {tuple(adv_batch.shape)}")
