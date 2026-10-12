"""Core SCD-Det feature modules.

The implementation matches the architecture used for the revision experiments:
FREG (k=12, S=10, 8 heads, ratio=0.25), CEBA (9x9 context kernel),
and UMA aligned to P3. Parameters remain configurable for ablation studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class CEBA(nn.Module):
    """Contrast-enhanced background-aware module with local/context interaction."""

    def __init__(self, c1: int, c2: int, reduction: int = 8, context_kernel: int = 9):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"CEBA requires equal input/output channels, got {c1} and {c2}")
        if context_kernel not in {5, 7, 9}:
            raise ValueError(f"context_kernel must be 5, 7, or 9, got {context_kernel}")
        _ = reduction  # Kept in the public signature for YAML/config compatibility.
        self.local = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(c1, c1, context_kernel, padding=context_kernel // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(),
        )
        self.fusion = nn.Conv2d(c1, c1, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.fusion(self.local(x) * self.context(x))
        return x + self.gamma * enhanced


class FREG(nn.Module):
    """Feature refinement with max-energy routing and depthwise edge compensation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 12,
        s: int = 10,
        num_heads: int = 8,
        ratio: float = 0.25,
    ):
        super().__init__()
        if s <= 0 or k <= 0 or k > s * s:
            raise ValueError(f"FREG requires s > 0 and 0 < k <= s^2, got k={k}, s={s}")
        mid_channels = int(c1 * ratio)
        if mid_channels <= 0 or mid_channels % num_heads:
            raise ValueError(
                f"int(c1 * ratio) must be positive and divisible by num_heads, got "
                f"c1={c1}, ratio={ratio}, num_heads={num_heads}"
            )

        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.num_heads = num_heads
        self.mid_channels = mid_channels
        self.scale = (mid_channels // num_heads) ** -0.5
        self.qkv = nn.Linear(c1, mid_channels * 3, bias=False)
        self.proj = nn.Linear(mid_channels, c1, bias=False)
        self.edge = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.channel_adjust = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        identity = x

        pad_h = (self.s - height % self.s) % self.s
        pad_w = (self.s - width % self.s) % self.s
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        padded_h, padded_w = x.shape[2:]
        region_h, region_w = padded_h // self.s, padded_w // self.s
        pixels_per_region = region_h * region_w
        regions = (
            x.view(batch, channels, self.s, region_h, self.s, region_w)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(batch, self.s * self.s, pixels_per_region, channels)
        )

        energy = regions.max(dim=2).values.mean(dim=-1)
        top_indices = torch.topk(energy, self.k, dim=-1).indices
        sorted_indices = torch.argsort(energy, dim=-1, descending=True)
        step = max(1, (self.s * self.s - self.k) // self.k)
        skip_indices = sorted_indices[:, self.k :: step][:, : self.k]
        if skip_indices.shape[1] < self.k:
            missing = self.k - skip_indices.shape[1]
            skip_indices = torch.cat((skip_indices, sorted_indices[:, -missing:]), dim=1)
        route_indices = torch.cat((top_indices, skip_indices), dim=1)

        query, key, value = self.qkv(regions).chunk(3, dim=-1)
        head_dim = self.mid_channels // self.num_heads
        query = query.reshape(
            batch, self.s * self.s, pixels_per_region, self.num_heads, head_dim
        ).permute(0, 1, 3, 2, 4)
        key = key.flatten(2)
        value = value.flatten(2)
        batch_indices = torch.arange(batch, device=x.device).view(batch, 1, 1)
        routed_key = key[batch_indices, route_indices]
        routed_value = value[batch_indices, route_indices]
        routed_key = routed_key.view(batch, 1, self.num_heads, -1, head_dim)
        routed_value = routed_value.view(batch, 1, self.num_heads, -1, head_dim)

        attention = (query @ routed_key.transpose(-2, -1)) * self.scale
        main = attention.softmax(dim=-1) @ routed_value
        main = main.permute(0, 1, 3, 2, 4).reshape(
            batch, self.s * self.s, pixels_per_region, self.mid_channels
        )
        main = self.proj(main)

        edge_input = regions.transpose(1, 2).reshape(batch, channels, padded_h, padded_w)
        edge = self.edge(edge_input)
        edge = (
            edge.view(batch, channels, self.s, region_h, self.s, region_w)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(batch, self.s * self.s, pixels_per_region, channels)
        )

        out = (main + edge).view(batch, self.s, self.s, region_h, region_w, channels)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)
        if pad_h or pad_w:
            out = out[:, :, :height, :width]
        out = self.channel_adjust(out)
        return identity + out if self.c1 == self.c2 else out


class UMA(nn.Module):
    """Unified multiscale aggregator with learned, non-exclusive scale weights."""

    def __init__(self, ch: list[int], out_channels: int = 256, target_level: int = 0):
        super().__init__()
        if len(ch) != 3:
            raise ValueError(f"UMA expects three input scales, got {len(ch)}")
        if target_level not in {0, 1}:
            raise ValueError(f"target_level must be 0 (P3) or 1 (P4), got {target_level}")
        self.target_level = target_level
        self.align = nn.ModuleList(Conv(c, out_channels, 1) for c in ch)
        self.scale_attention = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels // 4),
            nn.SiLU(),
            nn.Linear(out_channels // 4, 3),
            nn.Sigmoid(),
        )
        self.refine = Conv(out_channels, out_channels, 3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        target_size = x[self.target_level].shape[2:]
        aligned = []
        for feature, align in zip(x, self.align):
            feature = align(feature)
            if feature.shape[2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="nearest")
            aligned.append(feature)

        context = torch.cat([F.adaptive_avg_pool2d(a, 1).flatten(1) for a in aligned], dim=1)
        weights = self.scale_attention(context).unsqueeze(-1).unsqueeze(-1)
        fused = sum(aligned[i] * weights[:, i : i + 1] for i in range(3))
        fused = fused + aligned[self.target_level]
        return [self.refine(fused)]
