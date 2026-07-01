"""
Self-contained TEED (Tiny and Efficient Edge Detector) module.

Architecture and weights from https://github.com/xavysp/TEED
Smish activation from Wang et al. (2022), Electronics 11.4.
"""

import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Smish activation
# ---------------------------------------------------------------------------

@torch.jit.script
def _smish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class _Smish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _smish(x)


# ---------------------------------------------------------------------------
# TED architecture (verbatim from ted.py, imports replaced)
# ---------------------------------------------------------------------------

def _weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class _CoFusion(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class _DoubleFusion(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24)
        self.AF = _Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return _smish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True))
        self.add_module("smish1", _Smish())
        self.add_module("conv2", nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True))

    def forward(self, x):  # type: ignore[override]
        x1, x2 = x
        new_features = super().forward(_smish(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, input_features: int, out_features: int):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module(f"denselayer{i + 1}", layer)
            input_features = out_features


class _UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int):
        super().__init__()
        self.features = nn.Sequential(*self._make_layers(in_features, up_scale))

    def _make_layers(self, in_features: int, up_scale: int):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        constant_features = 16
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = 1 if i == up_scale - 1 else constant_features
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(_Smish())
            layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, use_ac: bool = False):
        super().__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        if use_ac:
            self.smish = _Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.smish(x) if self.use_ac else x


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int | None = None,
                 stride: int = 1, use_act: bool = True):
        super().__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = _Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smish(self.conv1(x))
        x = self.conv2(x)
        return self.smish(x) if self.use_act else x


class TED(nn.Module):
    """Tiny and Efficient Edge Detector."""

    def __init__(self):
        super().__init__()
        self.block_1 = _DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = _DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = _SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = _SingleConvBlock(32, 48, 1)
        self.up_block_1 = _UpConvBlock(16, 1)
        self.up_block_2 = _UpConvBlock(32, 1)
        self.up_block_3 = _UpConvBlock(48, 2)
        self.block_cat = _DoubleFusion(3, 3)
        self.apply(_weight_init)

    def forward(self, x: torch.Tensor, single_test: bool = False) -> list[torch.Tensor]:
        assert x.ndim == 4
        h, w = x.shape[2], x.shape[3]

        if single_test:
            img_w = ((w // 8) + 1) * 8 if w % 8 != 0 else w
            img_h = ((h // 8) + 1) * 8 if h % 8 != 0 else h
            if img_w != w or img_h != h:
                x = F.interpolate(x, size=(img_h, img_w), mode="bicubic", align_corners=False)

        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        # resize all outputs back to original input size if needed
        results = []
        for out in [out_1, out_2, out_3]:
            if out.shape[2] != h or out.shape[3] != w:
                out = F.interpolate(out, size=(h, w), mode="bicubic", align_corners=False)
            results.append(out)

        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        return results


# ---------------------------------------------------------------------------
# Checkpoint download + model loading
# ---------------------------------------------------------------------------

_TEED_CHECKPOINT_URL = (
    "https://github.com/xavysp/TEED/raw/main/checkpoints/BIPED/5/5_model.pth"
)
_TEED_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def load_teed_net(model_dir: Path) -> TED:
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "teed_5_model.pth"

    if not ckpt_path.exists():
        print(f"  Downloading TEED checkpoint (~6 MB) → {ckpt_path}")
        urllib.request.urlretrieve(_TEED_CHECKPOINT_URL, ckpt_path)

    device = torch.device("cpu")
    model = TED().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def run_teed(model: TED, img_np: np.ndarray) -> np.ndarray:
    """Run TEED on an RGBA/RGB numpy image; returns a uint8 edge map [0,255]."""
    img_bgr = img_np[:, :, [2, 1, 0]].astype(np.float32)  # RGB→BGR
    img_bgr -= _TEED_MEAN_BGR

    tensor = torch.from_numpy(img_bgr.transpose(2, 0, 1)).unsqueeze(0)  # 1×C×H×W
    with torch.no_grad():
        preds = model(tensor, single_test=True)

    fused = torch.sigmoid(preds[-1]).squeeze().cpu().numpy()  # float32 in [0,1]
    return (fused * 255).clip(0, 255).astype(np.uint8)
