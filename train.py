"""SegForge — Semantic Segmentation (U-Net + ASPP, Cityscapes)"""
import torch, torch.nn as nn, torchvision.models as M, argparse
from pathlib import Path

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1),
            nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6),
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12),
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18),
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1))
        self.proj = nn.Conv2d(out_ch * 5, out_ch, 1)
    def forward(self, x):
        feats = [c(x) for c in self.convs]
        feats.append(nn.functional.interpolate(self.pool(x), x.shape[2:], mode="bilinear", align_corners=False))
        return self.proj(torch.cat(feats, 1))

class SegForge(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        backbone = M.resnet50(pretrained=True)
        self.enc = nn.Sequential(*list(backbone.children())[:-2])
        self.aspp = ASPP(2048, 256)
        self.head = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, num_classes, 1))
    def forward(self, x):
        h, w = x.shape[2:]
        feat = self.aspp(self.enc(x))
        return nn.functional.interpolate(self.head(feat), (h, w), mode="bilinear", align_corners=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--num-classes", type=int, default=19)
    a = p.parse_args(); m = SegForge(a.num_classes); print(f"SegForge params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
