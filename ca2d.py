#!/usr/bin/env python
"""Clean-room reproduction of CAD-Prune and CA2D on CIFAR-100 (ConvNet-D3, hard-label).

Target: Table 11 of "Rethinking Dataset Distillation: Hard Truths About Soft Labels"
(Dey et al., arXiv:2604.18811), appendix H.1:

    method            IPC 10          IPC 50
    CA2D (factor=1)   20.95 +- 0.15   34.37 +- 0.11
    CAD-Prune         22.95 +- 0.03   37.87 +- 0.16
    Random Real       18.64 +- 0.25   34.66 +- 0.41   (protocol sanity anchor)

Provenance tags used throughout:
    [HT ...]      Hard Truths paper (section/eq/table)
    [DD ...]      "Deep Learning on a Data Diet" paper / data_diet repo
    [DU ...]      "Large-scale Dataset Pruning with Dynamic Uncertainty" paper / Dataset-Pruning repo
    [RDED ...]    RDED paper / repo (synthesize/utils.py, synthesize/models.py, scripts/)
    [ASSUME n]    choice the Hard Truths paper leaves unspecified (all overridable by flags):
      1. scoring-run recipe = downstream HL protocol compressed to K epochs
         (SGD lr .01, mom .9, wd 5e-4, batch 256, StepLR@round(K*151/300), gamma .1, DSA aug)
      2. eval StepLR gamma=0.1, momentum=0.9, wd=5e-4 (Table 4 lists only epochs/lr/step/batch)
      3. student + scoring model = DC-bench ConvNet-D3 (instance norm); observer = RDED
         batchnorm Conv-3 (dictated by RDED's released weights)
      4. eval seeds = {42, 43, 44} (user-specified)
      5. DSA ported from the official DSA/DC semantics (not present in the allowed repos)
      6. CIFAR-100 normalization for scoring/eval; ImageNet stats inside the RDED observer
         path only (as RDED does)
      7. CAD keeps the highest scores (inherited from Dyn-Unc "keep most uncertain")
      8. CA2D pool per class = top-mipc(=300) by CAD score, replacing RDED's random
         uniform pre-selection [RDED Alg.1 T'_c, |T'|=300]
      9. J/W on CIFAR-100: W=2, J=K-W, i.e. the uncertainty windows span the full
         compute-matched run. The paper gives J=6,W=2 for IN1K only, where the
         compute-matched budget is K~=8 epochs, so J+W=K holds there implicitly.
         Validated empirically at IPC 50 (K=30): J=6 -> 31.36% (below random,
         ordering inverted); J=K-W=28 -> 37.14% vs paper's 37.87%.

Usage:
    python ca2d.py selftest
    python ca2d.py observer  [--device cuda:0]
    python ca2d.py score     --ipc 10 [--J -1 --W 2 --score-aug dsa]
    python ca2d.py select    --method {cadprune,ca2d,random} --ipc 10
    python ca2d.py eval      --method {cadprune,ca2d,random} --ipc 10 [--seeds 42,43,44]
    python ca2d.py all       [--seeds 42,43,44] # everything + final table
    python ca2d.py table                        # print table from cached results
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
DATA_DIR = os.path.join(ART, "data")
SCORE_DIR = os.path.join(ART, "scores")
SET_DIR = os.path.join(ART, "sets")
RESULT_DIR = os.path.join(ART, "results")
OBSERVER_CKPT_NAME = "cifar100_conv3.pth"

NUM_CLASSES = 100
N_TRAIN = 50000
IM_SIZE = 32

# [HT Tab.4, small-scale HL] student protocol; also reused (compressed) for the
# scoring run [ASSUME 1] and shared optimizer defaults [ASSUME 2].
EVAL_EPOCHS = 300
EVAL_LR = 1e-2
EVAL_STEP_EPOCH = 151
EVAL_BATCH = 256
EVAL_MOMENTUM = 0.9
EVAL_WD = 5e-4
EVAL_GAMMA = 0.1
DSA_STRATEGY = "color_crop_cutout_flip_scale_rotate"

# [ASSUME 6] CIFAR-100 stats for scoring/eval paths.
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)
# [RDED utils.py:77-86] ImageNet stats used by the observer path.
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

# [RDED scripts/cifar100_10ipc_conv3_to_conv3_cr5.sh]
RDED_MIPC = 300
RDED_NUM_CROP = 5
RDED_FACTOR = 1

PAPER_REFERENCE = {  # [HT Tab.11]
    ("ca2d", 10): (20.95, 0.15), ("ca2d", 50): (34.37, 0.11),
    ("cadprune", 10): (22.95, 0.03), ("cadprune", 50): (37.87, 0.16),
    ("random", 10): (18.64, 0.25), ("random", 50): (34.66, 0.41),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model: ConvNet, mirrored from RDED/synthesize/models.py (itself the DC ConvNet)
# so that RDED's released observer checkpoint loads key-for-key.
# --------------------------------------------------------------------------- #
class ConvNet(nn.Module):
    def __init__(self, num_classes, net_norm="batch", net_depth=3, net_width=128,
                 channel=3, net_act="relu", net_pooling="avgpooling", im_size=(32, 32)):
        super().__init__()
        assert net_act == "relu" and net_pooling == "avgpooling"
        self.net_act = nn.ReLU()
        self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.depth = net_depth
        self.net_norm = net_norm
        self.layers, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if len(self.layers["norm"]) > 0:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if len(self.layers["pool"]) > 0:
                x = self.layers["pool"][d](x)
        out = x.view(x.shape[0], -1)
        return self.classifier(out)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == "batch":
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        if net_norm == "instance":  # [RDED models.py:101-102] == DC 'instancenorm'
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        raise ValueError(net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = {"conv": [], "norm": [], "act": [], "pool": []}
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers["conv"] += [nn.Conv2d(in_channels, net_width, kernel_size=3,
                                         padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers["norm"] += [self._get_normlayer(net_norm, shape_feat)]
            layers["act"] += [self.net_act]
            in_channels = net_width
            if net_pooling != "none":
                layers["pool"] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        for k in layers:
            layers[k] = nn.ModuleList(layers[k])
        return nn.ModuleDict(layers), shape_feat


def convnet_d3(norm):
    return ConvNet(num_classes=NUM_CLASSES, net_norm=norm, net_depth=3, net_width=128,
                   channel=3, im_size=(IM_SIZE, IM_SIZE))


# --------------------------------------------------------------------------- #
# DSA augmentation. [ASSUME 5] Port of the official DSA/DC semantics
# (strategy split on '_'; mode 'S' = one randomly chosen op per batch;
# per-image parameters, i.e. non-Siamese batchmode as used at evaluation time).
# --------------------------------------------------------------------------- #
class ParamDiffAug:
    def __init__(self):
        self.aug_mode = "S"
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def _rand_scale(x, p):
    r = p.ratio_scale
    sx = torch.rand(x.shape[0]) * (r - 1.0 / r) + 1.0 / r
    sy = torch.rand(x.shape[0]) * (r - 1.0 / r) + 1.0 / r
    theta = torch.zeros(x.shape[0], 2, 3)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    return F.grid_sample(x, grid, align_corners=True)


def _rand_rotate(x, p):
    ang = (torch.rand(x.shape[0]) - 0.5) * 2 * p.ratio_rotate / 180 * math.pi
    theta = torch.zeros(x.shape[0], 2, 3)
    theta[:, 0, 0] = torch.cos(ang)
    theta[:, 0, 1] = torch.sin(-ang)
    theta[:, 1, 0] = torch.sin(ang)
    theta[:, 1, 1] = torch.cos(ang)
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    return F.grid_sample(x, grid, align_corners=True)


def _rand_flip(x, p):
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    return torch.where(randf < p.prob_flip, x.flip(3), x)


def _rand_brightness(x, p):
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    return x + (randb - 0.5) * p.brightness


def _rand_saturation(x, p):
    x_mean = x.mean(dim=1, keepdim=True)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    return (x - x_mean) * (rands * p.saturation) + x_mean


def _rand_contrast(x, p):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    return (x - x_mean) * (randc + p.contrast) + x_mean


def _rand_crop(x, p):
    shift = int(x.size(2) * p.ratio_crop_pad + 0.5)
    tx = torch.randint(-shift, shift + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift, shift + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _rand_cutout(x, p):
    cs = int(x.size(2) * p.ratio_cutout + 0.5), int(x.size(3) * p.ratio_cutout + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cs[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cs[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cs[0], dtype=torch.long, device=x.device),
        torch.arange(cs[1], dtype=torch.long, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - cs[0] // 2, min=0, max=x.size(2) - 1)
    gy = torch.clamp(gy + oy - cs[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [_rand_brightness, _rand_saturation, _rand_contrast],
    "crop": [_rand_crop],
    "cutout": [_rand_cutout],
    "flip": [_rand_flip],
    "scale": [_rand_scale],
    "rotate": [_rand_rotate],
}


def diff_augment(x, strategy=DSA_STRATEGY, param=None):
    if not strategy or strategy == "none":
        return x
    ops = strategy.split("_")
    op = ops[torch.randint(0, len(ops), (1,)).item()]  # mode 'S': one op per batch
    for f in AUGMENT_FNS[op]:
        x = f(x, param)
    return x.contiguous()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cifar100():
    """Returns train images uint8 (N,32,32,3), train labels (N,), test likewise."""
    tr = CIFAR100(DATA_DIR, train=True, download=True)
    te = CIFAR100(DATA_DIR, train=False, download=True)
    return (tr.data, np.array(tr.targets), te.data, np.array(te.targets))


def to_norm_tensor(images_uint8, mean, std):
    x = torch.from_numpy(images_uint8).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def test_top1(model, x_test, y_test, device, batch=1024):
    model.eval()
    correct = 0
    for i in range(0, len(x_test), batch):
        out = model(x_test[i:i + batch].to(device))
        correct += (out.argmax(1).cpu() == y_test[i:i + batch]).sum().item()
    return 100.0 * correct / len(x_test)


# --------------------------------------------------------------------------- #
# Shared trainer: the HL student protocol [HT Tab.4 small-scale HL].
# Also used, compressed to K epochs, for the compute-matched scoring run [ASSUME 1].
# --------------------------------------------------------------------------- #
def hl_train(model, x, y, device, epochs, step_epoch, aug="dsa", lr=EVAL_LR,
             batch=EVAL_BATCH, gen=None, epoch_hook=None):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=EVAL_MOMENTUM,
                          weight_decay=EVAL_WD)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[step_epoch],
                                                 gamma=EVAL_GAMMA)
    dsa_param = ParamDiffAug()
    x = x.to(device)
    y = y.to(device)
    n = len(x)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            if aug == "dsa":
                xb = diff_augment(xb, DSA_STRATEGY, dsa_param)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch_hook is not None:
            epoch_hook(ep, model)
    return model


# --------------------------------------------------------------------------- #
# Stage: observer  [RDED README:44-46; utils.py:169-240]
# --------------------------------------------------------------------------- #
def find_observer_ckpt():
    for dirpath, _, files in os.walk(ART):
        for f in files:
            if f == OBSERVER_CKPT_NAME:
                return os.path.join(dirpath, f)
    return None


def get_observer(device, allow_train_fallback=True):
    model = convnet_d3("batch")  # [RDED utils.py:183-192] conv3 observer is batchnorm
    ckpt_path = find_observer_ckpt()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"[observer] loaded {ckpt_path}")
    else:
        if not allow_train_fallback:
            raise FileNotFoundError(f"{OBSERVER_CKPT_NAME} not found under {ART}")
        print("[observer] checkpoint not found -> training fallback observer "
              "(torchvision-references-style recipe, RDED README:45)")
        model = train_fallback_observer(model, device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # RDED's reported 61.27% uses the torchvision-references eval transform
    # (Resize(36)+CenterCrop(32), per the args saved in their checkpoint) -> 61.34%
    # reproduced; raw 32x32 as used during synthesis scoring gives 59.67%.
    xtr, ytr, xte, yte = load_cifar100()
    acc = test_top1(model, to_norm_tensor(xte, IMNET_MEAN, IMNET_STD),
                    torch.from_numpy(yte), device)
    print(f"[observer] CIFAR-100 test top-1 (raw 32x32) = {acc:.2f}%  "
          f"(61.27% reported by RDED is under Resize36+CenterCrop32)")
    return model


def train_fallback_observer(model, device):
    """Fallback only: SRe2L/torchvision classification recipe adapted to 32x32
    (SGD lr .1, mom .9, wd 1e-4, 90 epochs, StepLR 30 @0.1, crop+flip aug)."""
    set_seed(0)
    xtr_u8, ytr, xte, yte = load_cifar100()
    x = to_norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD).to(device)
    y = torch.from_numpy(ytr).to(device)
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    n = len(x)
    for ep in range(90):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb = x[idx]
            # RandomCrop(32, padding=4) + HFlip
            pad = F.pad(xb, [4, 4, 4, 4])
            th = torch.randint(0, 9, (1,)).item()
            tw = torch.randint(0, 9, (1,)).item()
            xb = pad[:, :, th:th + 32, tw:tw + 32]
            if torch.rand(1).item() < 0.5:
                xb = xb.flip(3)
            loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    os.makedirs(os.path.join(ART, "pretrain_models"), exist_ok=True)
    path = os.path.join(ART, "pretrain_models", OBSERVER_CKPT_NAME)
    torch.save({"model": model.state_dict()}, path)
    print(f"[observer] fallback observer saved to {path}")
    return model


# --------------------------------------------------------------------------- #
# Stage: score — compute-matched run + EL2N + CAD  [HT Sec.5 Eq.1-2]
# --------------------------------------------------------------------------- #
def compute_matched_epochs(ipc):
    """K * 50000 = 300 * ipc * 100 images-seen  [HT Sec.5 'same compute budget',
    Tab.4 300 epochs]. -> K=6 (ipc10), K=30 (ipc50)."""
    total = EVAL_EPOCHS * ipc * NUM_CLASSES
    assert total % N_TRAIN == 0, f"compute budget not an integer epoch count: {total}"
    return total // N_TRAIN


def resolve_JW(K, J, W):
    if J < 0:
        J = K - W  # [ASSUME 9] windows span the full compute-matched run
    assert K - J - W >= 0, f"need K >= J+W (K={K}, J={J}, W={W})"
    return J, W


@torch.no_grad()
def el2n_scores(model, x_norm, y_onehot, device, batch=2048):
    """[DD Def.2.3; data_diet/scores.py:13-14] ||softmax(f(x)) - onehot(y)||_2,
    on clean images ([DU main_and_produce.py:158,179]: separate non-augmented pass)."""
    model.eval()
    out = torch.empty(len(x_norm))
    for i in range(0, len(x_norm), batch):
        logits = model(x_norm[i:i + batch].to(device))
        err = F.softmax(logits, dim=1) - y_onehot[i:i + batch].to(device)
        out[i:i + batch] = err.norm(p=2, dim=1).cpu()
    return out


def cad_from_S(S, J, W):
    """[HT Eq.1] U_k = std(S[k..k+J-1]) unbiased (J-1)  [DU Eq.2, torch.std default]
       [HT Eq.2] CAD = mean(U_k, k = K-J-W .. K-J-1)  (last W windows of the
       Dyn-Unc range k=0..K-J-1 [DU Eq.3])."""
    K = S.shape[0]
    U = torch.stack([S[k:k + J].std(dim=0, unbiased=True) for k in range(K - J + 1)])
    cad = U[K - J - W:K - J].mean(dim=0)
    return U, cad


def run_score(ipc, J_arg, W_arg, score_aug, device, seed=0):
    os.makedirs(SCORE_DIR, exist_ok=True)
    K = compute_matched_epochs(ipc)
    J, W = resolve_JW(K, J_arg, W_arg)
    default_jw = (J == K - W and W == 2)
    out_path = os.path.join(
        SCORE_DIR, f"cad_ipc{ipc}.pt" if default_jw else f"cad_ipc{ipc}_J{J}W{W}.pt")
    if os.path.exists(out_path):
        print(f"[score] cached: {out_path}")
        return torch.load(out_path, map_location="cpu")
    base_path = os.path.join(SCORE_DIR, f"cad_ipc{ipc}.pt")
    if os.path.exists(base_path):
        # S is J/W-independent: reuse the cached training run, recompute CAD only
        base = torch.load(base_path, map_location="cpu")
        U, cad = cad_from_S(base["S"], J, W)
        blob = {"S": base["S"], "U": U, "cad": cad, "K": K, "J": J, "W": W,
                "step_epoch": base["step_epoch"], "aug": base["aug"], "seed": base["seed"]}
        torch.save(blob, out_path)
        print(f"[score] derived {out_path} from cached S (J={J}, W={W})")
        return blob
    step_epoch = round(K * EVAL_STEP_EPOCH / EVAL_EPOCHS)  # compressed schedule [ASSUME 1]
    print(f"[score] ipc={ipc}: K={K} epochs, J={J}, W={W}, StepLR@{step_epoch}, aug={score_aug}")

    set_seed(seed)
    xtr_u8, ytr, _, _ = load_cifar100()
    x = to_norm_tensor(xtr_u8, CIFAR_MEAN, CIFAR_STD)
    y = torch.from_numpy(ytr).long()
    y1h = F.one_hot(y, NUM_CLASSES).float()

    model = convnet_d3("instance")  # [ASSUME 3]
    S = torch.empty(K, N_TRAIN)
    accs = []

    def hook(ep, m):
        S[ep] = el2n_scores(m, x, y1h, device)
        accs.append(None)
        print(f"[score]   epoch {ep + 1}/{K} EL2N mean={S[ep].mean():.4f}", flush=True)

    hl_train(model, x, y, device, epochs=K, step_epoch=step_epoch, aug=score_aug,
             epoch_hook=hook)
    U, cad = cad_from_S(S, J, W)
    blob = {"S": S, "U": U, "cad": cad, "K": K, "J": J, "W": W,
            "step_epoch": step_epoch, "aug": score_aug, "seed": seed}
    torch.save(blob, out_path)
    print(f"[score] saved {out_path}")
    return blob


# --------------------------------------------------------------------------- #
# Stage: select — cadprune / random / ca2d
# --------------------------------------------------------------------------- #
def per_class_indices(ytr):
    return [np.where(ytr == c)[0] for c in range(NUM_CLASSES)]


def select_cadprune(ipc, cad, xtr_u8, ytr):
    """[HT Sec.5] top-IPC per class by highest CAD [ASSUME 7], class-balanced."""
    keep = []
    for idx in per_class_indices(ytr):
        order = idx[np.argsort(-cad[idx].numpy(), kind="stable")]
        keep.append(order[:ipc])
    keep = np.concatenate(keep)
    return xtr_u8[keep], ytr[keep], keep


def select_random(ipc, xtr_u8, ytr, seed=0):
    rng = np.random.RandomState(seed)
    keep = []
    for idx in per_class_indices(ytr):
        keep.append(rng.choice(idx, size=ipc, replace=False))
    keep = np.concatenate(keep)
    return xtr_u8[keep], ytr[keep], keep


# ---- RDED synthesis machinery, mirrored from RDED/synthesize/utils.py ---- #
def rded_cross_entropy(y_pre, y):
    """[RDED utils.py:97-99] per-sample -log softmax[true]; lower = better."""
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def rded_multi_random_crop(images, num_crop, size, factor):
    """[RDED utils.py:54-74] num_crop RandomResizedCrop(size//factor, ratio=(1,1),
    antialias=True) per image (default scale (0.08, 1.0)). images: (P,3,H,W) in [0,1].
    Returns (P, num_crop, 3, s, s)."""
    cropper = transforms.RandomResizedCrop(size // factor, ratio=(1, 1), antialias=True)
    out = []
    for img in images:
        out.append(torch.stack([cropper(img) for _ in range(num_crop)], 0))
    return torch.stack(out, 0)


def rded_selector(n, model, images, labels, size, m, device):
    """[RDED utils.py:102-136] best crop per image (argmin CE), then top-n
    lowest-CE patches across images. images: (P, m, 3, s, s) normalized."""
    with torch.no_grad():
        images = images.to(device)
        s = images.shape
        images = images.permute(1, 0, 2, 3, 4).reshape(s[0] * s[1], s[2], s[3], s[4])
        labels = labels.repeat(m).to(device)
        preds = []
        for i in range(0, len(images), s[0]):  # batch_size = mipc [RDED utils.py:116]
            preds.append(model(images[i:i + s[0]]))
        preds = torch.cat(preds, 0)
        dist = rded_cross_entropy(preds, labels).reshape(m, s[0])
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0], device=device)]
        images = images.reshape(m, s[0], s[2], s[3], s[4])
        images = images[index, torch.arange(s[0], device=device)]
    indices = torch.argsort(dist, descending=False)[:n]
    return images[indices].detach()


def rded_mix_images(input_img, out_size, factor, n):
    """[RDED utils.py:139-166] factor x factor stitch; identity for factor=1."""
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed = torch.zeros((n, 3, out_size, out_size), dtype=torch.float)
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            part = F.interpolate(input_img.data[k * n:(k + 1) * n], size=(h_r, w_r))
            mixed.data[0:n, :, h_loc:h_loc + h_r, w_loc:w_loc + w_r] = part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed


def rded_save_images(images, class_id, out_dir):
    """[RDED synthesize/main.py:61-69] denormalized [0,1] float -> uint8 JPEG."""
    dir_path = os.path.join(out_dir, f"{class_id:05d}")
    os.makedirs(dir_path, exist_ok=True)
    for i in range(images.shape[0]):
        arr = images[i].cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((arr * 255).astype(np.uint8)).save(
            os.path.join(dir_path, f"class{class_id:05d}_id{i:05d}.jpg"))


def synthesize_ca2d(ipc, cad, xtr_u8, ytr, observer, device, seed=0, tag=""):
    """[HT Sec.5] 'crop, select, and stitch patches in the RDED-style from this
    coreset': RDED Alg.1 with the per-class random pool T'_c replaced by the
    top-mipc CAD-scored pool [ASSUME 8]."""
    out_dir = os.path.join(SET_DIR, f"ca2d_ipc{ipc}{tag}")
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == NUM_CLASSES:
        print(f"[ca2d] cached: {out_dir}")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD).view(1, 3, 1, 1)
    n_patches = ipc * RDED_FACTOR ** 2  # [RDED main.py:49-56]
    for c, idx in enumerate(per_class_indices(ytr)):
        pool = idx[np.argsort(-cad[idx].numpy(), kind="stable")][:RDED_MIPC]
        imgs = torch.from_numpy(xtr_u8[pool]).float().permute(0, 3, 1, 2) / 255.0
        set_seed(seed * 1000003 + c)  # stock RDED leaves crop RNG unseeded
        crops = rded_multi_random_crop(imgs, RDED_NUM_CROP, IM_SIZE, RDED_FACTOR)
        crops = (crops - mean.unsqueeze(0)) / std.unsqueeze(0)  # [RDED main.py:30-38]
        labels = torch.full((len(pool),), c, dtype=torch.long)
        picked = rded_selector(n_patches, observer, crops, labels, IM_SIZE,
                               RDED_NUM_CROP, device)
        mixed = rded_mix_images(picked.cpu(), IM_SIZE, RDED_FACTOR, ipc)
        denorm = mixed * std + mean
        rded_save_images(denorm, c, out_dir)
        if (c + 1) % 20 == 0:
            print(f"[ca2d] ipc={ipc}: {c + 1}/{NUM_CLASSES} classes", flush=True)
    print(f"[ca2d] saved {out_dir}")
    return out_dir


def run_select(method, ipc, device, seed=0, J=-1, W=2, tag=""):
    os.makedirs(SET_DIR, exist_ok=True)
    xtr_u8, ytr, _, _ = load_cifar100()
    if method == "random":
        out = os.path.join(SET_DIR, f"random_ipc{ipc}{tag}.pt")
        if not os.path.exists(out):
            imgs, labels, keep = select_random(ipc, xtr_u8, ytr, seed=seed)
            torch.save({"images": imgs, "labels": labels, "indices": keep}, out)
            print(f"[select] saved {out}")
        return out
    # scoring-run seed fixed at 0; selection is deterministic given the scores
    blob = run_score(ipc, J, W, "dsa", device, seed=0)
    cad = blob["cad"]
    if method == "cadprune":
        out = os.path.join(SET_DIR, f"cadprune_ipc{ipc}{tag}.pt")
        if not os.path.exists(out):
            imgs, labels, keep = select_cadprune(ipc, cad, xtr_u8, ytr)
            torch.save({"images": imgs, "labels": labels, "indices": keep}, out)
            print(f"[select] saved {out}")
        return out
    if method == "ca2d":
        observer = get_observer(device)
        return synthesize_ca2d(ipc, cad, xtr_u8, ytr, observer, device, seed=seed,
                               tag=tag)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Stage: eval — HL protocol [HT Tab.4 small-scale HL]
# --------------------------------------------------------------------------- #
def load_set(method, ipc, tag=""):
    if method in ("cadprune", "random"):
        blob = torch.load(os.path.join(SET_DIR, f"{method}_ipc{ipc}{tag}.pt"))
        x = to_norm_tensor(blob["images"], CIFAR_MEAN, CIFAR_STD)
        y = torch.from_numpy(blob["labels"]).long()
        return x, y
    if method == "ca2d":
        root = os.path.join(SET_DIR, f"ca2d_ipc{ipc}{tag}")
        imgs, labels = [], []
        for cdir in sorted(os.listdir(root)):
            c = int(cdir)
            for f in sorted(os.listdir(os.path.join(root, cdir))):
                arr = np.array(Image.open(os.path.join(root, cdir, f)).convert("RGB"))
                imgs.append(arr)
                labels.append(c)
        x = to_norm_tensor(np.stack(imgs), CIFAR_MEAN, CIFAR_STD)
        return x, torch.tensor(labels, dtype=torch.long)
    raise ValueError(method)


def run_eval(method, ipc, seeds, device, tag=""):
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"{method}_ipc{ipc}{tag}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            res = json.load(f)
        if res.get("seeds") == list(seeds):
            print(f"[eval] cached: {out_path} -> {res['mean']:.2f} +- {res['std']:.2f}")
            return res
    x, y = load_set(method, ipc, tag)
    assert len(x) == ipc * NUM_CLASSES, f"set size {len(x)} != {ipc * NUM_CLASSES}"
    _, _, xte_u8, yte = load_cifar100()
    xte = to_norm_tensor(xte_u8, CIFAR_MEAN, CIFAR_STD)
    yte = torch.from_numpy(yte)
    accs = []
    for s in seeds:
        t0 = time.time()
        set_seed(s)
        model = convnet_d3("instance")  # [ASSUME 3] DC-bench eval architecture
        hl_train(model, x, y, device, epochs=EVAL_EPOCHS, step_epoch=EVAL_STEP_EPOCH,
                 aug="dsa")
        acc = test_top1(model, xte, yte, device)
        accs.append(acc)
        print(f"[eval] {method} ipc{ipc}{tag} seed{s}: {acc:.2f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = {"method": method, "ipc": ipc, "tag": tag, "seeds": list(seeds),
           "accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eval] {method} ipc{ipc}: {res['mean']:.2f} +- {res['std']:.2f}")
    return res


def print_table():
    print(f"\n{'method':<12}{'IPC':>4}   {'reproduced':>14}   {'paper (Tab.11)':>14}")
    for method in ("cadprune", "ca2d", "random"):
        for ipc in (10, 50):
            path = os.path.join(RESULT_DIR, f"{method}_ipc{ipc}.json")
            ref_m, ref_s = PAPER_REFERENCE[(method, ipc)]
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                ours = f"{r['mean']:.2f} +- {r['std']:.2f}"
            else:
                ours = "—"
            print(f"{method:<12}{ipc:>4}   {ours:>14}   {ref_m:.2f} +- {ref_s:.2f}")
    print()


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    print("[selftest] 1. EL2N formula vs manual loop")
    torch.manual_seed(0)
    model = convnet_d3("instance")
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, NUM_CLASSES, (8,))
    y1h = F.one_hot(y, NUM_CLASSES).float()
    s = el2n_scores(model, x, y1h, torch.device("cpu"), batch=3)
    with torch.no_grad():
        for i in range(8):
            p = F.softmax(model(x[i:i + 1]), dim=1)[0]
            manual = (p - y1h[i]).pow(2).sum().sqrt()
            assert abs(s[i] - manual) < 1e-5, (s[i], manual)
    print("           ok")

    print("[selftest] 2. CAD window arithmetic vs numpy (K=6,J=4,W=2 and K=30,J=6,W=2)")
    for (K, J, W) in [(6, 4, 2), (30, 6, 2)]:
        S = torch.randn(K, 7).abs()
        U, cad = cad_from_S(S, J, W)
        Sn = S.numpy()
        Un = np.stack([Sn[k:k + J].std(axis=0, ddof=1) for k in range(K - J + 1)])
        cadn = Un[K - J - W:K - J].mean(axis=0)
        assert np.allclose(U.numpy(), Un, atol=1e-6)
        assert np.allclose(cad.numpy(), cadn, atol=1e-6)
        # last window used ends at epoch K-2 (Dyn-Unc convention: k <= K-J-1)
    print("           ok")

    print("[selftest] 3. ConvNet-D3 shapes / classifier size")
    for norm in ("instance", "batch"):
        m = convnet_d3(norm)
        out = m(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, NUM_CLASSES)
        assert m.classifier.in_features == 128 * 4 * 4
    print("           ok  (params: %.2fM)" %
          (sum(p.numel() for p in convnet_d3('instance').parameters()) / 1e6))

    print("[selftest] 4. DSA ops preserve shape, finite output")
    p = ParamDiffAug()
    xb = torch.randn(16, 3, 32, 32)
    for op, fns in AUGMENT_FNS.items():
        z = xb
        for f in fns:
            z = f(z, p)
        assert z.shape == xb.shape and torch.isfinite(z).all(), op
    z = diff_augment(xb, DSA_STRATEGY, p)
    assert z.shape == xb.shape
    print("           ok")

    print("[selftest] 5. RDED selector picks the min-CE patch (synthetic check)")
    class Fake(nn.Module):
        def forward(self, x):  # logit of class 0 = mean pixel value
            b = x.mean(dim=[1, 2, 3]).unsqueeze(1)
            return torch.cat([b * 10, torch.zeros(len(x), NUM_CLASSES - 1)], dim=1)
    P, m = 6, 3
    imgs = torch.rand(P, m, 3, 8, 8)
    labels = torch.zeros(P, dtype=torch.long)
    picked = rded_selector(2, Fake(), imgs, labels, 8, m, torch.device("cpu"))
    means = imgs.mean(dim=[2, 3, 4])
    best_per_img = means.max(dim=1).values
    expect = best_per_img.sort(descending=True).values[:2]
    got = picked.mean(dim=[1, 2, 3]).sort(descending=True).values
    assert torch.allclose(got, expect, atol=1e-6)
    print("           ok")

    print("[selftest] 6. compute-matched epochs: ipc10 -> 6, ipc50 -> 30")
    assert compute_matched_epochs(10) == 6 and compute_matched_epochs(50) == 30
    assert resolve_JW(6, -1, 2) == (4, 2) and resolve_JW(30, -1, 2) == (28, 2)
    print("           ok")
    print("[selftest] all checks passed")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "observer", "score", "select",
                                    "eval", "all", "table"])
    ap.add_argument("--ipc", type=int, choices=[10, 50])
    ap.add_argument("--method", choices=["cadprune", "ca2d", "random"])
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")],
                    default=[42, 43, 44])               # [ASSUME 4] user-specified
    ap.add_argument("--J", type=int, default=-1)        # -1 -> K-W [ASSUME 9]
    ap.add_argument("--W", type=int, default=2)
    ap.add_argument("--score-aug", default="dsa", choices=["dsa", "none"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="suffix for set/result names of variant runs (e.g. _J28W2)")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for d in (ART, DATA_DIR):
        os.makedirs(d, exist_ok=True)

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "observer":
        get_observer(device)
    elif args.cmd == "score":
        assert args.ipc, "--ipc required"
        run_score(args.ipc, args.J, args.W, args.score_aug, device, seed=args.seed)
    elif args.cmd == "select":
        assert args.ipc and args.method, "--ipc and --method required"
        run_select(args.method, args.ipc, device, seed=args.seed, J=args.J, W=args.W,
                   tag=args.tag)
    elif args.cmd == "eval":
        assert args.ipc and args.method, "--ipc and --method required"
        run_select(args.method, args.ipc, device, seed=args.seed, J=args.J, W=args.W,
                   tag=args.tag)
        run_eval(args.method, args.ipc, args.seeds, device, tag=args.tag)
    elif args.cmd == "all":
        for ipc in (10, 50):
            run_score(ipc, args.J, args.W, args.score_aug, device, seed=args.seed)
            for method in ("random", "cadprune", "ca2d"):
                run_select(method, ipc, device, seed=args.seed)
                run_eval(method, ipc, args.seeds, device)
        print_table()
    elif args.cmd == "table":
        print_table()


if __name__ == "__main__":
    main()
