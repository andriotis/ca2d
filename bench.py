#!/usr/bin/env python
"""bench.py — single-file driver for the four CVPR tables.

Tables ([HT] Tab.1 structure): conv3-cifar100, resnet18-cifar100, conv4-tinyimagenet,
resnet18-tinyimagenet. Columns = HL / SL / KD+SL x IPC {10,50,100}; rows = Coreset
Selection (Random Real, EL2N-Best, CAD-Prune, SHARP (ours), R-CAD (ours)) and
Dataset Distillation (RDED, CA2D, SHARP-2D (ours), R-CAD-2D (ours) — the 2D
variants replace CA2D's CAD-ranked mipc pool with the SHARP/R-CAD FL selection at
the same budget [REPORT Sec.4]); mean +- std over seeds {42,43,44} with one-sided
Welch p-values. A compute-matched Full-Dataset reference cell can be produced on
demand (`cell --method full` / `run --methods full`) but is not shown in tables.

Protocols
  HL     [HT Tab.4 small-scale HL]  ca2d.hl_train verbatim (CE, SGD 1e-2, StepLR@151,
         300 ep, batch 256, DSA). Students = DC-bench variants: instance-norm
         ConvNet-D3/D4, DC-bench RN18 for the resnet18 tables [HT Tab.7/8 variant].
  SL     [HT Tab.4 small-scale SL]  xarch.sl_train verbatim (fixed per-image teacher
         soft label, KL T=20, AdamW 1e-3, cosine, 300 ep, batch 256, DSA).
  KD+SL  RDED's official validation code run UNMODIFIED in a subprocess worker
         (on-the-fly teacher soft labels on cutmix batches, KL T=20, AdamW,
         quarter-cosine, 300 ep, official batch/lr incl. the hidden cifar100-conv3
         override bs=25/lr=2e-3 [RDED argument.py:309-315]). Students = RDED's own
         models (batch-norm conv3/conv4, resnet18_modified).

Teachers/observers = released RDED checkpoints of the table's arch
(artifacts/pretrain_models/{subset}_{arch}.pth). Selection is computed once per
(dataset, table-arch, ipc) from an HL-recipe compute-matched scoring run (seed 0)
and the same subsets are evaluated under all three regimes [HT convention].

Implicit choices (documented for the paper):
  [B1]  small-scale KD+SL is undefined in [HT]; we use RDED's official validation.
  [B2]  KD ipc100 batch: cifar100=200 ([HT Tab.4] large-scale HL ipc100 batch; RDED
        has no ipc100 rule), tinyimagenet=100 (official rule for all ipc).
  [B3]  EL2N = mean over 10 runs (seeds 1000-1009) of ||softmax-onehot||_2 at epoch
        20 of the truncated HL recipe, table-arch student [DD Def.2.3; DD README
        ">= 10 runs"; epoch-20 = DD Fig.1 early-epoch convention].
  [B4]  EL2N-Best window grid = start offsets {0,10,..,90}% of (pool-ipc) over the
        per-class descending EL2N ranking; winner by TEST accuracy (as [HT]
        implicitly does — flagged in the paper text); search at seed 42 only,
        winner re-run at 3 seeds (seed-42 search result reused).
  [B5]  coresets are exported as PNG for the KD path (JPEG re-encoding costs
        1.6-3.1pp at 32px); synthesized sets stay official JPEG.
  [B6]  Full-Dataset rows are compute-matched: epochs = 300*ipc*nclass/N = 6/30/60,
        schedules compressed proportionally; in KD the best-acc window degenerates
        to the final-epoch validation (validation/main.py:179-183 fires at the last
        epoch iff re_epochs >= 6).
  [B7]  per-column student convention: DC-bench students for HL/SL (comparable to
        [HT Tab.2/6/11]); RDED students for KD (forced by B1). Footnoted in tables.
  [B8]  scoring run = single seed-0 run ([HT] is silent on averaging).
  [B9]  RDED synthesis via official synthesize/main.py; its pool shuffle / crop RNG
        is unseeded upstream — the worker seeds python/np/torch with 0.
  [B10] Random rows: fixed class-balanced draw (RandomState(0)) x 3 training seeds.

Usage:
    python bench.py selftest
    python bench.py prepare                       # RDED-layout data trees + symlinks
    python bench.py score  --ds tin --arch rn18 --ipc 100
    python bench.py el2n   --ds cifar100 --arch conv
    python bench.py select --ds cifar100 --arch conv --method rcad --ipc 100
    python bench.py cell   --ds cifar100 --arch conv --method rded --regime kd --ipc 10
    python bench.py run    --table conv-cifar100 [--regimes hl,sl,kd] [--methods ...]
    python bench.py table  [--table rn18-tin]
    python bench.py tex
    python bench.py timing
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as thmodels
from PIL import Image
from tqdm import tqdm

import ca2d
import rcad
import tin
import xarch
from ca2d import (IMNET_MEAN, IMNET_STD, RESULT_DIR, SCORE_DIR, SET_DIR,
                  cad_from_S, el2n_scores, hl_train, set_seed, test_top1,
                  to_norm_tensor)
from xarch import KD_T, sl_train, welch_p

ROOT = ca2d.ROOT
ART = ca2d.ART
RDED_ROOT = os.path.join(os.path.dirname(ROOT), "RDED")
RDED_CWD = os.path.join(ART, "rded")          # worker cwd (./data/... resolves here)
EXPORT_DIR = os.path.join(ART, "rded_sets")   # PNG ImageFolder exports of coresets
LOG_DIR = os.path.join(ART, "logs")
TEX_DIR = os.path.join(ROOT, "tex")

BDS = {
    "cifar100": dict(nclass=100, size=32, ntrain=50000, pool=500, subset="cifar100",
                     mean=ca2d.CIFAR_MEAN, std=ca2d.CIFAR_STD, conv="conv3",
                     val_ipc=100),
    "tin": dict(nclass=200, size=64, ntrain=100000, pool=500, subset="tinyimagenet",
                mean=tin.TIN_MEAN, std=tin.TIN_STD, conv="conv4", val_ipc=50),
}
TABLES = {
    "conv-cifar100": ("cifar100", "conv"), "rn18-cifar100": ("cifar100", "rn18"),
    "conv-tin": ("tin", "conv"), "rn18-tin": ("tin", "rn18"),
}
IPCS = (10, 50, 100)
REGIMES = ("hl", "sl", "kd")
METHODS = ("random", "el2nbest", "cadprune", "sharp", "rcad", "rded", "ca2d",
           "sharp2d", "rcad2d", "full")
OURS = ("sharp", "rcad", "sharp2d", "rcad2d")
EL2N_SEEDS = tuple(range(1000, 1010))   # [B3]
EL2N_EPOCH = 20                         # [B3]
EL2N_OFFSETS = tuple(range(0, 100, 10))  # [B4] % of (pool - ipc)
RE_BEST = re.compile(r"Best accuracy is ([0-9.]+)@(\d+)")
RE_EPOCH = re.compile(r"TRAIN Iter (\d+): loss = ([0-9.eE+-]+)")

# Pipeline anchors (not shown in tables): [RDED Tab.2] KD+SL under RDED's own
# validation — cifar100 conv 48.1/57.0, rn18 42.6/62.6; tin conv 39.6/47.6,
# rn18 41.9/58.2 (ipc 10/50). Check the rded/kd cells against these.


def rded_arch(ds, arch):
    """RDED model name for the table's teacher and KD student."""
    return BDS[ds]["conv"] if arch == "conv" else "resnet18_modified"


def compute_K(ds, ipc):
    """Compute-matched epochs: K*N = 300*ipc*nclass [HT Sec.5] -> 6/30/60."""
    total = 300 * ipc * BDS[ds]["nclass"]
    assert total % BDS[ds]["ntrain"] == 0, (ds, ipc)
    return total // BDS[ds]["ntrain"]


def kd_batch_lr(ds, arch, ipc):
    """Official KD batch/lr [RDED argument.py:234-258,277-315]; ipc100 per [B2]."""
    if BDS[ds]["subset"] == "tinyimagenet":
        bs = 100                                     # argument.py:252-254 (all ipc)
    elif ipc in (1, 10, 50):
        bs = {1: 10, 10: 50, 50: 100}[ipc] * 2       # nclass==100 -> x2
    elif ipc == 100:
        bs = 200                                     # [B2]
    else:
        raise ValueError(ipc)
    lr = 0.001
    if ds == "cifar100" and rded_arch(ds, arch) == "conv3":
        bs, lr = 25, 0.002                           # argument.py:309-315
    return bs, lr


def class_indices(y, nclass):
    return [np.where(y == c)[0] for c in range(nclass)]


def load_train(ds):
    if ds == "cifar100":
        xtr, ytr, _, _ = ca2d.load_cifar100()
    else:
        xtr, ytr, _, _, _ = tin.load_tin()
    return xtr, ytr


def gpu_index(device):
    s = str(device)
    if "cuda" not in s:
        raise ValueError(f"RDED worker needs a CUDA device, got {s}")
    return s.split(":")[1] if ":" in s else "0"


def bar_hook(bar):
    return lambda ep, m: bar.update(1)


# --------------------------------------------------------------------------- #
# Students (HL/SL: DC-bench variants) and teachers (released RDED checkpoints)
# --------------------------------------------------------------------------- #
def build_student(ds, arch):
    if arch == "conv":
        return ca2d.convnet_d3("instance") if ds == "cifar100" \
            else tin.tin_convnet("instance")
    return xarch.build_model("rn18", ds)  # DC-bench RN18 [HT Tab.7/8 variant]


_TEACHERS = {}


def build_teacher(ds, arch, device):
    """Released RDED checkpoint of the table's arch, frozen eval."""
    key = (ds, arch)
    if key in _TEACHERS:
        return _TEACHERS[key]
    name = rded_arch(ds, arch)
    nclass = BDS[ds]["nclass"]
    if name.startswith("conv"):
        model = ca2d.convnet_d3("batch") if ds == "cifar100" \
            else tin.tin_convnet("batch")   # RDED conv observers are batch-norm
    else:  # resnet18_modified [RDED synthesize/utils.py:193-198]
        model = thmodels.resnet18(weights=None, num_classes=nclass)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    ckpt_path = os.path.join(ART, "pretrain_models",
                             f"{BDS[ds]['subset']}_{name}.pth")
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _TEACHERS[key] = model
    return model


@torch.no_grad()
def teacher_features(model, x, device, batch=512, normalize=True):
    """Penultimate features; L2-normalized for cosine kernels (repo convention)."""
    if isinstance(model, ca2d.ConvNet):
        return rcad.observer_features(model, x, device, normalize=normalize)
    feats = []
    for i in range(0, len(x), batch):
        z = x[i:i + batch].to(device)
        z = model.relu(model.bn1(model.conv1(z)))
        z = model.maxpool(z)
        z = model.layer4(model.layer3(model.layer2(model.layer1(z))))
        z = model.avgpool(z).flatten(1)
        feats.append((F.normalize(z, dim=1) if normalize else z).cpu())
    return torch.cat(feats)


@torch.no_grad()
def collect_probs(model, x, nclass, device, batch=1024):
    out = torch.empty(len(x), nclass, dtype=torch.float16)
    for i in range(0, len(x), batch):
        out[i:i + batch] = F.softmax(model(x[i:i + batch].to(device)), 1).half().cpu()
    return out


# --------------------------------------------------------------------------- #
# Scoring stacks (S, P, q per dataset x table-arch x ipc), cache-first w/ aliases
# --------------------------------------------------------------------------- #
def score_blob(ds, arch, ipc, device):
    path = os.path.join(SCORE_DIR, f"bench_{ds}_{arch}_ipc{ipc}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    if arch == "conv" and ipc in (10, 50):  # legacy aliases — never recompute
        if ds == "tin":
            return torch.load(os.path.join(SCORE_DIR, f"tin_score_ipc{ipc}.pt"),
                              map_location="cpu")
        cad_p = os.path.join(SCORE_DIR, f"cad_ipc{ipc}.pt")
        probe_p = os.path.join(SCORE_DIR, f"probe_ipc{ipc}.pt")
        if os.path.exists(cad_p) and os.path.exists(probe_p):
            blob = torch.load(cad_p, map_location="cpu")
            probe = torch.load(probe_p, map_location="cpu")
            blob["P"], blob["q"] = probe["P"], probe["q"]
            return blob
    return run_score(ds, arch, ipc, device, path)


def run_score(ds, arch, ipc, device, out_path):
    """Instrumented compute-matched HL run (generalizes tin.run_score): per-epoch
    EL2N S, fp16 softmax trajectories P, teacher soft labels q, in one seed-0 run."""
    os.makedirs(SCORE_DIR, exist_ok=True)
    d = BDS[ds]
    K = compute_K(ds, ipc)
    step_epoch = round(K * 151 / 300)  # compressed schedule [ca2d ASSUME 1]
    t0 = time.time()
    xtr_u8, ytr = load_train(ds)
    teacher = build_teacher(ds, arch, device)
    q = collect_probs(teacher, to_norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD)
                      .contiguous(), d["nclass"], device)
    set_seed(0)  # [B8] scoring-run seed fixed at 0 (repo convention)
    x = to_norm_tensor(xtr_u8, d["mean"], d["std"]).contiguous()
    y = torch.from_numpy(ytr).long()
    y1h = F.one_hot(y, d["nclass"]).float()
    model = build_student(ds, arch)
    S = torch.empty(K, d["ntrain"])
    P = torch.empty(K, d["ntrain"], d["nclass"], dtype=torch.float16)
    bar = tqdm(total=K, desc=f"score {ds}/{arch}/ipc{ipc}", unit="ep",
               dynamic_ncols=True)

    def hook(ep, m):
        S[ep] = el2n_scores(m, x, y1h, device)
        P[ep] = collect_probs(m, x, d["nclass"], device)
        bar.update(1)
        bar.set_postfix(el2n=f"{S[ep].mean():.4f}")

    hl_train(model, x, y, device, epochs=K, step_epoch=step_epoch, aug="dsa",
             epoch_hook=hook)
    bar.close()
    U, cad = cad_from_S(S, K - 2, 2)  # J=K-W, W=2 [ca2d ASSUME 9]
    blob = {"S": S, "U": U, "cad": cad, "P": P, "q": q, "K": K,
            "step_epoch": step_epoch, "seed": 0, "aug": "dsa",
            "secs": time.time() - t0}
    torch.save(blob, out_path)
    print(f"[score] saved {out_path} ({blob['secs']:.0f}s)")
    return blob


def el2n_mean(ds, arch, device):
    """[B3] mean EL2N over 10 independently-seeded truncated-HL runs at epoch 20."""
    d = BDS[ds]
    runs = []
    for r in EL2N_SEEDS:
        p = os.path.join(SCORE_DIR, f"el2n_{ds}_{arch}_e{EL2N_EPOCH}_run{r}.pt")
        if not os.path.exists(p):
            os.makedirs(SCORE_DIR, exist_ok=True)
            t0 = time.time()
            xtr_u8, ytr = load_train(ds)
            set_seed(r)
            x = to_norm_tensor(xtr_u8, d["mean"], d["std"]).contiguous()
            y = torch.from_numpy(ytr).long()
            y1h = F.one_hot(y, d["nclass"]).float()
            model = build_student(ds, arch)
            out = {}
            bar = tqdm(total=EL2N_EPOCH, desc=f"el2n {ds}/{arch} run{r}", unit="ep",
                       dynamic_ncols=True, leave=False)

            def hook(ep, m):
                bar.update(1)
                if ep == EL2N_EPOCH - 1:
                    out["el2n"] = el2n_scores(m, x, y1h, device)

            # StepLR@151 never fires inside 20 epochs -> constant early-training lr
            hl_train(model, x, y, device, epochs=EL2N_EPOCH, step_epoch=151,
                     aug="dsa", epoch_hook=hook)
            bar.close()
            torch.save({"el2n": out["el2n"], "seed": r, "epoch": EL2N_EPOCH,
                        "secs": time.time() - t0}, p)
            print(f"[el2n] saved {p} ({time.time() - t0:.0f}s)")
        runs.append(torch.load(p, map_location="cpu")["el2n"])
    return torch.stack(runs).mean(0)


def a_grad_scores(P, q, y, nclass, chunk=5000):
    """SHARP A_grad = mean_k cos(p_k - onehot(y), p_k - q) (tin.a_grad_scores,
    nclass-parameterized)."""
    N = P.shape[1]
    A = torch.empty(N)
    for i in range(0, N, chunk):
        p = P[:, i:i + chunk].float()
        e_h = p - F.one_hot(y[i:i + chunk], nclass).float()
        e_s = p - q[i:i + chunk].float()
        A[i:i + chunk] = ((e_h * e_s).sum(-1)
                          / (e_h.norm(dim=-1) * e_s.norm(dim=-1) + 1e-12)).mean(0)
    return A


# --------------------------------------------------------------------------- #
# Sets: legacy aliases, selections, synthesis, exports
# --------------------------------------------------------------------------- #
_CIFAR_LEGACY = {"rcad": "rcad_feat_fl_ipc{i}", "sharp": "sharp_fl_grad_a100_ipc{i}",
                 "cadprune": "cadprune_ipc{i}", "ca2d": "ca2d_ipc{i}",
                 "rcad2d": "rcad2d_feat_fl_ipc{i}", "random": "random_ipc{i}"}
_TIN_LEGACY = {"rcad": "tin_rcad_ipc{i}", "sharp": "tin_sharpfl_ipc{i}",
               "cadprune": "tin_cadprune_ipc{i}", "ca2d": "tin_ca2d_ipc{i}",
               "random": "tin_random_ipc{i}"}


def legacy_set(ds, arch, method, ipc):
    """Existing set name under SET_DIR, or None. Random is arch-independent."""
    if ipc not in (10, 50):
        return None
    if arch != "conv" and method != "random":
        return None
    tpl = (_CIFAR_LEGACY if ds == "cifar100" else _TIN_LEGACY).get(method)
    if tpl is None:
        return None
    name = tpl.format(i=ipc)
    if os.path.exists(os.path.join(SET_DIR, name + ".pt")) or \
            os.path.isdir(os.path.join(SET_DIR, name)):
        return name
    return None


def set_key(ds, arch, method, ipc):
    if method == "random":  # selection is arch-independent
        return f"bench_{ds}_random_ipc{ipc}"
    return f"bench_{ds}_{arch}_{method}_ipc{ipc}"


def dir_complete(root, nclass, per_class):
    if not os.path.isdir(root):
        return False
    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    return len(dirs) == nclass and all(
        len(os.listdir(os.path.join(root, d))) == per_class for d in dirs)


def save_blob(name, x_u8, y, keep):
    torch.save({"images": x_u8[keep], "labels": y[keep], "indices": keep},
               os.path.join(SET_DIR, name + ".pt"))
    print(f"[select] saved {name}.pt")


def fl_select(ds, arch, method, blob, xtr_u8, ytr, budget, device, desc):
    """Per-class feat-kernel FL selection shared by the coreset rows (budget=ipc)
    and the 2D pool variants (budget=RDED_MIPC): CAD demand for rcad*, shifted
    A_grad z-score demand for sharp*. Returns per-class index arrays."""
    d = BDS[ds]
    cad = blob["cad"]
    teacher = build_teacher(ds, arch, device)
    feats = teacher_features(
        teacher, to_norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD).contiguous(), device)
    A = None
    if method.startswith("sharp"):
        A = a_grad_scores(blob["P"], blob["q"], torch.from_numpy(ytr).long(),
                          d["nclass"])
    sel = []
    for idx in tqdm(class_indices(ytr, d["nclass"]), desc=desc, unit="cls",
                    leave=False):
        kappa = rcad.cos_gram(feats[idx].float())
        if method.startswith("rcad"):  # headline: feat kernel, FL, CAD demand
            w = cad[idx].float()
        else:                          # sharp: shifted A_grad z-score demand
            s = A[idx]
            s = (s - s.mean()) / (s.std() + 1e-12)
            w = s - s.min() + 1e-6
        sel.append(idx[np.array(rcad.greedy_fl(kappa, w, budget))])
    return sel


def build_set(ds, arch, method, ipc, device):
    """Ensure the selection/synthesis artifact exists; return set name (SET_DIR)."""
    assert method not in ("full", "el2nbest")
    name = legacy_set(ds, arch, method, ipc)
    if name is not None:
        return name
    name = set_key(ds, arch, method, ipc)
    d = BDS[ds]
    if method == "rded":
        return build_rded_set(ds, arch, ipc, device)
    if method in ("ca2d", "sharp2d", "rcad2d"):
        out_dir = os.path.join(SET_DIR, name)
        if dir_complete(out_dir, d["nclass"], ipc):
            return name
        blob = score_blob(ds, arch, ipc, device)
        xtr_u8, ytr = load_train(ds)
        if method == "ca2d":  # [HT Sec.5] per-class top-mipc CAD pool
            cad = blob["cad"]
            pools = [idx[np.argsort(-cad[idx].numpy(), kind="stable")]
                     [:ca2d.RDED_MIPC] for idx in class_indices(ytr, d["nclass"])]
        else:                 # 2D variants: our FL pool, same budget [REPORT Sec.4]
            pools = fl_select(ds, arch, method, blob, xtr_u8, ytr, ca2d.RDED_MIPC,
                              device, desc=f"pool {name}")
        return synth_ca2d(ds, arch, ipc, pools, name, device)
    if os.path.exists(os.path.join(SET_DIR, name + ".pt")):
        return name
    os.makedirs(SET_DIR, exist_ok=True)
    xtr_u8, ytr = load_train(ds)
    if method == "random":  # [B10]
        rng = np.random.RandomState(0)
        keep = np.concatenate([rng.choice(idx, size=ipc, replace=False)
                               for idx in class_indices(ytr, d["nclass"])])
        save_blob(name, xtr_u8, ytr, keep)
        return name
    blob = score_blob(ds, arch, ipc, device)
    if method == "cadprune":  # [HT Sec.5] top-IPC by highest CAD
        cad = blob["cad"]
        keep = [idx[np.argsort(-cad[idx].numpy(), kind="stable")][:ipc]
                for idx in class_indices(ytr, d["nclass"])]
    else:                     # rcad / sharp coresets
        keep = fl_select(ds, arch, method, blob, xtr_u8, ytr, ipc, device,
                         desc=f"select {name}")
    save_blob(name, xtr_u8, ytr, np.concatenate(keep))
    return name


def synth_ca2d(ds, arch, ipc, pools, name, device, seed=0):
    """CA2D [HT Sec.5]: per-class pool (top-mipc CAD, or FL for the 2D variants)
    -> RDED Alg.1 crop/select/stitch (mirrors ca2d.synthesize_ca2d / tin.synth_ca2d,
    arch-parameterized observer)."""
    d = BDS[ds]
    out_dir = os.path.join(SET_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    xtr_u8, _ = load_train(ds)
    observer = build_teacher(ds, arch, device)
    mean = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD).view(1, 3, 1, 1)
    n_patches = ipc * ca2d.RDED_FACTOR ** 2
    for c, pool in enumerate(tqdm(pools, desc=f"synth {name}", unit="cls",
                                  leave=False)):
        imgs = torch.from_numpy(xtr_u8[pool]).float().permute(0, 3, 1, 2) / 255.0
        set_seed(seed * 1000003 + c)
        crops = ca2d.rded_multi_random_crop(imgs, ca2d.RDED_NUM_CROP, d["size"],
                                            ca2d.RDED_FACTOR)
        crops = (crops - mean.unsqueeze(0)) / std.unsqueeze(0)
        labels = torch.full((len(pool),), c, dtype=torch.long)
        picked = ca2d.rded_selector(n_patches, observer, crops, labels, d["size"],
                                    ca2d.RDED_NUM_CROP, device)
        mixed = ca2d.rded_mix_images(picked.cpu(), d["size"], ca2d.RDED_FACTOR, ipc)
        ca2d.rded_save_images((mixed * std + mean).clamp(0, 1), c, out_dir)
    print(f"[synth] saved {out_dir}")
    return name


def build_rded_set(ds, arch, ipc, device):
    """RDED row: OFFICIAL synthesis (synthesize/main.py) via the worker. [B9]"""
    name = set_key(ds, arch, "rded", ipc)
    out_dir = os.path.join(SET_DIR, name)
    d = BDS[ds]
    if dir_complete(out_dir, d["nclass"], ipc):
        return name
    train_dir = os.path.join(RDED_CWD, "data", d["subset"], "train")
    assert os.path.isdir(train_dir), "run `python bench.py prepare` first"
    cfg = dict(mode="synth", subset=d["subset"], arch_name=rded_arch(ds, arch),
               nclass=d["nclass"], mipc=ca2d.RDED_MIPC, ipc=ipc,
               num_crop=ca2d.RDED_NUM_CROP, factor=ca2d.RDED_FACTOR,
               input_size=d["size"], train_dir=train_dir,
               syn_data_path=out_dir, workers=4)
    log = os.path.join(LOG_DIR, f"{name}.log")
    run_worker(cfg, device, log, show_child_tqdm=True)
    assert dir_complete(out_dir, d["nclass"], ipc), \
        f"official synthesis incomplete; see {log}"
    print(f"[rded] saved {out_dir}")
    return name


def raw_set_images(set_name):
    """uint8 images of a set (pt blob or image dir), class-sorted."""
    pt = os.path.join(SET_DIR, set_name + ".pt")
    if os.path.exists(pt):
        return np.asarray(torch.load(pt)["images"])
    root = os.path.join(SET_DIR, set_name)
    imgs = [np.array(Image.open(os.path.join(root, c, f)).convert("RGB"))
            for c in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, c))
            for f in sorted(os.listdir(os.path.join(root, c)))]
    return np.stack(imgs)


def export_imagefolder(set_name):
    """[B5] materialize a .pt coreset as a PNG ImageFolder tree for the KD path."""
    dst = os.path.join(EXPORT_DIR, set_name)
    blob = torch.load(os.path.join(SET_DIR, set_name + ".pt"))
    imgs, labels = np.asarray(blob["images"]), np.asarray(blob["labels"])
    indices = np.asarray(blob["indices"])
    nclass = len(np.unique(labels))
    if dir_complete(dst, nclass, len(labels) // nclass):
        return dst
    for i in tqdm(range(len(labels)), desc=f"export {set_name}", unit="img",
                  leave=False):
        cdir = os.path.join(dst, f"{labels[i]:05d}")
        os.makedirs(cdir, exist_ok=True)
        Image.fromarray(imgs[i]).save(
            os.path.join(cdir, f"idx{int(indices[i]):07d}.png"))
    print(f"[export] {dst}")
    return dst


def kd_syn_path(ds, set_name):
    """ImageFolder root for the KD worker: synthesis dirs as-is, coresets as PNG."""
    root = os.path.join(SET_DIR, set_name)
    if os.path.isdir(root):
        return root
    return export_imagefolder(set_name)


# --------------------------------------------------------------------------- #
# RDED official worker (subprocess; zero edits to the RDED repo)
# --------------------------------------------------------------------------- #
def kd_cfg(ds, arch, ipc, set_name, seed, re_epochs=None):
    """Namespace fields for validation/main.py, replicating argument.py's derived
    values (parity selftest-verified). set_name=None -> Full-Dataset row [B6]."""
    d = BDS[ds]
    name = rded_arch(ds, arch)
    bs, lr = kd_batch_lr(ds, arch, ipc)
    if set_name is None:
        syn = os.path.join(RDED_CWD, "data", d["subset"], "train")
        files, eps = d["pool"], compute_K(ds, ipc)
    else:
        syn = kd_syn_path(ds, set_name)
        files, eps = ipc, 300
    if re_epochs is not None:
        eps = re_epochs
    val_dir = os.path.join(RDED_CWD, "data", d["subset"], "val")
    assert os.path.isdir(val_dir), "run `python bench.py prepare` first"
    bs = min(bs, files * d["nclass"])  # argument.py:256-258
    return dict(mode="kd", subset=d["subset"], arch_name=name, stud_name=name,
                nclass=d["nclass"], ipc=files, input_size=d["size"],
                val_ipc=d["val_ipc"], val_dir=val_dir, syn_data_path=syn,
                re_epochs=eps, re_batch_size=bs, adamw_lr=lr,
                adamw_weight_decay=0.01, sgd=False, learning_rate=0.1,
                momentum=0.9, weight_decay=1e-4, cos=True, mix_type="cutmix",
                mixup=0.8, cutmix=1.0, temperature=20, re_accum_steps=1,
                min_scale_crops=0.08, max_scale_crops=1, factor=1, workers=4,
                seed=seed)


def run_worker(cfg, device, log_path, desc=None, show_child_tqdm=False):
    """Launch `bench.py rded-worker` pinned to one GPU; stream/parse its output.
    Returns (best_acc, best_epoch, secs) for kd mode, (None, None, secs) for synth."""
    os.makedirs(LOG_DIR, exist_ok=True)
    cfg_path = log_path + ".cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_index(device)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "rded-worker", "--cfg", cfg_path],
        stdout=subprocess.PIPE,
        stderr=None if show_child_tqdm else subprocess.STDOUT,
        text=True, bufsize=1, env=env)
    bar = None
    if cfg["mode"] == "kd":
        bar = tqdm(total=cfg["re_epochs"], desc=desc or "kd", unit="ep",
                   dynamic_ncols=True, leave=False)
    best = None
    with open(log_path, "w") as lg:
        for line in proc.stdout:
            lg.write(line)
            if bar is not None:
                m = RE_EPOCH.match(line)
                if m:
                    bar.update(1)
                    bar.set_postfix(loss=float(m.group(2)))
            m = RE_BEST.search(line)
            if m:
                best = (float(m.group(1)), int(m.group(2)))
    rc = proc.wait()
    if bar is not None:
        bar.close()
    if rc != 0:
        raise RuntimeError(f"RDED worker failed (rc={rc}); log: {log_path}")
    if cfg["mode"] == "kd":
        if best is None:
            raise RuntimeError(f"no 'Best accuracy' line; log: {log_path}")
        if best[0] == 0.0:
            raise RuntimeError(
                f"validation never fired (need re_epochs >= 6); log: {log_path}")
        return best[0], best[1], time.time() - t0
    return None, None, time.time() - t0


def cmd_rded_worker(cfg_path, selfcheck=False):
    """Runs inside the subprocess: cwd + import shims, then the OFFICIAL entry."""
    os.makedirs(RDED_CWD, exist_ok=True)
    os.chdir(RDED_CWD)
    sys.path.insert(0, RDED_ROOT)
    import types
    stub = types.ModuleType("argument")   # synthesize/utils.py:3 does
    stub.args = None                      # `from argument import args` at import
    sys.modules["argument"] = stub        # time; the value is never used after.
    if selfcheck:
        from validation import main as vmain          # noqa: F401
        from synthesize import main as smain          # noqa: F401
        assert "argument" in sys.modules and sys.modules["argument"] is stub
        print("WORKER-OK")
        return 0
    with open(cfg_path) as f:
        cfg = json.load(f)
    ns = argparse.Namespace(**{k: v for k, v in cfg.items() if k != "mode"})
    ns.classes = range(ns.nclass)
    if cfg["mode"] == "kd":
        # validation/main.py:45-47 seeds random+torch from args.seed; numpy
        # (cutmix's np.random.beta/randint) is never seeded upstream.
        np.random.seed(ns.seed)
        from validation import main as vmain
        vmain.main(ns)
    else:
        import random as _random                     # [B9] stock leaves the pool
        _random.seed(0)                              # shuffle / crop RNG unseeded
        np.random.seed(0)
        torch.manual_seed(0)
        from synthesize import main as smain
        smain.main(ns)
    return 0


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def result_path(ds, arch, method, regime, ipc):
    return os.path.join(RESULT_DIR,
                        f"bench_{ds}_{arch}_{method}_{regime}_ipc{ipc}.json")


def legacy_result(ds, arch, method, regime, ipc):
    """Path of a protocol-identical legacy result JSON, or None."""
    if arch != "conv" or ipc not in (10, 50) or regime == "kd":
        return None
    if ds == "cifar100":
        tpl = _CIFAR_LEGACY.get(method)
        if tpl is None:
            return None
        name = tpl.format(i=ipc)
        p = (os.path.join(RESULT_DIR, f"{name}.json") if regime == "hl" else
             os.path.join(RESULT_DIR, f"xarch_cifar100_{name}_convnet_sl.json"))
    else:
        tpl = _TIN_LEGACY.get(method)
        if tpl is None:
            return None
        p = os.path.join(RESULT_DIR,
                         f"xarch_tin_{tpl.format(i=ipc)}_convnet_{regime}.json")
    return p if os.path.exists(p) else None


def load_result(ds, arch, method, regime, ipc, seeds=None):
    for p in (result_path(ds, arch, method, regime, ipc),
              legacy_result(ds, arch, method, regime, ipc)):
        if p and os.path.exists(p):
            with open(p) as f:
                res = json.load(f)
            if seeds is None or res.get("seeds") == list(seeds):
                res["path"] = os.path.basename(p)
                return res
    return None


def hl_sl_setup(ds, arch, set_name, regime, ipc, device):
    """Load data/labels once per cell. set_name=None -> Full-Dataset row [B6]."""
    d = BDS[ds]
    if set_name is None:
        xtr_u8, ytr = load_train(ds)
        x = to_norm_tensor(xtr_u8, d["mean"], d["std"]).contiguous()
        y = torch.from_numpy(ytr).long()
        raw = xtr_u8
        epochs = compute_K(ds, ipc)
    else:
        x, y = xarch.load_set(set_name, ds)
        assert len(x) == ipc * d["nclass"], (set_name, len(x))
        raw = raw_set_images(set_name)
        epochs = 300
    q_T = None
    if regime == "sl":
        teacher = build_teacher(ds, arch, device)
        xt = to_norm_tensor(raw, IMNET_MEAN, IMNET_STD).contiguous()
        out = []
        with torch.no_grad():
            for i in range(0, len(xt), 1024):
                out.append(F.softmax(teacher(xt[i:i + 1024].to(device)) / KD_T,
                                     dim=1).cpu())
        q_T = torch.cat(out)
    xte, yte = xarch.load_test(ds)
    return dict(x=x, y=y, q_T=q_T, xte=xte, yte=yte, epochs=epochs,
                step=round(epochs * 151 / 300))


def hl_sl_seed_run(st, ds, arch, regime, seed, device, desc, batch=256):
    t0 = time.time()
    set_seed(seed)
    model = build_student(ds, arch)
    bar = tqdm(total=st["epochs"], desc=desc, unit="ep", dynamic_ncols=True,
               leave=False)
    if regime == "hl":
        hl_train(model, st["x"], st["y"], device, epochs=st["epochs"],
                 step_epoch=st["step"], aug="dsa", batch=batch,
                 epoch_hook=bar_hook(bar))
    else:
        sl_train(model, st["x"], st["q_T"], device, epochs=st["epochs"],
                 batch=batch, epoch_hook=bar_hook(bar))
    bar.close()
    acc = test_top1(model, st["xte"], st["yte"], device)
    return acc, time.time() - t0


def train_once(ds, arch, set_name, regime, ipc, seed, device, re_epochs=None,
               desc=None):
    """One seed of one cell (any regime). set_name=None -> full dataset."""
    desc = desc or f"{ds}/{arch}/{regime}/ipc{ipc} s{seed}"
    if regime == "kd":
        cfg = kd_cfg(ds, arch, ipc, set_name, seed, re_epochs)
        log = os.path.join(
            LOG_DIR, f"kd_{ds}_{arch}_{set_name or 'full'}_ipc{ipc}_s{seed}.log")
        acc, _, secs = run_worker(cfg, device, log, desc=desc)
        return acc, secs
    st = hl_sl_setup(ds, arch, set_name, regime, ipc, device)
    return hl_sl_seed_run(st, ds, arch, regime, seed, device, desc)


def eval_cell(ds, arch, method, regime, ipc, seeds, device, smoke_epochs=None):
    if method == "el2nbest":
        return eval_el2nbest(ds, arch, regime, ipc, seeds, device)
    if smoke_epochs is None:
        res = load_result(ds, arch, method, regime, ipc, seeds)
        if res is not None:
            print(f"[cell] cached ({res['path']}): "
                  f"{res['mean']:.2f} +- {res['std']:.2f}")
            return res
    t_all = time.time()
    set_name = None if method == "full" else build_set(ds, arch, method, ipc, device)
    cell = f"{ds}/{arch}/{method}/{regime}/ipc{ipc}"
    accs, secs = [], []
    if regime == "kd":
        for s in seeds:
            acc, sec = train_once(ds, arch, set_name, "kd", ipc, s, device,
                                  re_epochs=smoke_epochs, desc=f"{cell} s{s}")
            accs.append(acc)
            secs.append(sec)
            print(f"[cell] {cell} seed{s}: {acc:.2f}%  ({sec:.0f}s)", flush=True)
    else:
        st = hl_sl_setup(ds, arch, set_name, regime, ipc, device)
        for s in seeds:
            acc, sec = hl_sl_seed_run(st, ds, arch, regime, s, device,
                                      f"{cell} s{s}")
            accs.append(acc)
            secs.append(sec)
            print(f"[cell] {cell} seed{s}: {acc:.2f}%  ({sec:.0f}s)", flush=True)
    res = {"dataset": ds, "arch": arch, "method": method, "regime": regime,
           "ipc": ipc, "set": set_name, "seeds": list(seeds), "accs": accs,
           "mean": float(np.mean(accs)), "std": float(np.std(accs)),
           "seed_secs": [round(t, 1) for t in secs],
           "total_secs": round(time.time() - t_all, 1)}
    if smoke_epochs is not None:
        print(f"[cell] SMOKE ({smoke_epochs} ep, NOT persisted): "
              f"{res['mean']:.2f} +- {res['std']:.2f}")
        return res
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(result_path(ds, arch, method, regime, ipc), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[cell] {cell}: {res['mean']:.2f} +- {res['std']:.2f} "
          f"({res['total_secs']:.0f}s)")
    return res


def build_el2n_window(ds, arch, ipc, off, device):
    """[B4] per-class descending-EL2N window starting at off% of (pool - ipc)."""
    name = f"bench_{ds}_{arch}_el2nw{off}_ipc{ipc}"
    if os.path.exists(os.path.join(SET_DIR, name + ".pt")):
        return name
    d = BDS[ds]
    scores = el2n_mean(ds, arch, device).numpy()
    xtr_u8, ytr = load_train(ds)
    keep = []
    for idx in class_indices(ytr, d["nclass"]):
        order = idx[np.argsort(-scores[idx], kind="stable")]
        start = int(round(off / 100 * (len(idx) - ipc)))
        keep.append(order[start:start + ipc])
    save_blob(name, xtr_u8, ytr, np.concatenate(keep))
    return name


def eval_el2nbest(ds, arch, regime, ipc, seeds, device):
    res = load_result(ds, arch, "el2nbest", regime, ipc, seeds)
    if res is not None:
        print(f"[cell] cached ({res['path']}): "
              f"{res['mean']:.2f} +- {res['std']:.2f}")
        return res
    t_all = time.time()
    cell = f"{ds}/{arch}/el2nbest/{regime}/ipc{ipc}"
    search_p = os.path.join(RESULT_DIR,
                            f"bench_{ds}_{arch}_el2nsearch_{regime}_ipc{ipc}.json")
    search = {}
    if os.path.exists(search_p):
        with open(search_p) as f:
            search = json.load(f)
    for off in EL2N_OFFSETS:  # [B4] seed-42 search, resumable per window
        if str(off) in search:
            continue
        name = build_el2n_window(ds, arch, ipc, off, device)
        acc, sec = train_once(ds, arch, name, regime, ipc, seeds[0], device,
                              desc=f"{cell} w{off} s{seeds[0]}")
        search[str(off)] = {"acc": acc, "secs": round(sec, 1)}
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(search_p, "w") as f:
            json.dump(search, f, indent=2)
        print(f"[cell] {cell} window {off}%: {acc:.2f}%  ({sec:.0f}s)", flush=True)
    best_off = max(search, key=lambda k: search[k]["acc"])
    name = build_el2n_window(ds, arch, ipc, int(best_off), device)
    accs = [search[best_off]["acc"]]           # seed-42 search result reused
    secs = [search[best_off]["secs"]]
    for s in seeds[1:]:
        acc, sec = train_once(ds, arch, name, regime, ipc, s, device,
                              desc=f"{cell} s{s}")
        accs.append(acc)
        secs.append(round(sec, 1))
        print(f"[cell] {cell} seed{s}: {acc:.2f}%  ({sec:.0f}s)", flush=True)
    res = {"dataset": ds, "arch": arch, "method": "el2nbest", "regime": regime,
           "ipc": ipc, "set": name, "window_offset": int(best_off),
           "search": search, "seeds": list(seeds), "accs": accs,
           "mean": float(np.mean(accs)), "std": float(np.std(accs)),
           "seed_secs": secs, "total_secs": round(time.time() - t_all, 1)}
    with open(result_path(ds, arch, "el2nbest", regime, ipc), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[cell] {cell}: {res['mean']:.2f} +- {res['std']:.2f} "
          f"(window {best_off}%)")
    return res


# --------------------------------------------------------------------------- #
# prepare — RDED-layout data trees under artifacts/rded
# (5-digit ImageFolder layout per RDED/prepare/*.md; CIFAR as lossless PNG —
#  teachers.py:227 measured 1.6-3.1pp teacher-top-1 loss under JPEG — and
#  TinyImageNet as symlinks to the original JPEGs.)
# --------------------------------------------------------------------------- #
def _prep_cifar_tree(data):
    out = os.path.join(data, "cifar100")
    xtr, ytr, xte, yte = ca2d.load_cifar100()
    for split, (x, y), per in (("train", (xtr, ytr), 500), ("val", (xte, yte), 100)):
        root = os.path.join(out, split)
        if dir_complete(root, 100, per):
            print(f"[prepare] {root} already complete")
            continue
        counts = [0] * 100
        for img, lab in zip(tqdm(x, desc=f"cifar100 {split}", unit="img",
                                 leave=False), y):
            cdir = os.path.join(root, f"{lab:05d}")
            os.makedirs(cdir, exist_ok=True)
            Image.fromarray(img).save(
                os.path.join(cdir, f"img_{counts[lab]:05d}.png"), optimize=False)
            counts[lab] += 1
        print(f"[prepare] wrote {sum(counts)} PNGs -> {root}")


def _prep_tin_tree(data):
    out = os.path.join(data, "tinyimagenet")
    assert os.path.isdir(tin.TIN_DIR), f"{tin.TIN_DIR} missing"
    wnids = tin.tin_wnids()  # alphabetical == RDED index order; verify vs the doc
    if os.path.exists(tin.RDED_MAPPING_DOC):
        with open(tin.RDED_MAPPING_DOC) as f:
            pairs = [ln.strip().split(": ") for ln in f
                     if ln.strip()[:1].isdigit() and ": n" in ln]
        assert len(pairs) == 200
        for k, w in pairs:
            assert wnids[int(k)] == w, (k, w, wnids[int(k)])
    troot = os.path.join(out, "train")
    if not dir_complete(troot, 200, 500):
        for ci, w in enumerate(tqdm(wnids, desc="tinyimagenet train", unit="cls",
                                    leave=False)):
            src = os.path.join(tin.TIN_DIR, "train", w, "images")
            dst = os.path.join(troot, f"{ci:05d}")
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(src):
                lp = os.path.join(dst, fn)
                if not os.path.exists(lp):
                    os.symlink(os.path.abspath(os.path.join(src, fn)), lp)
        print(f"[prepare] linked train -> {troot}")
    vroot = os.path.join(out, "val")
    if not dir_complete(vroot, 200, 50):
        cls = {w: i for i, w in enumerate(wnids)}
        with open(os.path.join(tin.TIN_DIR, "val", "val_annotations.txt")) as fh:
            ann = dict(line.split("\t")[:2] for line in fh)
        for fn, w in tqdm(ann.items(), desc="tinyimagenet val", unit="img",
                          leave=False):
            dst = os.path.join(vroot, f"{cls[w]:05d}")
            os.makedirs(dst, exist_ok=True)
            lp = os.path.join(dst, fn)
            if not os.path.exists(lp):
                os.symlink(os.path.abspath(
                    os.path.join(tin.TIN_DIR, "val", "images", fn)), lp)
        print(f"[prepare] linked val -> {vroot}")


def cmd_prepare():
    for p in (RDED_CWD, EXPORT_DIR, LOG_DIR, SCORE_DIR, SET_DIR, RESULT_DIR):
        os.makedirs(p, exist_ok=True)
    data = os.path.join(RDED_CWD, "data")
    os.makedirs(data, exist_ok=True)
    link = os.path.join(data, "pretrain_models")
    if not os.path.exists(link):
        os.symlink(os.path.join(ART, "pretrain_models"), link)
        print(f"[prepare] {link} -> {os.path.join(ART, 'pretrain_models')}")
    _prep_cifar_tree(data)
    _prep_tin_tree(data)
    for ds in BDS.values():
        for split, per in (("train", ds["pool"]), ("val", ds["val_ipc"])):
            root = os.path.join(data, ds["subset"], split)
            assert dir_complete(root, ds["nclass"], per), \
                f"{root}: expected {ds['nclass']} x {per}"
    print("[prepare] done")


# --------------------------------------------------------------------------- #
# run / table / tex / timing
# --------------------------------------------------------------------------- #
def cmd_run(table_key, seeds, device, ipcs, regimes, methods):
    ds, arch = TABLES[table_key]
    cells = [(m, r, i) for i in ipcs for r in regimes for m in methods]
    outer = tqdm(cells, desc=table_key, unit="cell", dynamic_ncols=True)
    for m, r, i in outer:
        outer.set_postfix_str(f"{m}/{r}/ipc{i}")
        eval_cell(ds, arch, m, r, i, seeds, device)


COLS9 = [(r, i) for r in REGIMES for i in IPCS]
LABELS = {"random": "Random Real", "el2nbest": "EL2N-Best",
          "cadprune": "CAD-Prune", "sharp": "SHARP (ours)",
          "rcad": "R-CAD (ours)", "rded": "RDED", "ca2d": "CA2D",
          "sharp2d": "SHARP-2D (ours)", "rcad2d": "R-CAD-2D (ours)"}
BLOCKS = (("Coreset Selection", ("random", "el2nbest", "cadprune", "sharp", "rcad")),
          ("Dataset Distillation", ("rded", "ca2d", "sharp2d", "rcad2d")))
TABLE_METHODS = tuple(m for _, ms in BLOCKS for m in ms)


def _cell_res(ds, arch, method, regime, ipc):
    r = load_result(ds, arch, method, regime, ipc)
    return None if r is None else (r["mean"], r["std"], r["accs"])


def render_table(table_key):
    ds, arch = TABLES[table_key]
    B, E = "\033[1m", "\033[0m"
    CW = 15
    width = 24 + 9 * CW + 2
    stud = {"conv": f"ConvNet-D{3 if ds == 'cifar100' else 4}",
            "rn18": "ResNet-18"}[arch]
    print("\n" + "=" * width)
    print(f"  {table_key}  ({stud}; HL/SL: DC-bench student, KD: RDED student "
          f"[B7]; seeds 42-44)")
    print("=" * width)
    head1 = f"{'':<24}" + "".join(
        f"{t:^{3 * CW}}" for t in ("Hard Label (HL)", "Fixed Soft Label (SL)",
                                   "KD + Soft Label"))
    head2 = f"{'Method':<24}" + "".join(f"{f'IPC {i}':^{CW}}" for _, i in COLS9)
    print(head1 + "\n" + head2)
    local = {m: {c: _cell_res(ds, arch, m, c[0], c[1]) for c in COLS9}
             for m in TABLE_METHODS}
    ref = {}  # strongest non-ours local row per column (for daggers on ours)
    for c in COLS9:
        base = {m: v[c][0] for m in ("random", "el2nbest", "cadprune", "rded",
                                     "ca2d")
                for v in [local[m]] if v[c] is not None}
        if base:
            ref[c] = local[max(base, key=base.get)][c][2]
    for btitle, ms in BLOCKS:
        print(f"{' ' + btitle + ' ':-^{width}}")
        best = {c: max((local[m][c][0] for m in ms if local[m][c] is not None),
                       default=None) for c in COLS9}
        for m in ms:
            line = f"{LABELS[m]:<24}"
            for c in COLS9:
                v = local[m][c]
                if v is None:
                    line += f"{'--':^{CW}}"
                    continue
                mean, std, accs = v
                mark = ""
                if m in OURS and c in ref:
                    p = welch_p(accs, ref[c])
                    mark = "‡" if p < 0.01 else ("†" if p < 0.05 else "")
                cell = f"{f'{mean:.2f}±{std:.2f}{mark}':^{CW}}"
                if best[c] is not None and abs(mean - best[c]) < 1e-9:
                    cell = B + cell + E
                line += cell
            print(line)
    print("-" * width)
    print("bold = best per block & column; †/‡ = one-sided Welch p<0.05/0.01 of "
          "ours vs the strongest\nbaseline in the column (n=3)")


def tex_table(table_key):
    ds, arch = TABLES[table_key]
    stud = {"conv": f"ConvNet-D{3 if ds == 'cifar100' else 4}",
            "rn18": "ResNet-18"}[arch]
    local = {m: {c: _cell_res(ds, arch, m, c[0], c[1]) for c in COLS9}
             for m in TABLE_METHODS}
    done = sum(v is not None for row in local.values() for v in row.values())
    ref, best = {}, {}
    for c in COLS9:
        means = {m: v[c][0] for m, v in local.items() if v[c] is not None}
        if means:
            best[c] = max(means.values())
        base = {m: v for m, v in means.items() if m not in OURS}
        if base:
            ref[c] = local[max(base, key=base.get)][c][2]

    def cell(m, c):
        v = local[m][c]
        if v is None:
            return "--"
        mean, std, accs = v
        star = ""
        if m in OURS and c in ref:
            p = welch_p(accs, ref[c])
            star = "$^{\\ddagger}$" if p < 0.01 else \
                ("$^{\\dagger}$" if p < 0.05 else "")
        txt = f"{mean:.2f}{{\\scriptsize$\\pm${std:.2f}}}{star}"
        return f"\\textbf{{{txt}}}" if abs(mean - best.get(c, -1)) < 1e-9 else txt

    lines = [
        f"% auto-generated by `python bench.py tex` "
        f"({time.strftime('%Y-%m-%d %H:%M')}); {done}/{len(TABLE_METHODS) * 9} "
        "cells. '--' = pending.",
        "\\begin{table*}[t]", "\\centering",
        f"\\caption{{{ds.upper() if ds == 'tin' else 'CIFAR-100'}, {stud}, "
        "IPC 10/50/100 under hard-label (HL), fixed soft-label (SL) and RDED-style "
        "KD+SL training. HL/SL follow the small-scale protocol of Dey et al.; "
        "KD+SL is RDED's official validation (RDED student variant). $\\dagger$/"
        "$\\ddagger$: one-sided Welch $p<0.05$/$0.01$ vs the strongest baseline "
        "in the column (3 seeds).}",
        f"\\label{{tab:bench_{table_key.replace('-', '_')}}}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l ccc ccc ccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Hard Label (HL)} & \\multicolumn{3}{c}{Fixed Soft "
        "Label (SL)} & \\multicolumn{3}{c}{KD + Soft Label} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "Method & " + " & ".join(f"IPC {i}" for _, i in COLS9) + " \\\\",
    ]
    for btitle, ms in BLOCKS:
        lines.append("\\midrule")
        for m in ms:
            pre = "\\textbf{" + LABELS[m] + "}" if m in OURS else LABELS[m]
            lines.append(f"{pre} & " + " & ".join(cell(m, c) for c in COLS9)
                         + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, f"bench_{table_key}.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex] {path}: {done}/{len(TABLE_METHODS) * 9} cells")


def fmt_dur(secs):
    """60 -> '1.0m', 5400 -> '1h30m'; sub-minute stays in seconds."""
    if round(secs) < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    return f"{int(secs // 3600)}h{int(secs % 3600 // 60):02d}m"


def cmd_timing():
    print(f"\n{'cell':<58}{'mean/seed':>12}{'total':>10}")
    for f in sorted(os.listdir(RESULT_DIR)):
        if not (f.startswith("bench_") and f.endswith(".json")):
            continue
        with open(os.path.join(RESULT_DIR, f)) as fh:
            r = json.load(fh)
        if "seed_secs" in r:
            print(f"{f[6:-5]:<58}{fmt_dur(np.mean(r['seed_secs'])):>12}"
                  f"{fmt_dur(r['total_secs']):>10}")


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
OFFICIAL_ARGVS = {  # the five official cifar100/tinyimagenet scripts
    ("cifar100", "conv", 10): "--subset cifar100 --arch-name conv3 --stud-name conv3",
    ("cifar100", "rn18", 10): "--subset cifar100 --arch-name resnet18_modified "
                              "--stud-name resnet18_modified",
    ("tin", "conv", 10): "--subset tinyimagenet --arch-name conv4 --stud-name conv4",
    ("tin", "rn18", 10): "--subset tinyimagenet --arch-name resnet18_modified "
                         "--stud-name resnet18_modified",
    ("tin", "rn18", 50): "--subset tinyimagenet --arch-name resnet18_modified "
                         "--stud-name resnet18_modified",
}


def selftest(device):
    print("[selftest] 1. compute-matched epochs (both datasets): 6/30/60")
    for ds in BDS:
        assert [compute_K(ds, i) for i in IPCS] == [6, 30, 60], ds
    print("           ok")

    print("[selftest] 2. KD batch/lr rules vs argument.py (official script configs)")
    os.makedirs(RDED_CWD, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = RDED_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for (ds, arch, ipc), extra in OFFICIAL_ARGVS.items():
        argv = (extra + f" --factor 1 --num-crop 5 --mipc 300 --ipc {ipc} "
                "--re-epochs 300").split()
        code = ("import sys, json; sys.argv=['x']+%r; import argument; "
                "a=argument.args; print(json.dumps(dict(bs=a.re_batch_size, "
                "lr=a.adamw_lr, T=a.temperature, size=a.input_size, "
                "vipc=a.val_ipc, nc=a.nclass)))" % argv)
        r = subprocess.run([sys.executable, "-c", code], cwd=RDED_CWD, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        got = json.loads(r.stdout.strip().splitlines()[-1])
        bs, lr = kd_batch_lr(ds, arch, ipc)
        d = BDS[ds]
        want = dict(bs=bs, lr=lr, T=20, size=d["size"], vipc=d["val_ipc"],
                    nc=d["nclass"])
        assert got == want, (ds, arch, ipc, got, want)
        print(f"           {ds}/{rded_arch(ds, arch)}/ipc{ipc}: {got} ok")

    print("[selftest] 3. worker import shim (subprocess)")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "rded-worker",
                        "--selfcheck"], capture_output=True, text=True)
    assert r.returncode == 0 and "WORKER-OK" in r.stdout, (r.stdout, r.stderr)
    print("           ok")

    print("[selftest] 4. PNG exporter roundtrip")
    name = "bench_selftest_tmp"
    rng = np.random.RandomState(0)
    imgs = rng.randint(0, 256, size=(6, 32, 32, 3), dtype=np.uint8)
    labels = np.array([0, 0, 0, 1, 1, 1])
    torch.save({"images": imgs, "labels": labels,
                "indices": np.arange(6)}, os.path.join(SET_DIR, name + ".pt"))
    dst = export_imagefolder(name)
    back = raw_set_images(name)  # from the .pt (ground truth)
    disk = np.stack([np.array(Image.open(os.path.join(dst, c, f)))
                     for c in sorted(os.listdir(dst))
                     for f in sorted(os.listdir(os.path.join(dst, c)))])
    assert np.array_equal(np.sort(back.reshape(6, -1), 0),
                          np.sort(disk.reshape(6, -1), 0))
    import shutil
    os.remove(os.path.join(SET_DIR, name + ".pt"))
    shutil.rmtree(dst)
    print("           ok (lossless)")

    print("[selftest] 5. legacy alias census (existing caches only)")
    n_res = n_set = 0
    for (ds, arch) in (("cifar100", "conv"), ("tin", "conv")):
        for m in ("random", "cadprune", "sharp", "rcad", "ca2d", "rcad2d"):
            for ipc in (10, 50):
                n_set += legacy_set(ds, arch, m, ipc) is not None
                for reg in ("hl", "sl"):
                    n_res += legacy_result(ds, arch, m, reg, ipc) is not None
    print(f"           {n_set}/24 legacy sets, {n_res}/48 legacy result cells")

    print("[selftest] 6. teacher builders + feature dims (released checkpoints)")
    for ds in BDS:
        for arch in ("conv", "rn18"):
            ckpt = os.path.join(ART, "pretrain_models",
                                f"{BDS[ds]['subset']}_{rded_arch(ds, arch)}.pth")
            if not os.path.exists(ckpt):
                print(f"           {ds}/{arch}: checkpoint missing -> skipped")
                continue
            t = build_teacher(ds, arch, device)
            x = torch.randn(2, 3, BDS[ds]["size"], BDS[ds]["size"])
            assert t(x.to(device)).shape == (2, BDS[ds]["nclass"])
            fdim = teacher_features(t, x, device).shape[1]
            assert fdim == (512 if arch == "rn18" else 128 * 4 * 4), fdim
            print(f"           {ds}/{rded_arch(ds, arch)}: forward + {fdim}-d "
                  "features ok")

    print("[selftest] 7. selector determinism (greedy FL twice on synthetic)")
    torch.manual_seed(0)
    E = torch.randn(50, 8)
    kappa = rcad.cos_gram(E)
    w = torch.rand(50)
    assert rcad.greedy_fl(kappa, w, 10) == rcad.greedy_fl(kappa, w, 10)
    print("           ok")

    print("[selftest] 8. EL2N window arithmetic on toy scores")
    scores = np.arange(500, dtype=np.float64)  # descending order = 499, 498, ...
    order = np.argsort(-scores, kind="stable")
    for ipc in IPCS:
        for off in EL2N_OFFSETS:
            start = int(round(off / 100 * (500 - ipc)))
            win = order[start:start + ipc]
            assert len(win) == ipc and win[0] == 499 - start
            assert off > 0 or win[0] == 499  # offset 0 == EL2N-Hard (top scores)
    print("           ok")
    print("[selftest] all checks passed")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "prepare", "score", "el2n",
                                    "select", "export", "cell", "run", "table",
                                    "tex", "timing", "rded-worker"])
    ap.add_argument("--ds", choices=list(BDS))
    ap.add_argument("--arch", choices=["conv", "rn18"])
    ap.add_argument("--method", choices=list(METHODS))
    ap.add_argument("--regime", choices=list(REGIMES))
    ap.add_argument("--ipc", type=int, choices=list(IPCS))
    ap.add_argument("--table", choices=list(TABLES))
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")],
                    default=[42, 43, 44])
    ap.add_argument("--ipcs", type=lambda s: [int(v) for v in s.split(",")],
                    default=list(IPCS))
    ap.add_argument("--regimes", type=lambda s: s.split(","),
                    default=list(REGIMES))
    ap.add_argument("--methods", type=lambda s: s.split(","),
                    default=[m for m in METHODS if m != "full"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--smoke-epochs", type=int, default=None,
                    help="cell only: reduced KD epochs; result NOT persisted")
    ap.add_argument("--cfg", help="rded-worker only: cfg json path")
    ap.add_argument("--selfcheck", action="store_true",
                    help="rded-worker only: import shim check")
    args = ap.parse_args()

    if args.cmd == "rded-worker":
        return cmd_rded_worker(args.cfg, selfcheck=args.selfcheck)

    torch.backends.cudnn.deterministic = True   # repo convention (non-KD paths;
    torch.backends.cudnn.benchmark = False      # the worker keeps RDED's own flags)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.cmd == "selftest":
        selftest(device)
    elif args.cmd == "prepare":
        cmd_prepare()
    elif args.cmd == "score":
        assert args.ds and args.arch and args.ipc
        score_blob(args.ds, args.arch, args.ipc, device)
    elif args.cmd == "el2n":
        assert args.ds and args.arch
        el2n_mean(args.ds, args.arch, device)
    elif args.cmd == "select":
        assert args.ds and args.arch and args.method and args.ipc
        print(build_set(args.ds, args.arch, args.method, args.ipc, device))
    elif args.cmd == "export":
        assert args.ds and args.arch and args.method and args.ipc
        name = build_set(args.ds, args.arch, args.method, args.ipc, device)
        print(kd_syn_path(args.ds, name))
    elif args.cmd == "cell":
        assert args.ds and args.arch and args.method and args.regime and args.ipc
        eval_cell(args.ds, args.arch, args.method, args.regime, args.ipc,
                  args.seeds, device, smoke_epochs=args.smoke_epochs)
    elif args.cmd == "run":
        assert args.table, "--table required"
        cmd_run(args.table, args.seeds, device, args.ipcs, args.regimes,
                args.methods)
    elif args.cmd == "table":
        for t in ([args.table] if args.table else list(TABLES)):
            render_table(t)
    elif args.cmd == "tex":
        for t in ([args.table] if args.table else list(TABLES)):
            tex_table(t)
    elif args.cmd == "timing":
        cmd_timing()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
