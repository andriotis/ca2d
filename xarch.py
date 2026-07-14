#!/usr/bin/env python
"""Universal evaluator: dataset x architecture x label-regime for the CVPR tables.

Covers every comparison cell (set, dataset, arch, regime, seeds):
  datasets  cifar100 (32px, 100c) | tin (64px, 200c)
  archs     convnet  ConvNet-D3/D4 instance-norm (main-table protocol [HT Tab.2/6])
            mlp      DC-bench networks/mlp.py: flatten -> 128 -> 128 -> C, ReLU
            rn18     DC-bench networks/resnet.py CIFAR variant: 3x3 stem stride 1,
                     no maxpool, BasicBlock [2,2,2,2], norm='instancenorm'
                     (GroupNorm(C,C)) default [ASSUME X3: DC-bench default norm]
            rn152    same, Bottleneck [3,8,36,3]
            (final pooling: adaptive avg-pool to 1x1 == DC-bench's fixed 4x4 pool at
             32px, and generalizes it to 64px inputs)
  regimes   hl  [HT Tab.4 small-scale HL]: CE, SGD 1e-2, StepLR@151, batch 256,
                DSA, 300 epochs (ca2d.hl_train verbatim)
            sl  [HT Tab.4 small-scale SL]: one fixed teacher-assigned soft label per
                image [HT Sec.3.2], KL-Div T=20 in the standard KD form (xT^2)
                [ASSUME X1], AdamW 1e-3 (default wd), cosine schedule, 300 epochs,
                batch 256, DSA. Teacher = per-dataset RDED observer [ASSUME X2].

Every result JSON records per-seed wall-clock (`seed_secs`) and `total_secs`.
`avg` prints the DCBench-trio average [HT Tab.7/8] with one-sided Welch p-values
(ours vs strongest local baseline, n=3 seeds - low power, reported honestly).

Usage:
    python xarch.py selftest
    python xarch.py eval  --dataset cifar100 --set rcad_feat_fl_ipc10 --arch convnet --regime sl
    python xarch.py grid  --dataset tin --ipc 10 --archs convnet --regime hl
    python xarch.py table   # paper-style console tables (Tab.2 / Tab.6 layout)
    python xarch.py tex     # regenerate tex/tab2_*.tex and tex/tab6_*.tex
    python xarch.py timing
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats

import ca2d
import tin
from ca2d import (CIFAR_MEAN, CIFAR_STD, EVAL_BATCH, EVAL_EPOCHS, EVAL_STEP_EPOCH,
                  IMNET_MEAN, IMNET_STD, RESULT_DIR, SET_DIR, ParamDiffAug,
                  diff_augment, hl_train, set_seed, test_top1, to_norm_tensor)

DATASETS = {
    "cifar100": dict(classes=100, size=32, mean=CIFAR_MEAN, std=CIFAR_STD),
    "tin": dict(classes=200, size=64, mean=tin.TIN_MEAN, std=tin.TIN_STD),
}
DEFAULT_SETS = {  # the six comparison sets per dataset ({ipc} filled in)
    "cifar100": ["rcad_feat_fl_ipc{ipc}", "sharp_fl_grad_a100_ipc{ipc}",
                 "cadprune_ipc{ipc}", "ca2d_ipc{ipc}", "rcad_feat_fl_uni_ipc{ipc}",
                 "random_ipc{ipc}"],
    "tin": ["tin_rcad_ipc{ipc}", "tin_sharpfl_ipc{ipc}", "tin_cadprune_ipc{ipc}",
            "tin_ca2d_ipc{ipc}", "tin_fluni_ipc{ipc}", "tin_random_ipc{ipc}"],
}
OURS = ("rcad_feat_fl_ipc", "sharp_fl_grad", "tin_rcad_", "tin_sharpfl_")
KD_T = 20.0  # [HT Tab.4: KL-Div (T=20)]


# --------------------------------------------------------------------------- #
# Architectures  [DC-bench networks/, fetched 2026-07-08]
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, channel, num_classes, im_size):
        super().__init__()
        self.fc_1 = nn.Linear(im_size[0] * im_size[1] * channel, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return self.fc_3(x)


def _norm2d(planes, norm):
    if norm == "instancenorm":
        return nn.GroupNorm(planes, planes, affine=True)
    return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = _norm2d(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = _norm2d(planes, norm)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                _norm2d(planes * self.expansion, norm))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = _norm2d(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = _norm2d(planes, norm)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = _norm2d(planes * self.expansion, norm)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                _norm2d(planes * self.expansion, norm))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """DC-bench CIFAR-variant ResNet (3x3 stem, no maxpool)."""

    def __init__(self, block, num_blocks, channel, num_classes, norm="instancenorm"):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(channel, 64, 3, 1, 1, bias=False)
        self.bn1 = _norm2d(64, norm)
        layers = []
        for planes, n, stride in zip((64, 128, 256, 512), num_blocks, (1, 2, 2, 2)):
            strides = [stride] + [1] * (n - 1)
            for s in strides:
                layers.append(block(self.in_planes, planes, s, norm))
                self.in_planes = planes * block.expansion
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.classifier(out)


def build_model(arch, dataset):
    ds = DATASETS[dataset]
    C, s = ds["classes"], ds["size"]
    if arch == "convnet":
        return ca2d.convnet_d3("instance") if dataset == "cifar100" \
            else tin.tin_convnet("instance")
    if arch == "mlp":
        return MLP(3, C, (s, s))
    if arch == "rn18":
        return ResNet(BasicBlock, (2, 2, 2, 2), 3, C)
    if arch == "rn152":
        return ResNet(Bottleneck, (3, 8, 36, 3), 3, C)
    raise ValueError(arch)


# --------------------------------------------------------------------------- #
# Data / sets
# --------------------------------------------------------------------------- #
def load_set(name, dataset):
    ds = DATASETS[dataset]
    pt = os.path.join(SET_DIR, name + ".pt")
    if os.path.exists(pt):
        b = torch.load(pt)
        x = to_norm_tensor(b["images"], ds["mean"], ds["std"]).contiguous()
        return x, torch.from_numpy(np.asarray(b["labels"])).long()
    root = os.path.join(SET_DIR, name)
    assert os.path.isdir(root), f"no set named {name}"
    imgs, labels = [], []
    for cdir in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, cdir)):
            continue
        for f in sorted(os.listdir(os.path.join(root, cdir))):
            imgs.append(np.array(Image.open(os.path.join(root, cdir, f)).convert("RGB")))
            labels.append(int(cdir))
    x = to_norm_tensor(np.stack(imgs), ds["mean"], ds["std"]).contiguous()
    return x, torch.tensor(labels, dtype=torch.long)


def load_test(dataset):
    ds = DATASETS[dataset]
    if dataset == "cifar100":
        _, _, xte, yte = ca2d.load_cifar100()
    else:
        _, _, xte, yte, _ = tin.load_tin()
    return (to_norm_tensor(xte, ds["mean"], ds["std"]).contiguous(),
            torch.from_numpy(yte))


def get_teacher(dataset, device):
    return ca2d.get_observer(device, allow_train_fallback=False) \
        if dataset == "cifar100" else tin.get_tin_observer(device)


@torch.no_grad()
def soft_labels_for_set(name, dataset, device):
    """One fixed soft label per set image [HT Sec.3.2]: softmax(teacher(x)/T),
    teacher on its ImageNet-stat path (as everywhere in this stack)."""
    ds = DATASETS[dataset]
    pt = os.path.join(SET_DIR, name + ".pt")
    if os.path.exists(pt):
        imgs = torch.load(pt)["images"]
    else:
        root = os.path.join(SET_DIR, name)
        imgs = np.stack([np.array(Image.open(os.path.join(root, c, f)).convert("RGB"))
                         for c in sorted(os.listdir(root))
                         if os.path.isdir(os.path.join(root, c))
                         for f in sorted(os.listdir(os.path.join(root, c)))])
    x = to_norm_tensor(np.asarray(imgs), IMNET_MEAN, IMNET_STD).contiguous()
    teacher = get_teacher(dataset, device)
    out = []
    for i in range(0, len(x), 1024):
        out.append(F.softmax(teacher(x[i:i + 1024].to(device)) / KD_T, dim=1).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# SL trainer  [HT Tab.4 small-scale SL; ASSUME X1/X2]
# --------------------------------------------------------------------------- #
def sl_train(model, x, q_T, device, epochs=EVAL_EPOCHS, batch=EVAL_BATCH,
             epoch_hook=None):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dsa = ParamDiffAug()
    x = x.to(device)
    q_T = q_T.to(device)
    n = len(x)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = diff_augment(x[idx], ca2d.DSA_STRATEGY, dsa)
            log_p = F.log_softmax(model(xb) / KD_T, dim=1)
            loss = F.kl_div(log_p, q_T[idx], reduction="batchmean") * KD_T * KD_T
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch_hook is not None:
            epoch_hook(ep, model)
    return model


# --------------------------------------------------------------------------- #
# Eval cells
# --------------------------------------------------------------------------- #
def cell_path(dataset, set_name, arch, regime):
    return os.path.join(RESULT_DIR, f"xarch_{dataset}_{set_name}_{arch}_{regime}.json")


def eval_cell(dataset, set_name, arch, regime, seeds, device, batch=EVAL_BATCH):
    out_path = cell_path(dataset, set_name, arch, regime)
    if os.path.exists(out_path):
        with open(out_path) as f:
            res = json.load(f)
        if res.get("seeds") == list(seeds):
            print(f"[eval] cached: {os.path.basename(out_path)} -> "
                  f"{res['mean']:.2f} +- {res['std']:.2f}")
            return res
    t_all = time.time()
    x, y = load_set(set_name, dataset)
    q_T = soft_labels_for_set(set_name, dataset, device) if regime == "sl" else None
    xte, yte = load_test(dataset)
    accs, seed_secs = [], []
    for s in seeds:
        t0 = time.time()
        set_seed(s)
        model = build_model(arch, dataset)
        try:
            if regime == "hl":
                hl_train(model, x, y, device, epochs=EVAL_EPOCHS,
                         step_epoch=EVAL_STEP_EPOCH, aug="dsa", batch=batch)
            else:
                sl_train(model, x, q_T, device, batch=batch)
        except torch.cuda.OutOfMemoryError:
            print(f"[eval] OOM at batch {batch} -> retry batch {batch // 2}")
            torch.cuda.empty_cache()
            return eval_cell(dataset, set_name, arch, regime, seeds, device,
                             batch=batch // 2)
        acc = test_top1(model, xte, yte, device)
        accs.append(acc)
        seed_secs.append(time.time() - t0)
        print(f"[eval] {dataset}/{set_name}/{arch}/{regime} seed{s}: {acc:.2f}%  "
              f"({seed_secs[-1]:.0f}s)", flush=True)
    res = {"dataset": dataset, "name": set_name, "arch": arch, "regime": regime,
           "seeds": list(seeds), "accs": accs, "mean": float(np.mean(accs)),
           "std": float(np.std(accs)), "seed_secs": [round(t, 1) for t in seed_secs],
           "total_secs": round(time.time() - t_all, 1), "batch": batch}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eval] {dataset}/{set_name}/{arch}/{regime}: "
          f"{res['mean']:.2f} +- {res['std']:.2f} ({res['total_secs']:.0f}s)")
    return res


# --------------------------------------------------------------------------- #
# Aggregation: trio average + Welch p-values
# --------------------------------------------------------------------------- #
def welch_p(ours, ref):
    """One-sided Welch: P(ours <= ref) small => ours significantly higher."""
    return float(stats.ttest_ind(ours, ref, equal_var=False,
                                 alternative="greater").pvalue)


TRIO = ("mlp", "rn18", "rn152")  # kept for manual `eval`/`grid` use only


def _cell_accs(dataset, set_name, arch, regime):
    """Per-seed accs for a cell; falls back to the original CIFAR HL convnet
    results (results/{set}.json from the ca2d/rcad/sharp study)."""
    p = cell_path(dataset, set_name, arch, regime)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)["accs"]
    if dataset == "cifar100" and arch == "convnet" and regime == "hl":
        legacy = os.path.join(RESULT_DIR, f"{set_name}.json")
        if os.path.exists(legacy):
            with open(legacy) as f:
                return json.load(f)["accs"]
    return None


def print_full_table():
    """Paper-style console tables (Tab. 2 / Tab. 6 layout): Dataset Distillation
    vs Coreset Selection blocks, HL/SL column groups, best per block & column in
    bold, daggers = Welch significance of ours vs strongest local baseline."""
    B, E = "\033[1m", "\033[0m"
    CW = 14
    width = 26 + 4 * CW + 3
    for dataset, title, dd_pub, cs_pub in (
            ("cifar100", "CIFAR-100 (ConvNet-D3)", ("TM", "DM", "DC"),
             ("K-centers", "Random Real")),
            ("tin", "TinyImageNet (ConvNet-D4)", ("TM", "DATM", "DM", "DC"),
             ("K-centers", "Random Real"))):
        pub = dict(PAPER_TEX[dataset])
        keys = [t.replace("_ipc{ipc}", "") for t in DEFAULT_SETS[dataset]]
        k_rcad, k_sharp, k_cad, k_ca2d, k_flu, k_rnd = keys
        local = {k: {c: _local_accs(dataset, k, c[0], c[1]) for c in COLS}
                 for k in keys}
        ref = {}  # strongest local baseline per column (for daggers on ours)
        for c in COLS:
            base = {k: np.mean(local[k][c]) for k in (k_cad, k_ca2d, k_flu, k_rnd)
                    if local[k][c] is not None}
            if base:
                ref[c] = local[max(base, key=base.get)][c]
        blocks = (
            ("Dataset Distillation",
             [("pub", n, n) for n in dd_pub] + [("loc", SET_LABELS[k_ca2d], k_ca2d)]),
            ("Coreset Selection",
             [("pub", n, n) for n in cs_pub]
             + [("loc", SET_LABELS[k], k)
                for k in (k_rnd, k_cad, k_flu, k_sharp, k_rcad)]),
        )
        print("\n" + "=" * width)
        print(f"  {title}   (mean ± std over seeds 42-44; published rows = Dey et al.)")
        print("=" * width)
        print(f"{'':<26}|{'Hard Label (HL)':^{2 * CW}}|{'Fixed Soft Label (SL)':^{2 * CW}}")
        print(f"{'Method':<26}|{'IPC 10':^{CW}}{'IPC 50':^{CW}}|{'IPC 10':^{CW}}{'IPC 50':^{CW}}")
        for btitle, rows in blocks:
            print(f"{' ' + btitle + ' ':-^{width}}")
            rowsv = []
            for kind, label, ident in rows:
                cm = {}
                for c in COLS:
                    if kind == "pub":
                        ms = pub[ident].get(c)
                        cm[c] = (ms[0], ms[1], None) if ms else None
                    else:
                        v = local[ident][c]
                        cm[c] = ((float(np.mean(v)), float(np.std(v)), v)
                                 if v is not None else None)
                rowsv.append((label, kind == "loc" and ident in OUR_SET_KEYS, cm))
            best = {c: max((cm[c][0] for _, _, cm in rowsv if cm[c] is not None),
                           default=None) for c in COLS}
            for label, is_ours, cm in rowsv:
                line = f"{label:<26}|"
                for j, c in enumerate(COLS):
                    v = cm[c]
                    if v is None:
                        cell = f"{'--':^{CW}}"
                    else:
                        m, s, accs = v
                        mark = ""
                        if is_ours and accs is not None and c in ref:
                            p = welch_p(accs, ref[c])
                            mark = "‡" if p < 0.01 else ("†" if p < 0.05 else "")
                        cell = f"{f'{m:.2f}±{s:.2f}{mark}':^{CW}}"
                        if best[c] is not None and abs(m - best[c]) < 1e-9:
                            cell = B + cell + E
                    line += cell + ("|" if j == 1 else "")
                print(line)
    print("\nbold = best per block & column; †/‡ = one-sided Welch p<0.05/0.01 vs")
    print("strongest locally reproduced baseline in the column (n=3 seeds)\n")


# --------------------------------------------------------------------------- #
# LaTeX table generation (one file per paper table; regenerate anytime)
# --------------------------------------------------------------------------- #
TEX_DIR = os.path.join(ca2d.ROOT, "tex")
COLS = ((10, "hl"), (50, "hl"), (10, "sl"), (50, "sl"))
SET_LABELS = {
    "rcad_feat_fl": "R-CAD (ours)", "sharp_fl_grad_a100": "SHARP-FL (ours)",
    "cadprune": "CAD-Prune (repro.)", "ca2d": "CA2D (repro.)",
    "rcad_feat_fl_uni": "Coverage-only FL", "random": "Random Real (repro.)",
    "tin_rcad": "R-CAD (ours)", "tin_sharpfl": "SHARP-FL (ours)",
    "tin_cadprune": "CAD-Prune (repro.)", "tin_ca2d": "CA2D (repro.)",
    "tin_fluni": "Coverage-only FL", "tin_random": "Random Real (repro.)",
}
# Published rows [HT Tab.2/6]; (mean, std) per column.
PAPER_TEX = {
    "tin": [
        ("TM", {(10, "hl"): (20.11, .16), (50, "hl"): (28.16, .45), (10, "sl"): (26.11, .30), (50, "sl"): (36.43, .08)}),
        ("DATM", {(10, "hl"): (19.26, .19), (50, "hl"): (29.51, .37), (10, "sl"): (28.43, .27), (50, "sl"): (37.39, .28)}),
        ("DM", {(10, "hl"): (13.51, .31), (50, "hl"): (22.76, .28), (10, "sl"): (16.12, .33), (50, "sl"): (36.58, .07)}),
        ("DC", {(10, "hl"): (12.83, .14), (50, "hl"): (12.66, .36), (10, "sl"): (7.23, .20), (50, "sl"): (10.18, .32)}),
        ("K-centers", {(10, "hl"): (11.38, .26), (50, "hl"): (22.02, .40), (10, "sl"): (27.11, .13), (50, "sl"): (37.45, .21)}),
        ("Random Real", {(10, "hl"): (6.88, .25), (50, "hl"): (18.62, .22), (10, "sl"): (26.07, .48), (50, "sl"): (36.87, .17)}),
    ],
    "cifar100": [
        ("TM", {(10, "hl"): (38.18, .42), (50, "hl"): (46.32, .26), (10, "sl"): (37.60, .25), (50, "sl"): (46.26, .30)}),
        ("DM", {(10, "hl"): (29.23, .26), (50, "hl"): (42.32, .37), (10, "sl"): (26.13, .10), (50, "sl"): (43.46, .18)}),
        ("DC", {(10, "hl"): (28.42, .29), (50, "hl"): (30.56, .56), (10, "sl"): (23.54, .31), (50, "sl"): (33.46, .38)}),
        ("K-centers", {(10, "hl"): (25.04, .30), (50, "hl"): (38.64, .43), (10, "sl"): (34.70, .13), (50, "sl"): (46.24, .12)}),
        ("Random Real", {(10, "hl"): (18.64, .25), (50, "hl"): (34.66, .41), (10, "sl"): (33.43, .18), (50, "sl"): (45.39, .23)}),
    ],
}
OUR_SET_KEYS = ("rcad_feat_fl", "sharp_fl_grad_a100", "tin_rcad", "tin_sharpfl")


def _local_accs(dataset, key, ipc, regime):
    """Per-seed accs for a local row/column (ConvNet main-table protocol)."""
    return _cell_accs(dataset, f"{key}_ipc{ipc}", "convnet", regime)


def _tex_table(dataset, fname, caption, label):
    keys = [t.replace("_ipc{ipc}", "") for t in DEFAULT_SETS[dataset]]
    local = {k: {c: _local_accs(dataset, k, c[0], c[1]) for c in COLS}
             for k in keys}
    done = sum(v is not None for row in local.values() for v in row.values())
    # per-column best local mean and significance of ours vs best baseline
    best, ref = {}, {}
    for c in COLS:
        means = {k: np.mean(v[c]) for k, v in local.items() if v[c] is not None}
        if means:
            best[c] = max(means.values())
        base = {k: m for k, m in means.items() if k not in OUR_SET_KEYS}
        if base:
            ref[c] = local[max(base, key=base.get)][c]

    def cell(k, c):
        v = local[k][c]
        if v is None:
            return "--"
        m, s = np.mean(v), np.std(v)
        star = ""
        if k in OUR_SET_KEYS and c in ref:
            p = welch_p(v, ref[c])
            star = "$^{\\ddagger}$" if p < 0.01 else ("$^{\\dagger}$" if p < 0.05 else "")
        txt = f"{m:.2f}{{\\scriptsize$\\pm${s:.2f}}}{star}"
        return f"\\textbf{{{txt}}}" if abs(m - best.get(c, -1)) < 1e-9 else txt

    lines = [
        f"% auto-generated by `python xarch.py tex` ({time.strftime('%Y-%m-%d %H:%M')});",
        f"% {done}/24 local cells available -- regenerate as runs finish. '--' = pending.",
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l cc cc}", "\\toprule",
        " & \\multicolumn{2}{c}{Hard Label (HL)} & \\multicolumn{2}{c}{Fixed Soft Label (SL)} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Method & IPC 10 & IPC 50 & IPC 10 & IPC 50 \\\\", "\\midrule",
    ]
    for name, row in PAPER_TEX[dataset]:
        cells = " & ".join(
            (f"{row[c][0]:.2f}" + (f"{{\\scriptsize$\\pm${row[c][1]:.2f}}}" if row[c][1] is not None else ""))
            for c in COLS)
        lines.append(f"{name} & {cells} \\\\")
    lines.append("\\midrule")
    for k in keys:
        if k in OUR_SET_KEYS:
            continue
        lines.append(f"{SET_LABELS[k]} & " + " & ".join(cell(k, c) for c in COLS) + " \\\\")
    lines.append("\\midrule")
    for k in keys:
        if k in OUR_SET_KEYS:
            lines.append(f"\\textbf{{{SET_LABELS[k]}}} & "
                         + " & ".join(cell(k, c) for c in COLS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tex] {path}: {done}/24 local cells")


SIG_NOTE = ("$\\dagger$/$\\ddagger$: one-sided Welch $p<0.05$/$p<0.01$ vs the "
            "strongest locally reproduced baseline in the column (3 seeds; "
            "low-power test, reported for transparency). Rows above the first "
            "rule are the published numbers of Dey et al.; all other rows share "
            "one identical local protocol.")


def tex_tables():
    _tex_table("tin", "tab2_tinyimagenet.tex",
               "TinyImageNet, ConvNet-D4, IPC 10/50, HL and fixed-SL. " + SIG_NOTE,
               "tab:tin_main")
    _tex_table("cifar100", "tab6_cifar100.tex",
               "CIFAR-100, ConvNet-D3, IPC 10/50, HL and fixed-SL. R-CAD/SHARP-FL "
               "selection adds $<$15\\,s on top of the compute-matched scoring run "
               "shared with CAD-Prune. " + SIG_NOTE, "tab:cifar_main")


def print_timing():
    print(f"\n{'cell':<64}{'mean s/seed':>12}{'total s':>9}")
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(RESULT_DIR, f)) as fh:
            r = json.load(fh)
        if "seed_secs" in r:
            print(f"{f[:-5]:<64}{np.mean(r['seed_secs']):>12.1f}"
                  f"{r['total_secs']:>9.1f}")


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    print("[selftest] 1. param counts vs [HT Tab.7 header] (cifar100)")
    counts = {}
    for arch, ref in (("mlp", 0.41e6), ("rn18", 11e6), ("rn152", 60e6)):
        m = build_model(arch, "cifar100")
        n = sum(p.numel() for p in m.parameters())
        counts[arch] = n
        print(f"           {arch}: {n / 1e6:.2f}M (paper ~{ref / 1e6:.2f}M)")
    assert 0.35e6 < counts["mlp"] < 0.5e6
    assert 10e6 < counts["rn18"] < 12e6
    assert 55e6 < counts["rn152"] < 62e6
    print("           ok")

    print("[selftest] 2. forward shapes at 32px/100c and 64px/200c")
    for dsname in ("cifar100", "tin"):
        s, C = DATASETS[dsname]["size"], DATASETS[dsname]["classes"]
        for arch in ("convnet", "mlp", "rn18", "rn152"):
            out = build_model(arch, dsname)(torch.randn(2, 3, s, s))
            assert out.shape == (2, C), (dsname, arch, out.shape)
    print("           ok")

    print("[selftest] 3. SL loss sanity: KL at T=1 with one-hot teacher == CE")
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    q = F.one_hot(y, 10).float()
    kl = F.kl_div(F.log_softmax(logits, 1), q, reduction="batchmean")
    ce = F.cross_entropy(logits, y)
    assert abs(kl - ce) < 1e-5
    print("           ok")

    print("[selftest] 4. welch_p direction: clearly-higher sample -> small p")
    assert welch_p([30.0, 30.2, 30.1], [20.0, 20.2, 20.1]) < 0.01
    assert welch_p([20.0, 20.2, 20.1], [30.0, 30.2, 30.1]) > 0.9
    print("           ok")
    print("[selftest] all checks passed")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "eval", "grid", "timing",
                                    "table", "tex"])
    ap.add_argument("--dataset", choices=list(DATASETS), default="cifar100")
    ap.add_argument("--set", dest="set_name")
    ap.add_argument("--arch", choices=["convnet", "mlp", "rn18", "rn152"])
    ap.add_argument("--archs", type=lambda s: s.split(","),
                    default=["mlp", "rn18", "rn152"])
    ap.add_argument("--regime", choices=["hl", "sl"], default="hl")
    ap.add_argument("--ipc", type=int, choices=[10, 50])
    ap.add_argument("--sets", type=lambda s: s.split(","))
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")],
                    default=[42, 43, 44])
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "eval":
        assert args.set_name and args.arch
        eval_cell(args.dataset, args.set_name, args.arch, args.regime, args.seeds,
                  device)
    elif args.cmd == "grid":
        assert args.ipc
        names = args.sets or [t.format(ipc=args.ipc) for t in DEFAULT_SETS[args.dataset]]
        for n in names:
            for a in args.archs:
                eval_cell(args.dataset, n, a, args.regime, args.seeds, device)
    elif args.cmd == "timing":
        print_timing()
    elif args.cmd == "table":
        print_full_table()
    elif args.cmd == "tex":
        tex_tables()


if __name__ == "__main__":
    main()
