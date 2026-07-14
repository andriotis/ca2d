#!/usr/bin/env python
"""TinyImageNet-200 set creation for the CVPR comparison campaign (Tab. 2/8 bars).

Creates the six per-IPC sets compared on TinyImageNet (ConvNet-D4, HL protocol of
[HT Tab.4 small-scale]) with hyperparameters frozen from the CIFAR-100 study:

    tin_rcad      R-CAD: facility location on observer-feature cosine gram with CAD
                  demand weights (rcad.py, kernel=feat, selector=fl)
    tin_sharpfl   SHARP-FL: same coverage objective, demand = per-class shifted
                  z-score of A_grad = mean_k cos(p_k - y, p_k - q)  (sharp.py)
    tin_cadprune  top-IPC by CAD [HT Sec.5 Eq.1-2]           (paper's method)
    tin_ca2d      CAD top-mipc pool -> RDED crop/select/stitch (paper's method;
                  knobs from RDED/scripts/tinyimagenet_10ipc_conv4_to_conv4_cr5.sh:
                  factor=1, num_crop=5, mipc=300 — identical to CIFAR-100; the same
                  knobs are assumed for IPC 50 [ASSUME T2], no conv4 50-IPC script)
    tin_fluni     uniform facility location (coverage-only pole)
    tin_random    class-balanced uniform draw

Scoring run: compute-matched K = 300*ipc*200/100000 = 6 (IPC 10) / 30 (IPC 50)
epochs [HT Sec.5], instrumented from the start: per-epoch EL2N S, fp16 softmax
trajectories P, and teacher soft labels q (RDED conv4 observer) in a single run.

[ASSUME T1] TIN normalization stats (0.4802,0.4481,0.3975)/(0.2770,0.2691,0.2821)
(DC-bench convention) on the student path; ImageNet stats on the observer path (RDED).
Class order = alphabetical WordNet id, verified against RDED/prepare/tinyimagenet.md.

Usage:
    python tin.py selftest
    python tin.py observer   [--device cuda:1]
    python tin.py score      --ipc 10
    python tin.py select     --method {rcad,sharpfl,cadprune,ca2d,fluni,random} --ipc 10
    python tin.py all-select --ipc 10       # score (cached) + all six sets
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import ca2d
from ca2d import (ConvNet, EVAL_EPOCHS, EVAL_STEP_EPOCH, IMNET_MEAN, IMNET_STD,
                  RDED_FACTOR, RDED_MIPC, RDED_NUM_CROP, SCORE_DIR, SET_DIR,
                  cad_from_S, el2n_scores, hl_train, rded_mix_images,
                  rded_multi_random_crop, rded_save_images, rded_selector, set_seed,
                  test_top1, to_norm_tensor)
from rcad import cos_gram, greedy_fl, observer_features

TIN_CLASSES = 200
TIN_TRAIN = 100000
TIN_SIZE = 64
TIN_MEAN = (0.4802, 0.4481, 0.3975)  # [ASSUME T1] DC-bench TinyImageNet stats
TIN_STD = (0.2770, 0.2691, 0.2821)
TIN_DIR = os.path.join(ca2d.DATA_DIR, "tiny-imagenet-200")
TIN_CACHE = os.path.join(ca2d.DATA_DIR, "tin_cache.pt")
TIN_OBSERVER = os.path.join(ca2d.ART, "pretrain_models", "tinyimagenet_conv4.pth")
RDED_MAPPING_DOC = os.path.join(os.path.dirname(ca2d.ROOT), "RDED", "prepare",
                                "tinyimagenet.md")


def norm_tensor(images_uint8, mean, std):
    """to_norm_tensor + contiguous: at 64px the permuted (channels-last) strides
    propagate through cudnn and break ConvNet's .view flatten."""
    return to_norm_tensor(images_uint8, mean, std).contiguous()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def tin_wnids():
    return sorted(d for d in os.listdir(os.path.join(TIN_DIR, "train"))
                  if d.startswith("n"))


def load_tin():
    """Returns (xtr u8 (100000,64,64,3), ytr, xte u8 (10000,...), yte, wnids)."""
    if os.path.exists(TIN_CACHE):
        b = torch.load(TIN_CACHE, map_location="cpu")
        return b["xtr"].numpy(), b["ytr"].numpy(), b["xte"].numpy(), b["yte"].numpy(), b["wnids"]
    assert os.path.isdir(TIN_DIR), f"{TIN_DIR} missing (download tiny-imagenet-200)"
    wnids = tin_wnids()
    assert len(wnids) == TIN_CLASSES
    cls = {w: i for i, w in enumerate(wnids)}
    xtr, ytr = [], []
    for w in wnids:
        d = os.path.join(TIN_DIR, "train", w, "images")
        for f in sorted(os.listdir(d)):
            xtr.append(np.array(Image.open(os.path.join(d, f)).convert("RGB")))
            ytr.append(cls[w])
    xte, yte = [], []
    with open(os.path.join(TIN_DIR, "val", "val_annotations.txt")) as fh:
        ann = dict(line.split("\t")[:2] for line in fh)
    vdir = os.path.join(TIN_DIR, "val", "images")
    for f in sorted(os.listdir(vdir)):
        xte.append(np.array(Image.open(os.path.join(vdir, f)).convert("RGB")))
        yte.append(cls[ann[f]])
    xtr, ytr = np.stack(xtr), np.array(ytr)
    xte, yte = np.stack(xte), np.array(yte)
    assert xtr.shape == (TIN_TRAIN, 64, 64, 3) and len(xte) == 10000
    torch.save({"xtr": torch.from_numpy(xtr), "ytr": torch.from_numpy(ytr),
                "xte": torch.from_numpy(xte), "yte": torch.from_numpy(yte),
                "wnids": wnids}, TIN_CACHE)
    print(f"[tin] cached dataset -> {TIN_CACHE}")
    return xtr, ytr, xte, yte, wnids


def per_class_indices(ytr):
    return [np.where(ytr == c)[0] for c in range(TIN_CLASSES)]


def tin_convnet(norm):
    return ConvNet(num_classes=TIN_CLASSES, net_norm=norm, net_depth=4,
                   net_width=128, channel=3, im_size=(TIN_SIZE, TIN_SIZE))


def get_tin_observer(device, check_acc=False):
    model = tin_convnet("batch")  # RDED conv observers are batchnorm
    ckpt = torch.load(TIN_OBSERVER, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)  # key-for-key or it throws
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if check_acc:
        xtr, ytr, xte, yte, _ = load_tin()
        acc = test_top1(model, norm_tensor(xte, IMNET_MEAN, IMNET_STD),
                        torch.from_numpy(yte), device)
        print(f"[tin] observer val top-1 (raw 64x64) = {acc:.2f}%")
    return model


def compute_K(ipc):
    total = EVAL_EPOCHS * ipc * TIN_CLASSES  # [HT Sec.5] same compute budget
    assert total % TIN_TRAIN == 0
    return total // TIN_TRAIN


# --------------------------------------------------------------------------- #
# Instrumented scoring run (S + P + q in one deterministic run)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_probs(model, x, device, batch=1024):
    model.eval()
    out = torch.empty(len(x), TIN_CLASSES, dtype=torch.float16)
    for i in range(0, len(x), batch):
        out[i:i + batch] = F.softmax(model(x[i:i + batch].to(device)), dim=1).half().cpu()
    return out


def score_path(ipc):
    return os.path.join(SCORE_DIR, f"tin_score_ipc{ipc}.pt")


def run_score(ipc, device):
    out = score_path(ipc)
    if os.path.exists(out):
        print(f"[score] cached: {out}")
        return torch.load(out, map_location="cpu")
    t0 = time.time()
    K = compute_K(ipc)
    step_epoch = round(K * EVAL_STEP_EPOCH / EVAL_EPOCHS)  # [ca2d ASSUME 1]
    xtr_u8, ytr, _, _, _ = load_tin()
    observer = get_tin_observer(device)
    q = collect_probs(observer, norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD), device)

    set_seed(0)
    x = norm_tensor(xtr_u8, TIN_MEAN, TIN_STD)
    y = torch.from_numpy(ytr).long()
    y1h = F.one_hot(y, TIN_CLASSES).float()
    model = tin_convnet("instance")
    S = torch.empty(K, TIN_TRAIN)
    P = torch.empty(K, TIN_TRAIN, TIN_CLASSES, dtype=torch.float16)

    def hook(ep, m):
        S[ep] = el2n_scores(m, x, y1h, device)
        P[ep] = collect_probs(m, x, device)
        print(f"[score]   epoch {ep + 1}/{K} EL2N mean={S[ep].mean():.4f}", flush=True)

    print(f"[score] tin ipc={ipc}: K={K}, StepLR@{step_epoch}, aug=dsa")
    hl_train(model, x, y, device, epochs=K, step_epoch=step_epoch, aug="dsa",
             epoch_hook=hook)
    U, cad = cad_from_S(S, K - 2, 2)  # J=K-W, W=2 frozen [ca2d ASSUME 9]
    blob = {"S": S, "U": U, "cad": cad, "P": P, "q": q, "K": K,
            "step_epoch": step_epoch, "seed": 0, "aug": "dsa",
            "secs": time.time() - t0}
    torch.save(blob, out)
    print(f"[score] saved {out} ({blob['secs']:.0f}s)")
    return blob


# --------------------------------------------------------------------------- #
# Selections (frozen configs from the CIFAR-100 study)
# --------------------------------------------------------------------------- #
def a_grad_scores(P, q, y):
    """A_grad(i) = mean_k cos(p_k - y, p_k - q); same math as sharp.py."""
    N = P.shape[1]
    A = torch.empty(N)
    for i in range(0, N, 5000):
        p = P[:, i:i + 5000].float()
        e_h = p - F.one_hot(y[i:i + 5000], TIN_CLASSES).float()
        e_s = p - q[i:i + 5000].float()
        A[i:i + 5000] = ((e_h * e_s).sum(-1)
                         / (e_h.norm(dim=-1) * e_s.norm(dim=-1) + 1e-12)).mean(0)
    return A


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def tin_features(device):
    observer = get_tin_observer(device)
    xtr_u8, _, _, _, _ = load_tin()
    return observer_features(observer, norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD),
                             device)


def select_indices(method, ipc, device):
    blob = run_score(ipc, device)
    xtr_u8, ytr, _, _, _ = load_tin()
    y = torch.from_numpy(ytr).long()
    cad = blob["cad"]
    feats = tin_features(device) if method in ("rcad", "sharpfl", "fluni") else None
    A = a_grad_scores(blob["P"], blob["q"], y) if method == "sharpfl" else None
    rng = np.random.RandomState(0)
    keep = []
    for c, idx in enumerate(per_class_indices(ytr)):
        if method == "random":
            keep.append(rng.choice(idx, size=ipc, replace=False))
            continue
        if method == "cadprune":
            order = idx[np.argsort(-cad[idx].numpy(), kind="stable")]
            keep.append(order[:ipc])
            continue
        kappa = cos_gram(feats[idx].float())
        if method == "rcad":
            w = cad[idx].float()
        elif method == "fluni":
            w = torch.ones(len(idx))
        elif method == "sharpfl":
            s = zscore(A[idx])
            w = s - s.min() + 1e-6
        else:
            raise ValueError(method)
        keep.append(idx[np.array(greedy_fl(kappa, w, ipc))])
        if (c + 1) % 40 == 0:
            print(f"[select] {c + 1}/{TIN_CLASSES} classes", flush=True)
    return np.concatenate(keep)


def synth_ca2d(ipc, device, seed=0):
    """[HT Sec.5] CA2D on TIN: CAD top-mipc pool -> RDED Alg.1 (factor=1, 5 crops,
    mipc=300 [RDED tinyimagenet conv4 script; ASSUME T2 for IPC 50])."""
    name = f"tin_ca2d_ipc{ipc}"
    out_dir = os.path.join(SET_DIR, name)
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == TIN_CLASSES:
        print(f"[synth] cached: {out_dir}")
        return name
    t0 = time.time()
    blob = run_score(ipc, device)
    cad = blob["cad"]
    xtr_u8, ytr, _, _, _ = load_tin()
    observer = get_tin_observer(device)
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD).view(1, 3, 1, 1)
    n_patches = ipc * RDED_FACTOR ** 2
    for c, idx in enumerate(per_class_indices(ytr)):
        pool = idx[np.argsort(-cad[idx].numpy(), kind="stable")][:RDED_MIPC]
        imgs = torch.from_numpy(xtr_u8[pool]).float().permute(0, 3, 1, 2) / 255.0
        set_seed(seed * 1000003 + c)
        crops = rded_multi_random_crop(imgs, RDED_NUM_CROP, TIN_SIZE, RDED_FACTOR)
        crops = (crops - mean.unsqueeze(0)) / std.unsqueeze(0)
        labels = torch.full((len(pool),), c, dtype=torch.long)
        picked = rded_selector(n_patches, observer, crops, labels, TIN_SIZE,
                               RDED_NUM_CROP, device)
        mixed = rded_mix_images(picked.cpu(), TIN_SIZE, RDED_FACTOR, ipc)
        rded_save_images(mixed * std + mean, c, out_dir)
        if (c + 1) % 40 == 0:
            print(f"[synth] {c + 1}/{TIN_CLASSES} classes", flush=True)
    with open(os.path.join(out_dir, "..", name + "_secs.json"), "w") as f:
        json.dump({"secs": time.time() - t0}, f)
    print(f"[synth] saved {out_dir} ({time.time() - t0:.0f}s)")
    return name


def run_select(method, ipc, device):
    if method == "ca2d":
        return synth_ca2d(ipc, device)
    name = f"tin_{method}_ipc{ipc}"
    out = os.path.join(SET_DIR, name + ".pt")
    if os.path.exists(out):
        print(f"[select] cached: {out}")
        return name
    t0 = time.time()
    xtr_u8, ytr, _, _, _ = load_tin()
    keep = select_indices(method, ipc, device)
    torch.save({"images": xtr_u8[keep], "labels": ytr[keep], "indices": keep,
                "secs": time.time() - t0}, out)
    print(f"[select] saved {out} ({time.time() - t0:.0f}s)")
    return name


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    print("[selftest] 1. compute-matched K: ipc10 -> 6, ipc50 -> 30")
    assert compute_K(10) == 6 and compute_K(50) == 30
    print("           ok")

    print("[selftest] 2. ConvNet-D4 shapes at 64px (200 classes)")
    m = tin_convnet("instance")
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, TIN_CLASSES)
    assert m.classifier.in_features == 128 * 4 * 4
    print("           ok  (params %.2fM)" % (sum(p.numel() for p in m.parameters()) / 1e6))

    print("[selftest] 3. A_grad identity vs manual loop on fake trajectories")
    K, n, C = 3, 5, TIN_CLASSES
    P = F.softmax(torch.randn(K, n, C), -1).half()
    q = F.softmax(torch.randn(n, C), -1).half()
    y = torch.randint(0, C, (n,))
    A = a_grad_scores(P, q, y)
    for i in range(n):
        vals = []
        for k in range(K):
            eh = P[k, i].float() - F.one_hot(y[i], C).float()
            es = P[k, i].float() - q[i].float()
            vals.append(float(F.cosine_similarity(eh, es, dim=0)))
        assert abs(A[i] - np.mean(vals)) < 1e-4
    print("           ok")

    if os.path.isdir(TIN_DIR):
        print("[selftest] 4. wnid order matches RDED index mapping")
        wnids = tin_wnids()
        assert len(wnids) == TIN_CLASSES
        if os.path.exists(RDED_MAPPING_DOC):
            with open(RDED_MAPPING_DOC) as f:
                pairs = [l.strip().split(": ") for l in f
                         if l.strip()[:1].isdigit() and ": n" in l]
            for k, w in pairs:
                assert wnids[int(k)] == w, (k, w, wnids[int(k)])
            print(f"           ok ({len(pairs)} mapping entries verified)")
        else:
            print("           mapping doc missing -> skipped")
        print("[selftest] 5. loader counts (may take a while on first run)")
        xtr, ytr, xte, yte, _ = load_tin()
        assert xtr.shape == (TIN_TRAIN, 64, 64, 3) and xte.shape[0] == 10000
        assert all(len(ix) == 500 for ix in per_class_indices(ytr))
        assert np.bincount(yte, minlength=TIN_CLASSES).min() == 50
        print("           ok")
    else:
        print("[selftest] 4-5. dataset not downloaded yet -> skipped")
    print("[selftest] all checks passed")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "observer", "score", "select",
                                    "all-select"])
    ap.add_argument("--ipc", type=int, choices=[10, 50])
    ap.add_argument("--method", choices=["rcad", "sharpfl", "cadprune", "ca2d",
                                         "fluni", "random"])
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "observer":
        get_tin_observer(device, check_acc=True)
    elif args.cmd == "score":
        assert args.ipc
        run_score(args.ipc, device)
    elif args.cmd == "select":
        assert args.ipc and args.method
        run_select(args.method, args.ipc, device)
    elif args.cmd == "all-select":
        assert args.ipc
        run_score(args.ipc, device)
        for m in ("random", "cadprune", "fluni", "rcad", "sharpfl", "ca2d"):
            run_select(m, args.ipc, device)


if __name__ == "__main__":
    main()
