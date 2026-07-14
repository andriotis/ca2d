#!/usr/bin/env python
"""R-CAD: Relational Compute-Aware Difficulty selection (novel method i).

CAD-Prune [HT Sec.5, Eq.1-2] ranks each image's learning trajectory in a vacuum and
keeps the per-class top-IPC — a pointwise rule that happily selects IPC near-copies of
the same difficult mode. R-CAD replaces the ranking with a *set-level* objective over
the relational structure of the trajectories: keep samples whose dynamics are
individually informative (high CAD) and jointly non-redundant (their trajectories
cover, rather than duplicate, the class's learning dynamics).

Per class c with pool P_c = all class images (full pool, equal terms with stock):

  Embeddings (--kernel), one vector per image i:
    softtraj  E_i = [p_1(i)-y_i ; ... ; p_K(i)-y_i]  in R^{K*C}   (probe.py fp16 probs)
    traj      E_i = centered EL2N row S[:,i] in R^K  (cosine == Pearson correlation;
              zero-extra-compute: reuses the cached scoring run of ca2d.py)
    feat      E_i = L2-normalized observer penultimate features (static geometry)
  Similarity kappa(i,j) = max(0, cos(E_i, E_j)).

  Selectors (--selector), class-balanced, budget = IPC (or 300 for synthesis pools):
    fl       CAD-weighted facility location:
                 max_{|A|=b}  sum_{i in P_c}  w_i * max_{j in A} kappa(i,j)
             monotone submodular -> greedy with (1-1/e) guarantee.
    dcad     correlation-discounted CAD, greedy:
                 pick argmax_i  w_i * (1 - lam * max_{j in A} kappa(i,j))
             lam=0 recovers CAD-Prune exactly (selftest-verified).
    kcenter  farthest-point traversal (start = class medoid [ASSUME R1]); with
             --weight uniform --kernel feat this is the K-centers coreset baseline
             [HT Tab.6: 25.04/38.64 on CIFAR-100 HL], the coverage-only pole of the
             same framework (CAD-Prune being the difficulty-only pole).

  Weights (--weight): cad = CAD scores from the cached scoring run; uniform = 1.

Artifacts are path-keyed by the full config, e.g. sets/rcad_softtraj_fl_ipc10.pt,
results/rcad_feat_kcenter_uni_ipc10.json. Baseline caches are never touched.

Usage:
    python rcad.py selftest
    python rcad.py select --ipc 10 --kernel softtraj --selector fl
    python rcad.py eval   --ipc 10 --kernel softtraj --selector fl [--seeds 42,43,44]
    python rcad.py synth  --ipc 10 --kernel softtraj --selector fl   # RDED-style DD
    python rcad.py table
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
from ca2d import (CIFAR_MEAN, CIFAR_STD, EVAL_EPOCHS, EVAL_STEP_EPOCH, IMNET_MEAN,
                  IMNET_STD, IM_SIZE, NUM_CLASSES, RDED_FACTOR, RDED_MIPC,
                  RDED_NUM_CROP, RESULT_DIR, SCORE_DIR, SET_DIR, convnet_d3,
                  get_observer, hl_train, load_cifar100, per_class_indices,
                  rded_mix_images, rded_multi_random_crop, rded_save_images,
                  rded_selector, set_seed, test_top1, to_norm_tensor)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
@torch.no_grad()
def observer_features(model, x, device, batch=1024, normalize=True):
    """Observer penultimate features. L2-normalized by default (mandatory for cosine
    kernels — established in the RDED-fork Mahalanobis work); raw for the Euclidean
    DeepCore-style k-center path (kernel=featraw)."""
    model.eval()
    feats = []
    for i in range(0, len(x), batch):
        z = x[i:i + batch].to(device)
        for d in range(model.depth):
            z = model.layers["conv"][d](z)
            if len(model.layers["norm"]) > 0:
                z = model.layers["norm"][d](z)
            z = model.layers["act"][d](z)
            if len(model.layers["pool"]) > 0:
                z = model.layers["pool"][d](z)
        z = z.reshape(z.shape[0], -1)
        feats.append((F.normalize(z, dim=1) if normalize else z).cpu())
    return torch.cat(feats)


def load_probe(ipc):
    path = os.path.join(SCORE_DIR, f"probe_ipc{ipc}.pt")
    assert os.path.exists(path), f"{path} missing: run `python probe.py --ipc {ipc}`"
    return torch.load(path, map_location="cpu")


def load_cad_blob(ipc):
    path = os.path.join(SCORE_DIR, f"cad_ipc{ipc}.pt")
    assert os.path.exists(path), f"{path} missing (baseline scoring cache)"
    return torch.load(path, map_location="cpu")


def build_embeddings(kernel, ipc, device):
    """Returns a callable idx -> (n, d) fp32 embedding matrix for a class pool."""
    if kernel == "traj":
        S = load_cad_blob(ipc)["S"]  # (K, N)

        def emb(idx):
            t = S[:, idx].T.clone().float()          # (n, K)
            t = t - t.mean(dim=1, keepdim=True)      # cosine of centered == Pearson
            return t
        return emb
    if kernel == "softtraj":
        P = load_probe(ipc)["P"]  # (K, N, C) fp16
        ytr = load_cifar100()[1]

        def emb(idx):
            E = P[:, idx, :].float()                 # (K, n, C)
            E[:, :, int(ytr[idx[0]])] -= 1.0         # p_k - onehot(y)
            return E.permute(1, 0, 2).reshape(len(idx), -1)
        return emb
    if kernel in ("feat", "featraw"):
        observer = get_observer(device, allow_train_fallback=False)
        xtr_u8, _, _, _ = load_cifar100()
        x_im = to_norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD)
        feats = observer_features(observer, x_im, device,
                                  normalize=(kernel == "feat"))

        def emb(idx):
            return feats[idx].float()
        return emb
    raise ValueError(kernel)


def cos_gram(E):
    """kappa = max(0, cos(E_i, E_j)); zero rows yield zero similarity (no NaN)."""
    En = F.normalize(E, dim=1)
    return (En @ En.T).clamp_(min=0)


# --------------------------------------------------------------------------- #
# Selectors (CPU, deterministic; n<=500 so exact greedy is trivial)
# --------------------------------------------------------------------------- #
def greedy_fl(kappa, w, budget):
    """Greedy max of F(A) = sum_i w_i * max_{j in A} kappa(i,j)."""
    n = kappa.shape[0]
    cur = torch.zeros(n)
    sel = []
    for _ in range(budget):
        gains = ((kappa - cur.unsqueeze(1)).clamp(min=0) * w.unsqueeze(1)).sum(0)
        gains[sel] = -1.0
        j = int(torch.argmax(gains))
        sel.append(j)
        cur = torch.maximum(cur, kappa[:, j])
    return sel


def greedy_dcad(kappa, w, budget, lam):
    """Greedy correlation-discounted CAD; lam=0 == stable top-w (CAD-Prune)."""
    n = kappa.shape[0]
    m = torch.zeros(n)
    sel = []
    for _ in range(budget):
        score = w * (1.0 - lam * m)
        score[sel] = -float("inf")
        j = int(torch.argmax(score))
        sel.append(j)
        m = torch.maximum(m, kappa[:, j])
    return sel


def greedy_kcenter(kappa, budget):
    """Farthest-point traversal in similarity space; start = medoid [ASSUME R1]."""
    sel = [int(torch.argmax(kappa.sum(0)))]
    near = kappa[:, sel[0]].clone()
    for _ in range(budget - 1):
        cand = near.clone()
        cand[sel] = float("inf")
        j = int(torch.argmin(cand))
        sel.append(j)
        near = torch.maximum(near, kappa[:, j])
    return sel


def kmeans_medoids(E, budget, seed=0, iters=100):
    """Lloyd k-means on embedding rows (k-means++ init, seeded), then the nearest
    *sample* to each centroid, deduplicated greedily [ASSUME R2: 'cluster centers'
    reading of K-centers, HT App.A.3]."""
    gen = torch.Generator().manual_seed(seed)
    n = E.shape[0]
    cent = E[torch.randint(n, (1,), generator=gen)]
    for _ in range(budget - 1):  # k-means++ seeding
        d2 = torch.cdist(E, cent).min(dim=1).values.pow(2)
        cent = torch.cat([cent, E[torch.multinomial(d2 + EPS_KM, 1, generator=gen)]])
    for _ in range(iters):
        assign = torch.cdist(E, cent).argmin(dim=1)
        new = torch.stack([E[assign == k].mean(0) if (assign == k).any() else cent[k]
                           for k in range(budget)])
        if torch.allclose(new, cent):
            break
        cent = new
    sel = []
    for k in range(budget):  # nearest unused sample per centroid
        order = torch.argsort(torch.cdist(E, cent[k:k + 1]).squeeze(1))
        sel.append(next(int(j) for j in order if int(j) not in sel))
    return sel


EPS_KM = 1e-9


def select_indices(ipc, kernel, selector, lam, weight, budget, device):
    """Per-class relational selection; returns global train indices (class-blocked)."""
    _, ytr, _, _ = load_cifar100()
    emb = build_embeddings(kernel, ipc, device)
    cad = load_cad_blob(ipc)["cad"]
    if kernel == "featraw":
        assert selector in ("kcenter", "kmed"), \
            "featraw (Euclidean) only supports kcenter/kmed"
    keep = []
    for c, idx in enumerate(per_class_indices(ytr)):
        E = emb(idx)
        # featraw: DeepCore-style Euclidean geometry (similarity = -distance);
        # otherwise clipped cosine.
        kappa = -torch.cdist(E, E) if kernel == "featraw" else cos_gram(E)
        w = cad[idx].float() if weight == "cad" else torch.ones(len(idx))
        if selector == "fl":
            sel = greedy_fl(kappa, w, budget)
        elif selector == "dcad":
            sel = greedy_dcad(kappa, w, budget, lam)
        elif selector == "kcenter":
            sel = greedy_kcenter(kappa, budget)
        elif selector == "kmed":
            sel = kmeans_medoids(E, budget, seed=c)
        else:
            raise ValueError(selector)
        keep.append(idx[np.array(sel)])
        if (c + 1) % 20 == 0:
            print(f"[select] {c + 1}/{NUM_CLASSES} classes", flush=True)
    return np.concatenate(keep)


# --------------------------------------------------------------------------- #
# Naming / IO
# --------------------------------------------------------------------------- #
def cfg_name(prefix, ipc, kernel, selector, lam, weight):
    parts = [prefix, kernel, selector]
    if selector == "dcad":
        parts.append(f"l{lam:g}")
    if weight == "uniform":
        parts.append("uni")
    return "_".join(parts) + f"_ipc{ipc}"


def run_select(args, device, budget=None, prefix="rcad"):
    name = cfg_name(prefix, args.ipc, args.kernel, args.selector, args.lam, args.weight)
    out = os.path.join(SET_DIR, name + ".pt")
    if os.path.exists(out):
        print(f"[select] cached: {out}")
        return name
    os.makedirs(SET_DIR, exist_ok=True)
    xtr_u8, ytr, _, _ = load_cifar100()
    keep = select_indices(args.ipc, args.kernel, args.selector, args.lam, args.weight,
                          budget or args.ipc, device)
    torch.save({"images": xtr_u8[keep], "labels": ytr[keep], "indices": keep}, out)
    print(f"[select] saved {out}")
    return name


# --------------------------------------------------------------------------- #
# RDED-style synthesis variant (relational pool of RDED_MIPC=300 -> crop/stitch)
# --------------------------------------------------------------------------- #
def run_synth(args, device, seed=0):
    name = cfg_name("rcad2d", args.ipc, args.kernel, args.selector, args.lam,
                    args.weight)
    out_dir = os.path.join(SET_DIR, name)
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == NUM_CLASSES:
        print(f"[synth] cached: {out_dir}")
        return name
    xtr_u8, ytr, _, _ = load_cifar100()
    pool_idx = select_indices(args.ipc, args.kernel, args.selector, args.lam,
                              args.weight, RDED_MIPC, device)
    pools = pool_idx.reshape(NUM_CLASSES, RDED_MIPC)
    observer = get_observer(device, allow_train_fallback=False)
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD).view(1, 3, 1, 1)
    n_patches = args.ipc * RDED_FACTOR ** 2
    for c in range(NUM_CLASSES):  # mirrors ca2d.synthesize_ca2d with our pool
        imgs = torch.from_numpy(xtr_u8[pools[c]]).float().permute(0, 3, 1, 2) / 255.0
        set_seed(seed * 1000003 + c)
        crops = rded_multi_random_crop(imgs, RDED_NUM_CROP, IM_SIZE, RDED_FACTOR)
        crops = (crops - mean.unsqueeze(0)) / std.unsqueeze(0)
        labels = torch.full((len(pools[c]),), c, dtype=torch.long)
        picked = rded_selector(n_patches, observer, crops, labels, IM_SIZE,
                               RDED_NUM_CROP, device)
        mixed = rded_mix_images(picked.cpu(), IM_SIZE, RDED_FACTOR, args.ipc)
        rded_save_images(mixed * std + mean, c, out_dir)
        if (c + 1) % 20 == 0:
            print(f"[synth] {c + 1}/{NUM_CLASSES} classes", flush=True)
    print(f"[synth] saved {out_dir}")
    return name


# --------------------------------------------------------------------------- #
# Eval (identical HL protocol to ca2d.run_eval; name-keyed artifacts)
# --------------------------------------------------------------------------- #
def load_selected(name):
    pt = os.path.join(SET_DIR, name + ".pt")
    if os.path.exists(pt):
        blob = torch.load(pt)
        return (to_norm_tensor(blob["images"], CIFAR_MEAN, CIFAR_STD),
                torch.from_numpy(blob["labels"]).long())
    root = os.path.join(SET_DIR, name)
    assert os.path.isdir(root), f"no set named {name}"
    imgs, labels = [], []
    for cdir in sorted(os.listdir(root)):
        for f in sorted(os.listdir(os.path.join(root, cdir))):
            imgs.append(np.array(Image.open(os.path.join(root, cdir, f)).convert("RGB")))
            labels.append(int(cdir))
    return (to_norm_tensor(np.stack(imgs), CIFAR_MEAN, CIFAR_STD),
            torch.tensor(labels, dtype=torch.long))


def eval_selected(name, ipc, seeds, device):
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"{name}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            res = json.load(f)
        if res.get("seeds") == list(seeds):
            print(f"[eval] cached: {name} -> {res['mean']:.2f} +- {res['std']:.2f}")
            return res
    x, y = load_selected(name)
    assert len(x) == ipc * NUM_CLASSES, f"set size {len(x)} != {ipc * NUM_CLASSES}"
    _, _, xte_u8, yte = load_cifar100()
    xte = to_norm_tensor(xte_u8, CIFAR_MEAN, CIFAR_STD)
    yte = torch.from_numpy(yte)
    accs = []
    for s in seeds:
        t0 = time.time()
        set_seed(s)
        model = convnet_d3("instance")
        hl_train(model, x, y, device, epochs=EVAL_EPOCHS, step_epoch=EVAL_STEP_EPOCH,
                 aug="dsa")
        acc = test_top1(model, xte, yte, device)
        accs.append(acc)
        print(f"[eval] {name} seed{s}: {acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    res = {"name": name, "ipc": ipc, "seeds": list(seeds), "accs": accs,
           "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eval] {name}: {res['mean']:.2f} +- {res['std']:.2f}")
    return res


def print_table():
    rows = []
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(RESULT_DIR, f)) as fh:
            r = json.load(fh)
        name = r.get("name") or f"{r['method']}_ipc{r['ipc']}{r.get('tag', '')}"
        rows.append((r["ipc"], -r["mean"], name, r["mean"], r["std"]))
    print(f"\n{'set':<40}{'IPC':>4}{'top-1':>18}")
    for ipc, _, name, m, s in sorted(rows):
        print(f"{name:<40}{ipc:>4}{m:>12.2f} +- {s:.2f}")
    print()


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    print("[selftest] 1. greedy FL >= (1-1/e) * exhaustive optimum (n=8, b=2)")
    torch.manual_seed(0)
    E = torch.randn(8, 5)
    kappa = cos_gram(E)
    w = torch.rand(8) + 0.1

    def F_val(A):
        return float((w * kappa[:, list(A)].max(dim=1).values).sum())
    best = max(F_val({i, j}) for i in range(8) for j in range(i + 1, 8))
    got = F_val(greedy_fl(kappa, w, 2))
    assert got >= (1 - 1 / np.e) * best - 1e-6, (got, best)
    print(f"           ok (greedy {got:.4f} vs opt {best:.4f})")

    print("[selftest] 2. dcad lam=0 == stable top-w order")
    sel = greedy_dcad(kappa, w, 4, lam=0.0)
    expect = torch.argsort(-w, stable=True)[:4].tolist()
    assert sel == expect, (sel, expect)
    print("           ok")

    print("[selftest] 3. kcenter picks one point per far cluster")
    E2 = torch.cat([torch.randn(5, 4) * 0.01 + torch.tensor([10., 0, 0, 0]),
                    torch.randn(5, 4) * 0.01 + torch.tensor([0., 10, 0, 0])])
    sel = greedy_kcenter(cos_gram(E2), 2)
    assert (sel[0] < 5) != (sel[1] < 5), sel
    print("           ok")

    print("[selftest] 4. zero-variance traj row -> zero similarity, no NaN")
    E3 = torch.tensor([[1., 1., 1.], [0.5, 1.0, 1.5]])
    E3 = E3 - E3.mean(dim=1, keepdim=True)
    k3 = cos_gram(E3)
    assert torch.isfinite(k3).all() and k3[0, 1] < 1e-6 and abs(k3[1, 1] - 1) < 1e-6
    print("           ok")

    print("[selftest] 5. dcad lam=0 pipeline == cached cadprune_ipc10 indices")
    cp_path = os.path.join(SET_DIR, "cadprune_ipc10.pt")
    if os.path.exists(cp_path):
        cad = load_cad_blob(10)["cad"]
        _, ytr, _, _ = load_cifar100()
        keep = []
        for idx in per_class_indices(ytr):
            # kappa content is irrelevant at lam=0; identity-sized zeros suffice
            sel = greedy_dcad(torch.zeros(len(idx), len(idx)), cad[idx].float(),
                              10, lam=0.0)
            keep.append(idx[np.array(sel)])
        keep = np.concatenate(keep)
        ref = torch.load(cp_path)["indices"]
        assert np.array_equal(np.sort(keep), np.sort(np.asarray(ref))), \
            "lam=0 selection != cached CAD-Prune set"
        print("           ok (10x100 indices identical)")
    else:
        print("           skipped (no cached cadprune set)")
    print("[selftest] all checks passed")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "select", "eval", "synth", "table"])
    ap.add_argument("--ipc", type=int, choices=[10, 50])
    ap.add_argument("--kernel", choices=["softtraj", "traj", "feat", "featraw"],
                    default="softtraj")
    ap.add_argument("--selector", choices=["fl", "dcad", "kcenter", "kmed"],
                    default="fl")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--weight", choices=["cad", "uniform"], default="cad")
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")],
                    default=[42, 43, 44])
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.cmd == "selftest":
        selftest()
        return
    if args.cmd == "table":
        print_table()
        return
    assert args.ipc, "--ipc required"
    if args.cmd == "select":
        run_select(args, device)
    elif args.cmd == "eval":
        name = run_select(args, device)
        eval_selected(name, args.ipc, args.seeds, device)
    elif args.cmd == "synth":
        name = run_synth(args, device)
        eval_selected(name, args.ipc, args.seeds, device)


if __name__ == "__main__":
    main()
