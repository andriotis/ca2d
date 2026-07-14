#!/usr/bin/env python
"""SHARP: Soft-Hard Alignment + Representativeness Pruning (novel method ii).

[HT Sec.3] shows that training with soft labels homogenizes per-sample influence and
approaches full-data accuracy regardless of subset quality, while hard-label (HL)
training stays quality-sensitive. SHARP selects the images whose *hard-label training
signal already behaves like soft-label training* — images whose one-hot label is so
informative that it delivers the teacher's supervision for free — intersected with
class representativeness.

Why the signal must be dynamic: statically, the hard/soft per-logit gradient
difference is (p-y) - (p-q) = q - y, independent of the student — matching on it
degenerates to teacher-confidence selection, i.e. RDED's criterion [RDED Eq.8].
On the trajectory, however, the last-layer gradients under the two labelings are
    grad_W CE_hard = (p_k - y) phi(x)^T,   grad_W CE_soft = (p_k - q) phi(x)^T,
which share the feature factor phi (also with bias, via phi -> [phi; 1]), so their
cosine is exactly cos(p_k - y, p_k - q): student-dependent, evolving over training,
and computable from the probe.py softmax trajectories (identity selftest-verified).

Alignment scores over the compute-matched scoring run (--align), K epochs [HT Sec.5]:
    grad   A(i) =  mean_k cos(p_k(i) - y_i, p_k(i) - q_i)        (last-layer gradient
           alignment between hard- and soft-label supervision)
    pred   A(i) = -mean_k ||p_k(i) - q_i||_2                     (EL2N-SL [HT Def.1]
           measured along a hard-label run)
    2run   A(i) = -mean_k ||p_k^HL(i) - p_k^SL(i)||_2            (probe --twin: exact
           counterfactual run trained on q with identical RNG stream)

Representativeness:
    R(i) = cos(phi_hat(x_i), mu_hat_c)   on L2-normalized observer penultimate
    features; mu_hat_c = renormalized class mean.

Selection: per-class z-scores, score = alpha * z(A) + (1 - alpha) * z(R), top-IPC.
Ablations: alpha in {0 (repr only), 0.5, 1 (alignment only)} x align definitions.
Teacher q: RDED observer ckpt, tau=1 [ASSUME P1], ImageNet-stat path only, as ca2d.py.

Usage:
    python sharp.py selftest
    python sharp.py select --ipc 10 --align grad --alpha 0.5
    python sharp.py eval   --ipc 10 --align grad --alpha 0.5 [--seeds 42,43,44]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import ca2d
from ca2d import (IMNET_MEAN, IMNET_STD, NUM_CLASSES, SCORE_DIR, SET_DIR,
                  convnet_d3, get_observer, load_cifar100, per_class_indices,
                  to_norm_tensor)
from rcad import (build_embeddings, cos_gram, eval_selected, greedy_fl, load_probe,
                  observer_features)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def alignment_scores(ipc, align):
    """A(i) from the probe trajectories; fp32, computed in sample chunks."""
    probe = load_probe(ipc)
    P, q = probe["P"], probe["q"]  # (K, N, C) fp16, (N, C) fp16
    K, N, C = P.shape
    P2 = None
    if align == "2run":
        twin_path = os.path.join(SCORE_DIR, f"probe_twin_ipc{ipc}.pt")
        assert os.path.exists(twin_path), \
            f"{twin_path} missing: run `python probe.py --ipc {ipc} --twin`"
        P2 = torch.load(twin_path, map_location="cpu")["P"]
    _, ytr, _, _ = load_cifar100()
    y = torch.from_numpy(ytr).long()
    A = torch.empty(N)
    for i in range(0, N, 5000):
        p = P[:, i:i + 5000].float()                       # (K, n, C)
        if align == "grad":
            e_h = p - F.one_hot(y[i:i + 5000], C).float()
            e_s = p - q[i:i + 5000].float()
            cos = (e_h * e_s).sum(-1) / (e_h.norm(dim=-1) * e_s.norm(dim=-1) + EPS)
            A[i:i + 5000] = cos.mean(0)
        elif align == "pred":
            A[i:i + 5000] = -(p - q[i:i + 5000].float()).norm(dim=-1).mean(0)
        elif align == "2run":
            A[i:i + 5000] = -(p - P2[:, i:i + 5000].float()).norm(dim=-1).mean(0)
        else:
            raise ValueError(align)
    return A


def representativeness_scores(device):
    """R(i) = cos(phi_hat_i, mu_hat_c) on L2-normalized observer features."""
    observer = get_observer(device, allow_train_fallback=False)
    xtr_u8, ytr, _, _ = load_cifar100()
    x_im = to_norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD)
    feats = observer_features(observer, x_im, device)      # (N, d), normalized
    R = torch.empty(len(feats))
    for idx in per_class_indices(ytr):
        mu = F.normalize(feats[idx].mean(0, keepdim=True), dim=1)
        R[idx] = (feats[idx] @ mu.T).squeeze(1)
    return R


def zscore(v):
    return (v - v.mean()) / (v.std() + EPS)


def select_sharp(ipc, align, alpha, mode, device):
    """mode=topk: per-class top-IPC of alpha*z(A) + (1-alpha)*z(R).
    mode=fl: representativeness as *coverage* instead of centroid proximity —
    facility location on the observer feature kernel with per-sample demand
    weights = the (shifted) alignment mixture, i.e. select a set that covers the
    class while prioritizing soft-hard-aligned samples. Returns global indices."""
    A = alignment_scores(ipc, align) if alpha > 0 else None
    R = representativeness_scores(device) if (alpha < 1 and mode == "topk") else None
    emb = build_embeddings("feat", ipc, device) if mode == "fl" else None
    _, ytr, _, _ = load_cifar100()
    keep = []
    for idx in per_class_indices(ytr):
        s = torch.zeros(len(idx))
        if A is not None:
            s = s + alpha * zscore(A[idx])
        if R is not None:
            s = s + (1 - alpha) * zscore(R[idx])
        if mode == "fl":
            w = s - s.min() + 1e-6  # nonneg demand weights for facility location
            sel = np.array(greedy_fl(cos_gram(emb(idx)), w, ipc))
            keep.append(idx[sel])
        else:
            order = torch.argsort(-s, stable=True)[:ipc].numpy()
            keep.append(idx[order])
    return np.concatenate(keep)


def cfg_name(ipc, align, alpha, mode):
    prefix = "sharp_fl" if mode == "fl" else "sharp"
    if alpha == 0:
        return f"{prefix}_repr_ipc{ipc}"
    return f"{prefix}_{align}_a{int(round(alpha * 100))}_ipc{ipc}"


def run_select(args, device):
    name = cfg_name(args.ipc, args.align, args.alpha, args.mode)
    out = os.path.join(SET_DIR, name + ".pt")
    if os.path.exists(out):
        print(f"[select] cached: {out}")
        return name
    os.makedirs(SET_DIR, exist_ok=True)
    xtr_u8, ytr, _, _ = load_cifar100()
    keep = select_sharp(args.ipc, args.align, args.alpha, args.mode, device)
    torch.save({"images": xtr_u8[keep], "labels": ytr[keep], "indices": keep}, out)
    print(f"[select] saved {out}")
    return name


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    print("[selftest] 1. last-layer gradient cosine factorization identity")
    torch.manual_seed(0)
    model = convnet_d3("instance")
    x = torch.randn(3, 3, 32, 32)
    y = torch.randint(0, NUM_CLASSES, (3,))
    q = F.softmax(torch.randn(3, NUM_CLASSES), dim=1)
    for i in range(3):
        logits = model(x[i:i + 1])
        p = F.softmax(logits, dim=1).detach()
        grads = []
        for loss in (F.cross_entropy(logits, y[i:i + 1]),
                     -(q[i:i + 1] * F.log_softmax(logits, dim=1)).sum()):
            g = torch.autograd.grad(loss, [model.classifier.weight,
                                           model.classifier.bias], retain_graph=True)
            grads.append(torch.cat([t.flatten() for t in g]))
        got = F.cosine_similarity(grads[0], grads[1], dim=0)
        e_h = p - F.one_hot(y[i:i + 1], NUM_CLASSES).float()
        e_s = p - q[i:i + 1]
        expect = F.cosine_similarity(e_h.flatten(), e_s.flatten(), dim=0)
        assert abs(got - expect) < 1e-5, (got, expect)
    print("           ok  (cos(gradHL, gradSL) == cos(p-y, p-q), incl. bias)")

    print("[selftest] 2. pred score vs manual loop on synthetic trajectories")
    K, n, C = 4, 6, 5
    P = F.softmax(torch.randn(K, n, C), dim=-1)
    q = F.softmax(torch.randn(n, C), dim=-1)
    a = -(P - q).norm(dim=-1).mean(0)
    for i in range(n):
        manual = -np.mean([float((P[k, i] - q[i]).norm()) for k in range(K)])
        assert abs(a[i] - manual) < 1e-6
    print("           ok")

    print("[selftest] 3. alpha extremes reproduce pure-A / pure-R orderings")
    A = torch.randn(10)
    R = torch.randn(10)
    top_a = torch.argsort(-(1.0 * zscore(A)), stable=True)[:3]
    top_r = torch.argsort(-(1.0 * zscore(R)), stable=True)[:3]
    assert torch.equal(top_a, torch.argsort(-A, stable=True)[:3])
    assert torch.equal(top_r, torch.argsort(-R, stable=True)[:3])
    print("           ok")
    print("[selftest] all checks passed")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "select", "eval"])
    ap.add_argument("--ipc", type=int, choices=[10, 50])
    ap.add_argument("--align", choices=["grad", "pred", "2run"], default="grad")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mode", choices=["topk", "fl"], default="topk")
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
    assert args.ipc, "--ipc required"
    if args.cmd == "select":
        run_select(args, device)
    elif args.cmd == "eval":
        name = run_select(args, device)
        eval_selected(name, args.ipc, args.seeds, device)


if __name__ == "__main__":
    main()
