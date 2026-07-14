#!/usr/bin/env python
"""Instrumented deterministic replica of the ca2d.py scoring run ("probe").

Re-executes the compute-matched scoring run of ca2d.run_score [HT Sec.5, ASSUME 1]
with the identical seed and operation sequence, but a richer epoch hook that records,
on clean (non-augmented) images:

    P[k] = softmax(f(theta_k, x))  in fp16, shape (K, 50000, 100)

plus the observer teacher's soft labels q = softmax(observer(x_imnet)) (fp16, tau=1
[ASSUME P1]) and the final scoring-model checkpoint. This single artifact powers both
R-CAD (rcad.py, error-trajectory kernels) and SHARP (sharp.py, soft/hard alignment).

Determinism contract: the scoring run is bitwise reproducible (seed 0, cudnn
deterministic, same GPU model), so the EL2N trajectories recomputed here must equal
the cached S in scores/cad_ipc{ipc}.pt. This is asserted; a probe that cannot
reproduce the cached scores refuses to save.

--twin additionally trains the soft-target counterfactual twin: identical seed, init,
batch order and DSA draws (the loss target consumes no RNG), with the hard labels y
replaced by the teacher distribution q in the cross-entropy (F.cross_entropy with
probabilistic targets). Saved to scores/probe_twin_ipc{ipc}.pt. [ASSUME P2] The twin
deliberately keeps the HL optimizer/schedule (SGD, StepLR, DSA) instead of the paper's
SL protocol [HT Tab.4] so that the *only* difference between the runs is the label,
making the pair an exact counterfactual.

Usage:
    python probe.py --ipc 10 [--twin] [--device cuda:1]
"""

import argparse
import os

import torch
import torch.nn.functional as F

import ca2d
from ca2d import (CIFAR_MEAN, CIFAR_STD, IMNET_MEAN, IMNET_STD, NUM_CLASSES, N_TRAIN,
                  SCORE_DIR, compute_matched_epochs, convnet_d3, el2n_scores,
                  get_observer, hl_train, load_cifar100, set_seed, to_norm_tensor)


@torch.no_grad()
def collect_probs(model, x, device, batch=2048):
    """Full softmax on clean images, fp16 on CPU. No RNG consumed."""
    model.eval()
    out = torch.empty(len(x), NUM_CLASSES, dtype=torch.float16)
    for i in range(0, len(x), batch):
        out[i:i + batch] = F.softmax(model(x[i:i + batch].to(device)), dim=1).half().cpu()
    return out


def teacher_soft_labels(device):
    """q = softmax(observer(x)) on ImageNet-stat images [RDED observer path, ASSUME P1]."""
    observer = get_observer(device, allow_train_fallback=False)
    xtr_u8, _, _, _ = load_cifar100()
    x_im = to_norm_tensor(xtr_u8, IMNET_MEAN, IMNET_STD)
    return collect_probs(observer, x_im, device)


def scoring_run_paths(ipc):
    return (os.path.join(SCORE_DIR, f"cad_ipc{ipc}.pt"),
            os.path.join(SCORE_DIR, f"probe_ipc{ipc}.pt"),
            os.path.join(SCORE_DIR, f"probe_twin_ipc{ipc}.pt"))


def run_probe(ipc, device, twin=False):
    cad_path, probe_path, twin_path = scoring_run_paths(ipc)
    assert os.path.exists(cad_path), (
        f"{cad_path} missing: the baseline scoring run must already be cached "
        "(never re-derive baselines here)")
    base = torch.load(cad_path, map_location="cpu")
    K = compute_matched_epochs(ipc)
    step_epoch = base["step_epoch"]
    assert base["K"] == K and base["seed"] == 0 and base["aug"] == "dsa"

    q = None
    if not os.path.exists(probe_path) or (twin and not os.path.exists(twin_path)):
        q = teacher_soft_labels(device)  # RNG-free; computed before set_seed anyway

    if os.path.exists(probe_path):
        print(f"[probe] cached: {probe_path}")
    else:
        # --- replica of ca2d.run_score's exact op sequence [determinism contract] ---
        set_seed(0)
        xtr_u8, ytr, _, _ = load_cifar100()
        x = to_norm_tensor(xtr_u8, CIFAR_MEAN, CIFAR_STD)
        y = torch.from_numpy(ytr).long()
        y1h = F.one_hot(y, NUM_CLASSES).float()
        model = convnet_d3("instance")
        init_sig = float(model.classifier.weight.sum())
        P = torch.empty(K, N_TRAIN, NUM_CLASSES, dtype=torch.float16)
        S_check = torch.empty(K, N_TRAIN)

        def hook(ep, m):
            S_check[ep] = el2n_scores(m, x, y1h, device)  # identical fn => bitwise
            P[ep] = collect_probs(m, x, device)
            print(f"[probe]   epoch {ep + 1}/{K} EL2N mean={S_check[ep].mean():.4f}",
                  flush=True)

        print(f"[probe] ipc={ipc}: K={K} epochs, StepLR@{step_epoch} (HL replica)")
        hl_train(model, x, y, device, epochs=K, step_epoch=step_epoch, aug="dsa",
                 epoch_hook=hook)

        if torch.equal(S_check, base["S"]):
            print("[probe] determinism check: EL2N == cached S (bitwise)")
        else:
            diff = (S_check - base["S"]).abs().max().item()
            assert diff < 1e-5, f"probe failed to reproduce cached S (max diff {diff})"
            print(f"[probe] determinism check: EL2N ~= cached S (max diff {diff:.2e})")

        blob = {"P": P, "q": q, "S_check": S_check, "K": K, "step_epoch": step_epoch,
                "seed": 0, "aug": "dsa", "init_sig": init_sig,
                "final_state": {k: v.cpu() for k, v in model.state_dict().items()}}
        torch.save(blob, probe_path)
        print(f"[probe] saved {probe_path}")

    if not twin:
        return
    if os.path.exists(twin_path):
        print(f"[probe] cached: {twin_path}")
        return
    # --- soft-target counterfactual twin [ASSUME P2] ---
    if q is None:
        q = torch.load(probe_path, map_location="cpu")["q"]
    set_seed(0)
    xtr_u8, ytr, _, _ = load_cifar100()
    x = to_norm_tensor(xtr_u8, CIFAR_MEAN, CIFAR_STD)
    model = convnet_d3("instance")
    hl_sig = torch.load(probe_path, map_location="cpu")["init_sig"]
    assert abs(float(model.classifier.weight.sum()) - hl_sig) < 1e-6, \
        "twin init differs from HL-run init (RNG sequence broken)"
    Q = q.float()  # (N, C) probabilistic targets; hl_train's CE accepts them
    P_twin = torch.empty(K, N_TRAIN, NUM_CLASSES, dtype=torch.float16)

    def hook_t(ep, m):
        P_twin[ep] = collect_probs(m, x, device)
        print(f"[probe]   twin epoch {ep + 1}/{K}", flush=True)

    print(f"[probe] ipc={ipc}: twin run on soft targets q (identical RNG stream)")
    hl_train(model, x, Q, device, epochs=K, step_epoch=step_epoch, aug="dsa",
             epoch_hook=hook_t)
    torch.save({"P": P_twin, "K": K, "step_epoch": step_epoch, "seed": 0}, twin_path)
    print(f"[probe] saved {twin_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ipc", type=int, required=True, choices=[10, 50])
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_probe(args.ipc, device, twin=args.twin)


if __name__ == "__main__":
    main()
