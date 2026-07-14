# R-CAD and SHARP: relational selection beyond CA2D on CIFAR-100 (hard-label)

Two novel selection methodologies built on the `ca2d.py` reproduction stack
(CIFAR-100, ConvNet-D3, hard-label protocol of [HT Tab.4], seeds 42/43/44, DSA,
300 epochs). All baselines are the locally cached reproductions; no baseline was
re-run. Implementation: `rcad.py` (method i), `sharp.py` (method ii), `probe.py`
(shared instrumentation). Every number below is a mean ± population std over the
three eval seeds under the identical protocol.

**Headline.** R-CAD (`--kernel feat --selector fl`) reaches **26.08 ± 0.21 (IPC 10)**
and **41.61 ± 0.20 (IPC 50)** — above every published coreset-selection and
realistic-image DD baseline on this benchmark ([HT Tab.6/11]: best prior selection =
K-centers 25.04 / 38.64; CAD-Prune 22.95 / 37.87; CA2D 20.95 / 34.37), and it does so
under a local protocol on which the published baselines reproduce 0.7–2.2 points
*lower* than their paper numbers. Among all methods in [HT Tab.6] only the bi-level
synthesis methods TM (38.18 / 46.32) and DM (29.23 / 42.32) remain above at IPC 10;
at IPC 50 R-CAD (41.61) passes DC (30.56), closes to within 0.7 of DM (42.32), and
only TM remains clearly ahead — methods that synthesize optimized images and are
shown in [HT Sec.4] not to scale beyond small settings.

---

## 1. Setup and bars

Compute-matched scoring run [HT Sec.5]: K = 6 epochs (IPC 10) / 30 epochs (IPC 50) of
full-dataset training under the downstream HL recipe compressed to K epochs
[ca2d.py ASSUME 1]; per-epoch EL2N scores S ∈ R^{K×50000} [DD Def.2.3]; CAD per
[HT Eq.1–2] with J = K−W, W = 2 [ca2d.py ASSUME 9].

| method | published [HT Tab.6/11] | local repro |
|---|---|---|
| TM (bi-level synthesis) | 38.18 / 46.32 | — |
| DM / DC (bi-level synthesis) | 29.23 / 42.32 · 28.42 / 30.56 | — |
| K-centers | 25.04 / 38.64 | not reproducible as farthest-point (§6) |
| CAD-Prune | 22.95 / 37.87 | 21.92 ± 0.21 / 37.14 ± 0.08 |
| CA2D (factor 1) | 20.95 / 34.37 | 20.94 ± 0.18 / 35.80 ± 0.22 |
| Random Real | 18.64 / 34.66 | 16.44 ± 0.05 / 33.42 ± 0.18 |

## 2. Method i — R-CAD (Relational Compute-Aware Difficulty)

CAD-Prune ranks each image's learning trajectory in a vacuum and keeps the per-class
top-IPC: a *pointwise* rule with no notion of what the rest of the selected set
already contains, so it happily spends the tiny budget on near-duplicates of the same
difficult mode. R-CAD replaces the ranking with a set-level objective: per class c
with candidate pool P_c (all 500 class images), select

$$A_c^\* \;=\; \arg\max_{A \subset P_c,\ |A| = m}\; F_c(A), \qquad
F_c(A) \;=\; \sum_{i \in P_c} \mathrm{CAD}_i \cdot \max_{j \in A} \kappa(i,j),$$

where κ(i,j) = max(0, cos(E_i, E_j)) is a similarity kernel over per-image embeddings
E. Reading: every pool image i demands to be *represented* by some selected exemplar
with a similar profile, and the demand is weighted by the sample's compute-aligned
difficulty CAD_i. F_c is a weighted facility-location function — monotone submodular
— so greedy selection carries the (1−1/e) guarantee (Nemhauser et al., 1978); with
n = 500 the greedy is exact enough and instant.

The two strongest prior baselines are the two degenerate poles of this single objective:

- **Difficulty-only pole.** Drop the relational term (greedy rule
  s(i|A) = CAD_i·(1 − λ·max_{j∈A} κ(i,j)) at λ = 0): this reproduces the CAD-Prune
  selection *index-for-index* (selftest 5 in `rcad.py`).
- **Coverage-only pole.** Set CAD_i ≡ 1: plain facility location, the
  K-centers/coverage family.

Kernels (`--kernel`): `feat` = L2-normalized observer penultimate features (static
geometry); `softtraj` = concatenated softmax-error trajectories
[p_1−y; …; p_K−y] ∈ R^{K·100} from `probe.py`; `traj` = Pearson correlation of the
cached scalar EL2N rows (zero extra compute).

### Results (IPC 10 / IPC 50, local protocol)

| config | IPC 10 | IPC 50 |
|---|---|---|
| **R-CAD: feat kernel, CAD-weighted FL** | **26.08 ± 0.21** | **41.61 ± 0.20** |
| feat, dcad λ=0.5 | 24.24 ± 0.08 | 38.49 ± 0.08 |
| feat, FL, uniform weights (coverage-only pole) | 23.93 ± 0.14 | 37.55 ± 0.31 |
| softtraj, dcad λ=0.5 | 22.52 ± 0.11 | 37.60 ± 0.32 |
| softtraj, FL | 22.18 ± 0.28 | — |
| softtraj, dcad λ=1 | 21.43 ± 0.34 | — |
| traj, dcad λ=0.5 | 22.02 ± 0.25 | — |
| traj, FL | 18.71 ± 0.20 | — |
| CAD-Prune (= dcad λ=0, difficulty-only pole) | 21.92 ± 0.21 | 37.14 ± 0.08 |

**The interior beats both poles, at both IPCs.** Relative to coverage-only, CAD
demand weights add +2.15 (IPC 10) and +4.06 (IPC 50); relative to difficulty-only,
the coverage term adds +4.16 and +4.47. Neither ingredient explains the result alone
— the product structure (difficulty-weighted coverage) is causal.

**Kernel choice.** Static feature geometry (feat) decisively beats trajectory kernels
for the *relational* term. Scalar EL2N trajectories (K = 6 points at IPC 10) are too
coarse (traj FL: 18.71); full softmax-error trajectories help (22.18) but remain
noisy. The dynamics information is already carried by the CAD weights; the kernel's
job is geometric coverage, which the observer features express best. (The λ-greedy
still lifts the trajectory kernels over CAD-Prune: dynamics-redundancy is real, just
weaker than feature-space redundancy.)

## 3. Method ii — SHARP (Soft-Hard Alignment + Representativeness Pruning)

Goal: select images whose hard label alone already delivers the teacher's soft-label
supervision — the "info-rich" images — intersected with representativeness.

**Why the signal must be dynamic.** Statically, the per-logit gradient difference
between hard- and soft-label training is (p−y) − (p−q) = q − y: independent of the
student. Matching on it collapses to teacher-confidence selection, which is RDED's
own criterion [RDED Eq.8] — nothing new. On the trajectory, however, the last-layer
gradients under the two labelings factor as (p_k−t) ⊗ [φ(x); 1] for t ∈ {y, q}, so

$$\cos\!\big(\nabla \ell_{HL},\, \nabla \ell_{SL}\big)\Big|_{\theta_k}
\;=\; \cos\big(p_k - y,\; p_k - q\big),$$

student-dependent and free to evaluate from the probe trajectories (identity verified
in selftest 1 of `sharp.py`, including the bias term). SHARP's main score is

$$A_{\text{grad}}(i) \;=\; \tfrac{1}{K}\sum_{k=1}^{K}
\cos\big(p_k(i) - y_i,\; p_k(i) - q_i\big),$$

with teacher q = softmax(observer(x)) [ASSUME P1: RDED observer, τ = 1]. Ablations:
A_pred = −mean_k‖p_k − q‖ (EL2N-SL [HT Def.1] along the HL run) and A_2run =
−mean_k‖p_k^{HL} − p_k^{SL}‖ against an exact counterfactual twin trained on q with
an identical RNG stream (`probe.py --twin`, [ASSUME P2]). Representativeness: either
top-k on α·z(A) + (1−α)·z(R) with R = cosine to the normalized class centroid
(`topk`), or — following the coverage lesson of §2 — facility location with
per-sample demand weights given by the alignment score (`fl`).

### Results (IPC 10 unless noted)

| config | top-1 |
|---|---|
| **SHARP-FL: coverage weighted by A_grad** | **24.18 ± 0.12** · IPC 50: 38.09 ± 0.35 |
| A_grad alone (α=1, top-k) | 22.02 ± 0.16 · IPC 50: 36.54 ± 0.38 |
| A_grad + centroid-R (α=0.5) | 20.78 ± 0.19 |
| centroid-R alone (α=0) | 20.51 ± 0.27 |
| A_pred + centroid-R (α=0.5) | 19.82 ± 0.10 |
| 2run counterfactual (α=1) | 18.92 ± 0.12 |
| A_pred alone (α=1) | 17.22 ± 0.13 |

The user-requested component ablation is clean: (a) the *gradient-alignment* score is
the value carrier — alone it already matches CAD-Prune (22.02 vs 21.92) while
selecting almost disjoint data (Jaccard 0.04, §5); (b) "representative" must be
operationalized as *coverage*, not centroid proximity — mixing in the centroid score
hurts (20.78 < 22.02) while switching to alignment-weighted facility location helps
by +2.16 (24.18), beating published CAD-Prune (22.95); at IPC 50 SHARP-FL reaches
38.09 ± 0.35, above local CAD-Prune (37.14) and on par with its published 37.87. (c) Among matching signals,
the cheap single-run cosine beats both the distance form and the literal two-run
behavioral matching: the norm-based variants (pred, 2run) reward samples whose
predictions already sit on the teacher — i.e. easy prototypes — whereas the cosine
normalizes magnitude away and asks only whether the hard label *pushes in the
teacher's direction while the sample is still being learned*.

## 4. Synthesis variant (vs CA2D head-to-head)

Replacing CA2D's CAD-ranked pool with the R-CAD (feat/FL) pool of the same size
(mipc = 300 [RDED Alg.1]), then the stock RDED crop/select/stitch:

| method | IPC 10 | IPC 50 |
|---|---|---|
| R-CAD-2D (ours) | 21.42 ± 0.35 | 35.98 ± 0.23 |
| CA2D (local) | 20.94 ± 0.18 | 35.80 ± 0.22 |

R-CAD-2D > CA2D like-for-like, but both lose badly to their pruning counterparts:
at 32×32 the crop/stitch pipeline destroys more information than pool quality adds
(consistent with [HT Tab.11]'s own CA2D < CAD-Prune on CIFAR-100). On this benchmark
the synthesis stage is a net negative; the pruning form is the method.

## 5. Do the methods select different data?

Score-level: Spearman(CAD, A_grad) = 0.39 over all 50k train images. Set-level
Jaccard overlaps at IPC 10: R-CAD∩CAD-Prune = 0.15, R-CAD∩coverage-only = 0.28,
SHARP∩CAD-Prune = 0.04, SHARP∩R-CAD = 0.05. R-CAD genuinely interpolates its two
poles rather than re-ranking either; SHARP taps a near-orthogonal signal at matching
accuracy — a natural future combination (SHARP alignment × CAD demand weights in one
FL objective).

## 6. What did not reproduce: "K-centers 25.04"

[HT Tab.6] cites K-centers via DeepCore/DCBench without small-scale specifics.
Farthest-point traversal (Sener & Savarese) fails catastrophically at this budget in
both metrics we tried — normalized cosine: 8.13 ± 0.18; DeepCore-style raw Euclidean:
14.23 ± 0.25 (it selects per-class outliers at IPC 10). The "cluster centers" reading
(k-means++ medoids [ASSUME R2]) gives 22.46 ± 0.06, and the strongest coverage-only
method we found is uniform facility location at 23.93 ± 0.14 — still 1.1 below the
published K-centers number on a protocol where every other baseline reproduces
*lower* than published. We therefore treat 25.04 as the bar and note that R-CAD
clears it on the stricter local protocol (+1.04 / +2.97).

## 7. Compute accounting and determinism

- **R-CAD (feat/FL), the headline method, costs nothing beyond CAD-Prune**: it reuses
  the cached scoring run for CAD and adds one observer forward pass (features) plus a
  CPU greedy — no extra training. Compute-matched comparison with CAD-Prune is exact.
- softtraj/SHARP variants consume the `probe.py` replica of the scoring run (same
  budget as the original run; only needed because the original stored scalars — a
  from-scratch implementation logs P during the single scoring run). The 2run
  ablation doubles the scoring budget and is reported as such.
- Both probe replicas reproduced the cached EL2N matrices **bitwise** (asserted in
  `probe.py`), so every selection is a pure deterministic function of the already
  published scoring run; all sets/results are path-keyed by full config and the
  baseline caches were never touched.

## 8. Assumptions added on top of ca2d.py [ASSUME 1–9]

- [ASSUME P1] SHARP teacher = RDED observer checkpoint, τ = 1, ImageNet-stat path.
- [ASSUME P2] the SL twin keeps the HL optimizer/schedule/augmentation so the label
  is the only difference (deliberate divergence from [HT Tab.4]'s SL recipe).
- [ASSUME R1] k-center farthest-point starts at the class medoid.
- [ASSUME R2] "K-centers as cluster centers" = per-class k-means++ / Lloyd, nearest
  sample per centroid, deduplicated.

## 9. Generalization benchmarks: TinyImageNet and fixed soft labels

Both methods were carried, with hyperparameters frozen from CIFAR-100, to the
paper's remaining small-scale bars: TinyImageNet (ConvNet-D4, [HT Tab.2]) and the
fixed-SL regime ([HT Tab.4 small-scale SL]: teacher-assigned soft labels, KL T=20,
AdamW, cosine; teacher = per-dataset RDED observer [ASSUME X1/X2]). CAD-Prune and
CA2D were reproduced on TinyImageNet under the identical protocol (CA2D via RDED's
own TIN recipe: factor=1, 5 crops, mipc=300). Camera-ready LaTeX in `tex/`
(`tab2_tinyimagenet.tex`, `tab6_cifar100.tex`); live view: `python xarch.py table`.

Headlines (mean ± std, seeds 42-44; †/‡ = Welch p<0.05/0.01 vs strongest local
baseline in the column):

- **TinyImageNet HL**: R-CAD 13.14±0.54 (IPC 10; best local method, above published
  K-centers 11.38 and DM 13.51's level) and 27.66±0.29 (IPC 50; statistical tie with
  CAD-Prune 27.83, p=0.73 — coverage is neutral in this difficulty-dominated regime,
  where coverage-only collapses to 19.32). CA2D reproduces at 10.44 / 23.77 — behind
  its own CAD-Prune at both IPCs. SHARP-FL (11.52 / 21.76) beats CA2D and coverage at
  IPC 10 but trails CAD-Prune on TIN: the alignment signal is dataset-dependent.
- **Fixed-SL panels (both datasets)**: quality differences compress exactly as
  [HT Sec.3] predicts, but the ordering survives — R-CAD is the best local method in
  all four SL columns (CIFAR 31.02 / 46.41†; TIN 19.20† / 34.21‡), and CA2D falls
  *below* Random Real under SL on CIFAR (40.55 vs 43.88 at IPC 50).
- Local protocol anchors: Random reproduces below the published Random in every
  column (e.g. TIN HL 6.33 vs 6.88), so all local-vs-published comparisons are
  conservative. Published SL rows sit systematically above the local SL protocol
  (different teacher; [ASSUME X2]) — cross-protocol SL comparisons are indicative
  only.
- Cost accounting (`python xarch.py timing`): TIN scoring runs 174 s (K=6) / 1113 s
  (K=30); R-CAD/SHARP-FL selection ≤14 s per IPC on top — the same order as
  CAD-Prune's own selection; CA2D synthesis adds ~78 s.

A cross-architecture transfer study (DC-bench MLP/ResNet-18/ResNet-152) was begun
and then descoped at the user's direction after a GPU failure; its cells were
removed from the result cache and are not reported.

## 10. Reproduction

```
python probe.py --ipc 10 --twin && python probe.py --ipc 50 --twin
python rcad.py  eval --ipc 10 --kernel feat --selector fl        # 26.08
python rcad.py  eval --ipc 50 --kernel feat --selector fl        # 41.61
python sharp.py eval --ipc 10 --align grad --alpha 1.0 --mode fl # 24.18
python tin.py   all-select --ipc 10 && python tin.py all-select --ipc 50
python xarch.py grid --dataset tin --ipc 10 --archs convnet --regime hl
python xarch.py table   # paper-style tables      xarch.py tex  # LaTeX
```

## 11. bench.py: the four CVPR tables (2026-07-13)

`bench.py` supersedes the xarch tables for the paper: 4 tables (conv3-cifar100,
resnet18-cifar100, conv4-tinyimagenet, resnet18-tinyimagenet), columns HL / SL /
KD+SL x IPC {10,50,100}, rows Random / EL2N-Best / CAD-Prune / SHARP / R-CAD |
RDED / CA2D | Full-Dataset. All protocol provenance and the implicit choices
[B1-B10] are in the bench.py docstring. Cache-first: every existing HL/SL cell
and scoring blob is alias-resolved, never recomputed (selftest: 20/20 sets,
40/40 result cells). KD+SL runs RDED's official validation code unmodified in a
pinned subprocess (argument.py parity selftest-verified, incl. the hidden
cifar100-conv3 bs=25/lr=2e-3 override).

**Teacher top-1 "drift" resolved.** The released RDED checkpoints store their
eval protocol (`val_resize_size=36/crop 32` cifar, `73/64` tiny = torchvision
references' `size*8//7`). Under it, all four benchmark teachers reproduce the
RDED README *exactly* (72.27 / 61.27 / 61.98 / 49.73, drift +0.00; measured on
the artifacts/rded trees). RDED's own validation transform
(validation/main.py:154, `size//7*8` = identity at 32px, 72 at 64px) is a
*different* protocol under which the same checkpoints score lower on CIFAR
(conv3 59.67, rn18-mod 70.93) and equal on TIN (50.15 / 62.02). `teachers.py
verify --protocol {ckpt,rded}` measures either. There is no checkpoint issue.

```
python bench.py selftest && python bench.py prepare
python bench.py cell --ds cifar100 --arch conv --method rded --regime kd --ipc 10  # anchor ~48.1
python bench.py run --table conv-cifar100    # then conv-tin, rn18-cifar100, rn18-tin
python bench.py table && python bench.py tex # renders with '--' for pending cells
```
