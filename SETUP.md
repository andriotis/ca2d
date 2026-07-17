# Experimental setup: the four-table selection/distillation benchmark

This document describes the experimental setup driven by `bench.py`, for review.
It has two goals: (1) to make explicit **every implicit choice** we take where
the source papers are silent or ambiguous, and (2) to explain our two novel
selection methods, **R-CAD** and **SHARP**, both mathematically and intuitively,
*exactly as the benchmark runs them*.

References are abbreviated as: **[HT]** = *Rethinking Dataset Distillation: Hard
Truths About Soft Labels*; **[RDED]** = *On the Diversity and Realism of
Distilled Dataset* (RDED); **[DD]** = *Deep Learning on a Data Diet* (EL2N).

---

## 1. Overview

The benchmark fills **four tables**, one per (dataset, architecture) pair:

| table | dataset | architecture |
|---|---|---|
| `conv-cifar100` | CIFAR-100 | ConvNet-D3 (conv3) |
| `rn18-cifar100` | CIFAR-100 | ResNet-18 |
| `conv-tin` | TinyImageNet | ConvNet-D4 (conv4) |
| `rn18-tin` | TinyImageNet | ResNet-18 |

Each table has the same layout, mirroring the structure of [HT Tab.1]:

- **Columns** = three supervision regimes **× three budgets**:
  `{HL, SL, KD+SL} × IPC {10, 50, 100}` (IPC = images per class).
- **Rows**, in two groups:
  - *Coreset selection* (pick real images): Random Real, EL2N-Best, CAD-Prune,
    **SHARP (ours)**, **R-CAD (ours)**.
  - *Dataset distillation* (synthesize images): RDED, CA2D, **SHARP-2D (ours)**,
    **R-CAD-2D (ours)**. The 2D variants replace CA2D's CAD-ranked candidate pool
    with our SHARP / R-CAD selection at the same budget, then run RDED's stock
    crop/select/stitch.

Every cell reports **mean ± population std over three training seeds
{42, 43, 44}**, with one-sided **Welch p-values** against the strongest local
baseline in the column. A compute-matched Full-Dataset reference can be produced
on demand but is **not shown in the tables**.

---

## 2. Datasets and budgets

| dataset | classes | image size | train images (N) | candidate pool / class |
|---|---|---|---|---|
| CIFAR-100 | 100 | 32×32 | 50,000 | 500 |
| TinyImageNet | 200 | 64×64 | 100,000 | 500 |

Budgets are **IPC ∈ {10, 50, 100}**.

**Compute-matched scoring budget.** Following [HT Sec.5], the trajectory-based
scores (EL2N, CAD) are computed from a full-dataset training run whose length is
*compute-matched* to a downstream run on the coreset:

$$K \;=\; \frac{300 \cdot \mathrm{IPC} \cdot n_{\text{class}}}{N} \;=\; 6 \,/\, 30 \,/\, 60 \text{ epochs}$$

for IPC 10 / 50 / 100 respectively (`compute_K`). The scoring run's StepLR
milestone is compressed proportionally, `round(K·151/300)`.

---

## 3. Training protocols (the three regimes)

The three columns are three different ways of supervising the downstream student.
Recipes are cited to their source and reproduced verbatim by the code.

### HL — hard labels [HT Tab.4, small-scale HL]
Standard one-hot cross-entropy training (`hl_train`):
- **SGD**, lr 0.01, momentum 0.9, weight decay 5e-4, batch 256.
- `MultiStepLR` milestone at epoch **151**, γ = 0.1; **300 epochs**.
- **DSA** differentiable augmentation.

> Weight decay, momentum and γ are not listed in [HT Tab.4] (which gives only
> epochs / lr / step / batch); we use the standard DC/DSA values. This is
> flagged in `ca2d.py` as ASSUME 1–2.

### SL — fixed soft labels [HT Tab.4, small-scale SL]
Each selected image carries a **fixed, precomputed teacher soft label**; the
student matches it (`sl_train`):
- Teacher soft label $q^{(T)} = \mathrm{softmax}(\text{observer}(x)/T)$, **KL
  divergence, temperature T = 20**.
- **AdamW**, lr 1e-3, cosine schedule, **300 epochs**, batch 256, DSA.

### KD+SL — on-the-fly knowledge distillation
RDED's **official** `validation/main.py` is run **unmodified** in a pinned
subprocess worker (zero edits to the RDED repo):
- Teacher soft labels computed **on the fly** over cutmix-mixed batches, **KL
  divergence, T = 20**.
- AdamW, quarter-cosine schedule, 300 epochs, RDED's official batch size and
  learning rate — **including the hidden `cifar100-conv3` override
  `bs = 25, lr = 2e-3`** (`argument.py:309–315`).

Small-scale KD+SL is **not defined by [HT]**; adopting RDED's official validation
code for it is implicit choice **[B1]** (see §6).

---

## 4. Students, teachers, and observers

This is the single most important implicit choice to surface, because the
student network **differs by column**.

| role | HL / SL columns | KD+SL column |
|---|---|---|
| **student** (the trained/evaluated net) | DC-Bench variants: **instance-norm** ConvNet-D3/D4, DC-Bench ResNet-18 (`build_student`) | RDED's **own** models: **batch-norm** conv3/conv4, `resnet18_modified` (`rded_arch`) |

- **HL / SL students** are the DC-Bench architectures ([HT Tab.7/8 variant]),
  chosen so the numbers are comparable to [HT]'s own selection tables.
- **KD students** are RDED's models, because the KD column runs RDED's official
  code, which instantiates those architectures. This split is forced by [B1] and
  documented as implicit choice **[B7]** — it is footnoted in the tables.
- **Teacher / observer** (both regimes): the **released RDED checkpoint** of the
  table's architecture, loaded frozen (`build_teacher`). It supplies (a) the soft
  labels $q$ for SL/KD, and (b) the **penultimate features** used by our
  selection methods. `resnet18_modified` uses RDED's small-image stem (3×3
  stride-1 conv1, no maxpool).

---

## 5. Selection provenance

All trajectory-based scores and our selections are computed **once** per
`(dataset, table-architecture, IPC)` from a **single seed-0 compute-matched HL
scoring run** (`run_score`). That one run instruments, in a single pass:

- per-epoch **EL2N** scores $S \in \mathbb{R}^{K \times N}$ [DD Def.2.3],
- the fp16 **softmax trajectories** $P \in \mathbb{R}^{K \times N \times C}$,
- the frozen **teacher soft labels** $q$,
- and hence **CAD** [HT Eq.1–2] with window $J = K-W,\ W = 2$.

The **same selected subset is then evaluated under all three regimes** (HL, SL,
KD+SL) — the [HT] convention. Using a single seed-0 scoring run (rather than
averaging several) is implicit choice **[B8]**; [HT] is silent on averaging.

---

## 6. Implicit choices [B1]–[B10]

These are the decisions we take where the papers are undefined or ambiguous. Each
is annotated at its point of use in the code.

| ID | Decision | Why / citation |
|---|---|---|
| **B1** | Small-scale KD+SL = RDED's official validation code, run unmodified. | [HT] does not define small-scale KD+SL; RDED's is the canonical on-the-fly KD recipe. |
| **B2** | IPC-100 KD batch size: CIFAR-100 = 200, TinyImageNet = 100. | RDED has no IPC-100 rule; CIFAR uses [HT Tab.4] large-scale HL IPC-100 batch, TIN uses RDED's single all-IPC rule (`argument.py:252–254`). |
| **B3** | EL2N = **mean over 10 runs** (seeds 1000–1009) of $\lVert \mathrm{softmax}-\text{onehot}\rVert_2$ at **epoch 20** of the truncated HL recipe, table-arch student. | [DD Def.2.3]; DD README "≥ 10 runs"; epoch-20 = DD Fig.1 early-epoch convention. |
| **B4** | EL2N-Best window: grid of start offsets {0,10,…,90}% of (pool−IPC) over the per-class descending EL2N ranking; **winner chosen by TEST accuracy**; search at seed 42 only, winner re-run at 3 seeds (search seed reused). | [HT] implicitly selects the window by test accuracy — flagged in the paper text. |
| **B5** | Coresets exported as **PNG** for the KD path (avoids RDED's JPEG re-encoding, which costs 1.6–3.1 pp at 32 px). Synthesized sets keep official JPEG. | Measured teacher-top-1 loss under JPEG (`teachers.py`). |
| **B6** | Full-Dataset rows are compute-matched: epochs = $300 \cdot \mathrm{IPC} \cdot n_{\text{class}}/N$ = 6/30/60, schedules compressed proportionally; in KD the best-acc window degenerates to the final-epoch validation (`validation/main.py:179–183` fires at the last epoch iff `re_epochs >= 6`). **Not shown in tables.** | [HT Sec.5] compute-matching. |
| **B7** | Per-column student convention: DC-Bench students for HL/SL, RDED students for KD. **Footnoted in tables.** | Comparability to [HT Tab.2/6/11] for HL/SL; forced by B1 for KD. |
| **B8** | Scoring run = single seed-0 run. | [HT] is silent on scoring-run averaging. |
| **B9** | RDED synthesis via official `synthesize/main.py`; its pool-shuffle / crop RNG is unseeded upstream — the worker seeds python/numpy/torch with 0. | Determinism; stock RDED leaves these unseeded. |
| **B10** | Random rows: one fixed class-balanced draw (`RandomState(0)`) × 3 training seeds. | Isolates training-seed variance from draw variance. |

---

## 7. The two novel selection methods

Both methods are **coreset selection** rules: they pick real images. Crucially,
inside the benchmark **they share one selection engine and differ only in a
single ingredient** — the per-sample *demand weight*.

### 7.0 The shared engine: difficulty-/alignment-weighted coverage

For each class $c$ with candidate pool $P_c$ (all 500 class images), we build a
similarity kernel over **L2-normalized observer penultimate features**
$\hat\varphi_i$,

$$\kappa(i,j) \;=\; \max\!\big(0,\ \cos(\hat\varphi_i, \hat\varphi_j)\big),$$

and select a subset $A$ of size = budget by maximizing a **weighted
facility-location** objective,

$$A_c^{*} \;=\; \arg\max_{A \subset P_c,\ |A|=b}\ F_c(A),
\qquad F_c(A) \;=\; \sum_{i \in P_c} w_i \cdot \max_{j \in A} \kappa(i,j).$$

$F_c$ is **monotone submodular**, so greedy selection carries the classic
$(1-1/e)$ approximation guarantee (Nemhauser et al., 1978); with $|P_c| = 500$
the greedy is effectively exact and instant (`greedy_fl`).

**Reading:** every pool image demands to be *represented* by some selected
exemplar with a similar feature profile; $w_i$ scales how loudly image $i$ demands
representation. The two methods are exactly this objective with two different
choices of $w$:

| method | demand weight $w_i$ |
|---|---|
| **R-CAD** | $w_i = \mathrm{CAD}_i$ — compute-aware difficulty |
| **SHARP** | $w_i =$ shifted per-class **z-score of** $A_{\text{grad}}(i)$ — soft–hard gradient alignment |

The `-2D` variants run the identical selection at budget `RDED_MIPC = 300`, then
feed that pool through RDED's stock crop/select/stitch (`RDED_NUM_CROP = 5`,
`RDED_FACTOR = 1`).

### 7.1 Method i — R-CAD (Relational Compute-Aware Difficulty)

**Intuition.** CAD-Prune ranks each image's learning trajectory *in a vacuum* and
keeps the per-class top-IPC. This is a **pointwise** rule with no notion of what
the rest of the selected set already contains, so under a tiny budget it happily
spends its picks on **near-duplicates of the same difficult mode**. R-CAD makes
the decision **set-level**: it keeps images that are individually informative
(high CAD) *and* jointly non-redundant, i.e. **difficulty-weighted coverage** of
the class.

**Math.** R-CAD is the shared objective with $w_i = \mathrm{CAD}_i$:

$$A_c^{*} \;=\; \arg\max_{|A|=\mathrm{IPC}}\ \sum_{i \in P_c} \mathrm{CAD}_i \cdot \max_{j \in A} \kappa(i,j).$$

The objective interpolates two familiar poles: dropping the coverage term
(weight all mass on the single best point) recovers **CAD-Prune**; setting
$\mathrm{CAD}_i \equiv 1$ recovers plain coverage (the K-centers family). R-CAD is
the difficulty-weighted interior.

### 7.2 Method ii — SHARP (Soft–Hard Alignment + Representativeness Pruning)

**Intuition.** [HT Sec.3] observes that **soft-label** training *homogenizes*
per-sample influence and approaches full-data accuracy regardless of subset
quality, whereas **hard-label** training stays quality-sensitive. SHARP therefore
selects the images whose **hard label alone already behaves like soft-label
training** — images whose one-hot label is so informative that it delivers the
teacher's supervision "for free" — and, as with R-CAD, places them by
**coverage** rather than proximity to a centroid.

**Why the signal must be dynamic.** Statically, the per-logit gradient difference
between hard- and soft-label training is

$$(p - y) - (p - q) \;=\; q - y,$$

which is **independent of the student**. Selecting on it collapses to
teacher-confidence selection — RDED's own criterion [RDED Eq.8] — and yields
nothing new. Along the training trajectory, however, the **last-layer** gradients
under the two labelings factor through the shared feature map,

$$\nabla_W \mathrm{CE}_{\text{hard}} = (p_k - y)\,[\varphi(x);1]^\top, \qquad
  \nabla_W \mathrm{CE}_{\text{soft}} = (p_k - q)\,[\varphi(x);1]^\top,$$

so their cosine is exactly

$$\cos\!\big(\nabla \ell_{HL},\, \nabla \ell_{SL}\big)\Big|_{\theta_k}
\;=\; \cos\!\big(p_k - y,\ p_k - q\big),$$

which is **student-dependent and evolves over training** (verified as an exact
identity, including the bias term, in `sharp.py` selftest 1). SHARP's score
averages this over the $K$ scoring epochs:

$$A_{\text{grad}}(i) \;=\; \frac{1}{K}\sum_{k=1}^{K} \cos\!\big(p_k(i) - y_i,\ p_k(i) - q_i\big),$$

with teacher $q = \mathrm{softmax}(\text{observer}(x))$ at temperature 1. In the
benchmark this becomes the demand weight, per class shifted to be non-negative:
$w_i = z_c(A_{\text{grad}})_i - \min_c z_c(A_{\text{grad}}) + \varepsilon$, then
fed to the same feat-kernel facility location as R-CAD.

### 7.3 One-line summary

> **R-CAD and SHARP are the same coverage selector with different demand
> weights** — R-CAD weights by *difficulty* (CAD), SHARP by *soft–hard gradient
> alignment* ($A_{\text{grad}}$). Both cover the class in observer-feature space;
> they disagree only on which images most deserve to be covered.

---

## 8. Determinism and compute

- Every run is **deterministic and cached**: a completed cell / scoring blob is
  reused, never recomputed. Selections are a **pure function of the seed-0
  scoring run**.
- **R-CAD costs essentially nothing beyond CAD-Prune**: it reuses the cached
  scoring run for CAD and adds one observer forward pass (features) plus a CPU
  greedy — no extra training. The comparison with CAD-Prune is compute-exact.
- SHARP additionally reads the softmax trajectories $P$ already logged by the
  same scoring run, so it too adds no training.
- All sets and results are **path-keyed by the full config**; baseline caches are
  never touched.
