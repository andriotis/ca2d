# `R-CAD` and `SHARP` explained — from intuition to math to code

> A teaching walkthrough, in the same spirit as `RELMATCH_EXPLAINED.md` / `RELDIST_EXPLAINED.md`.
> We reuse the **one running example** (three classes: 🐱 cat, 🐶 dog, 🚗 car) and carry both
> methods all the way: the idea, the math, the algorithm (traced by hand), the code, and the
> at-scale numbers.
>
> **Prerequisite:** the `ca2d.py` docstring (CAD-Prune / CA2D reproduction) — both methods are
> built directly on its cached scoring run. Provenance tags are the repo's usual ones:
> `[HT ...]` = the Hard Truths paper, `[DD ...]` = Data Diet, `[RDED ...]` = RDED,
> `[ASSUME ...]` = a choice the papers leave unspecified.

---

## 0. TL;DR (read this first)

**CAD-Prune** `[HT Sec.5]` scores each image's *learning trajectory* in a vacuum — how much its
error still fluctuates late in a compute-matched training run — and keeps the per-class top-IPC.
That is a **pointwise** rule: it has no notion of what the rest of the kept set already contains,
so it happily spends a 10-image budget on ten near-copies of the same difficult mode.

- **R-CAD** (Relational Compute-Aware Difficulty) fixes the *set* problem. It keeps CAD as a
  per-image **demand weight** but replaces the ranking with a **set-level facility-location
  objective**: choose the set that best *covers* the class, where every pool image demands to be
  represented by a similar selected exemplar, and harder images demand it more. CAD-Prune and
  K-centers fall out as the two degenerate poles (difficulty-only and coverage-only) of the one
  objective — and the interior beats both poles, at both IPCs.

- **SHARP** (Soft-Hard Alignment + Representativeness Pruning) attacks a different question that
  `[HT Sec.3]` raises: soft labels make subset quality almost irrelevant, hard labels make it
  critical — so *which images' hard label already behaves like soft-label supervision?* The static
  answer degenerates to teacher confidence (RDED's own criterion), so SHARP measures it
  **dynamically**: the cosine between the hard-label and soft-label *last-layer gradients* along
  the scoring trajectory, which collapses to the closed form `cos(p_k − y, p_k − q)` — free to
  evaluate from cached softmax trajectories, no gradients ever materialized.

Headline (CIFAR-100 HL, local protocol, seeds 42–44): **R-CAD (feat/FL) 26.08 ± 0.21 (IPC 10) /
41.61 ± 0.20 (IPC 50)** — above every published selection baseline `[HT Tab.6/11]` — and
**SHARP-FL 24.18 ± 0.12**, selecting *almost disjoint* data from R-CAD (Jaccard 0.05): an
orthogonal signal at CAD-Prune-beating accuracy.

---

## 1. Vocabulary (the six words you need)

- **Scoring run.** One training run of the scoring model (ConvNet-D3) on the *full* dataset,
  compressed to `K` epochs so its compute equals a downstream run on the distilled set:
  `K · 50000 = 300 · IPC · 100` images-seen, so `K = 6` at IPC 10 and `K = 30` at IPC 50
  `[HT Sec.5]`. Every score below is read off this one run — selection never trains anything new.
- **EL2N** `[DD Def.2.3]`. The per-image error norm `S_k(i) = ‖softmax(f(θ_k, x_i)) − y_i‖₂` at
  epoch `k`, on clean images. Collected every epoch → a matrix `S ∈ R^{K×50000}`: each image's
  **learning trajectory** as a curve of "how wrong am I still?".
- **CAD** (Compute-Aware Difficulty) `[HT Eq.1–2]`. The *variability* of that curve: sliding-window
  standard deviations `U_k = std(S[k..k+J−1])`, averaged over the last `W = 2` windows (with
  `J = K−W` the windows span the whole run `[ASSUME 9]`). High CAD = the image is **still being
  learned** within the compute budget — neither memorized on epoch 1 nor hopeless noise.
- **Observer features.** The penultimate-layer features `φ(x)` of the frozen RDED observer
  checkpoint, **L2-normalized** (mandatory for cosine kernels — the Mahalanobis lesson from the
  RDED fork). Static geometry: where the image *sits*, as opposed to how it *learns*.
- **Facility location.** The set function `F(A) = Σᵢ wᵢ · max_{j∈A} κ(i,j)`: place IPC
  "facilities" so every pool image `i` is near some facility, weighted by its demand `wᵢ`.
  Monotone **submodular**, so greedy selection carries the classic `(1 − 1/e)` guarantee.
- **Probe trajectories.** `probe.py` re-executes the scoring run *bitwise-identically* (asserted
  against the cached `S`) but records the **full softmax** at every epoch:
  `P ∈ R^{K×50000×100}` (fp16), plus the teacher's soft labels `q = softmax(observer(x))`.
  This one artifact powers R-CAD's trajectory kernels and all of SHARP.

---

## 2. The idea in one picture

Take the 🐱 cat class, budget IPC = 2, and five cat images of three *kinds*:

```
                     CAD (difficulty)     what it is
  T1, T2, T3  :         0.2 each          textbook cats  (easy — learned by epoch 1)
  D1, D2      :         1.0 each          dog-ish cats   (hard — error still moving)
```

The dog-ish cats are near-duplicates of each other, and so are the textbook cats.

- **CAD-Prune** ranks by difficulty and keeps the top 2: `{D1, D2}` — **two copies of the same
  boundary mode**. The whole textbook-cat region of the class is unrepresented; the student
  never sees what an ordinary cat looks like.
- **K-centers / coverage-only** spreads over the class but is blind to difficulty: it happily
  anchors on a textbook cat first (they're the bigger cluster) and treats the decision-boundary
  mode as just another region.
- **R-CAD** does both at once: *cover the class, but let hard images shout louder*. It picks
  **one dog-ish cat first** (the high-demand mode), **then one textbook cat** (the uncovered
  mass) — `{D1, T1}`. One exemplar per kind, hard kind first. (§5 traces this by hand.)

SHARP asks an entirely different question about the same pool: forget difficulty — *for which
cats does the plain one-hot label "cat" already teach what the teacher's soft label
`[0.7 cat, 0.25 dog, 0.05 car]` would teach?* For a still-confused boundary cat, pushing toward
"cat" **is** pushing toward the teacher (aligned gradients). For a cat the teacher thinks is a
dog, the hard label **fights** the teacher. For an already-mastered textbook cat, there is
nothing left to teach and the alignment washes out. SHARP keeps the first kind — the images whose
hard label delivers the teacher's supervision *for free* — and covers the class with them (§8).

> One sentence each: **R-CAD = difficulty-weighted coverage of the class's geometry.
> SHARP = coverage of the class weighted by how soft-label-like each image's hard label is.**
> Same skeleton (facility location on observer features), orthogonal demand signals
> (Jaccard overlap of the selected sets: 0.05).

---

## 3. Shared ground: everything is read off one cached run

Both methods are **pure deterministic functions of already-cached artifacts** — no new training:

| artifact | file | contents | consumed by |
|---|---|---|---|
| scoring run | `artifacts/scores/cad_ipc{ipc}.pt` | `S ∈ R^{K×N}` (EL2N per epoch), `cad ∈ R^N` | R-CAD weights, `traj` kernel |
| probe | `artifacts/scores/probe_ipc{ipc}.pt` | `P ∈ R^{K×N×C}` (softmax per epoch), `q ∈ R^{N×C}` (teacher) | `softtraj` kernel, SHARP |
| twin | `artifacts/scores/probe_twin_ipc{ipc}.pt` | `P^SL`: the same run trained on `q` instead of `y`, identical RNG stream `[ASSUME P2]` | SHARP `2run` ablation |
| observer | `artifacts/.../cifar100_conv3.pth` | frozen RDED teacher | `feat` kernel, `q`, representativeness |

The probe's **determinism contract** matters: it recomputes EL2N with the identical seed and op
sequence and *refuses to save* unless it reproduces the cached `S` (bitwise in practice). So every
selection below is auditable back to one already-published scoring run, and the compute-matched
comparison with CAD-Prune is exact — R-CAD (feat/FL) adds only one observer forward pass and a
CPU greedy.

---

## 4. R-CAD — the objective (the math, gently)

Fix class `c` with pool `P_c` (all 500 class images — full pool, equal terms with every baseline).
Each image gets an **embedding** `E_i` (choices in §6) and the similarity kernel is the clipped
cosine

$$ \kappa(i,j) \;=\; \max\big(0,\; \cos(E_i, E_j)\big) \;\in\; [0,1], $$

so `κ = 1` means "same profile", `κ = 0` means "unrelated" (negative correlations are treated as
unrelated, not anti-related — this also keeps `F` monotone). Select

$$ \boxed{\;A_c^{\ast} \;=\; \arg\max_{A \subset P_c,\, |A| = \mathrm{IPC}} \; F_c(A), \qquad
F_c(A) \;=\; \sum_{i \in P_c} \mathrm{CAD}_i \cdot \max_{j \in A} \kappa(i,j).\;} $$

**Reading.** Every pool image `i` — kept or not — demands to be *represented* by the selected
exemplar most similar to it (`max_{j∈A} κ(i,j)` is its coverage, in `[0,1]`), and the demand is
weighted by its compute-aware difficulty `CAD_i`. Redundancy is punished automatically: a second
near-copy of an already-selected image raises almost nobody's `max`, so its marginal gain is ≈ 0.

**Why greedy is safe.** `F_c` is a nonnegative weighted facility-location function — monotone and
**submodular** (adding a facility to a bigger set helps less: the `max` can only grow by less).
Greedy therefore achieves `≥ (1 − 1/e) ≈ 0.63` of the optimum (Nemhauser et al., 1978); with
`n = 500` per class it is instant, CPU-only, and deterministic.

**The two poles — prior baselines as special cases.** The strongest published selection baselines
are the two degenerate corners of this one objective:

- **Difficulty-only.** Kill the relational term. The greedy rule
  `s(i \mid A) = \mathrm{CAD}_i · (1 − λ · \max_{j∈A} κ(i,j))` at `λ = 0` ignores `κ` entirely and
  reduces to a stable top-CAD sort — it reproduces the cached **CAD-Prune** selection
  *index-for-index* (selftest 5 in `rcad.py`).
- **Coverage-only.** Set `CAD_i ≡ 1`: plain facility location, the **K-centers**/coverage family
  `[HT Tab.6]`.

So R-CAD is not "CAD-Prune plus a heuristic" — it is the *interpolating family* whose endpoints
are the two best prior methods, and §10 shows the interior beats both endpoints at both IPCs.
The `dcad` discounted-greedy above (with its `λ` knob) is kept as an ablation; the knob-free FL
form is the method.

---

## 5. R-CAD — the algorithm, traced by hand

Pool = the five cats of §2, IPC = 2. Kernel values: near-duplicates `κ = 0.95` within a kind,
`κ = 0.30` across kinds. Weights `w = CAD`: textbook `0.2`, dog-ish `1.0`.

```
 κ         T1    T2    T3    D1    D2        w
 T1       1.00  0.95  0.95  0.30  0.30      0.2
 T2       0.95  1.00  0.95  0.30  0.30      0.2
 T3       0.95  0.95  1.00  0.30  0.30      0.2
 D1       0.30  0.30  0.30  1.00  0.95      1.0
 D2       0.30  0.30  0.30  0.95  1.00      1.0
```

**Greedy step 1.** `F({j})` for each single facility:

```
 F({T1}) = 0.2·(1 + .95 + .95) + 1.0·(.30 + .30)         = 0.58 + 0.60 = 1.18
 F({D1}) = 1.0·(1 + .95)       + 0.2·(.30 + .30 + .30)   = 1.95 + 0.18 = 2.13   <-- pick D1
```

The dog-ish mode wins the first pick *because its demand is 5× higher*, even though the textbook
cluster is bigger. Coverage after step 1: `cur = κ(·, D1) = [.30, .30, .30, 1, .95]`.

**Greedy step 2.** Marginal gains `Σᵢ wᵢ · (κ(i,j) − curᵢ)₊`:

```
 add D2:  only D2 improves (.95 → 1):     1.0·0.05                            = 0.05
 add T1:  T1 .30→1, T2 .30→.95, T3 .30→.95:  0.2·(.70 + .65 + .65)            = 0.40   <-- pick T1
```

**Result `{D1, T1}`** — one exemplar per kind, hard kind first. Compare:

- **CAD-Prune** (top-2 by `w`): `{D1, D2}` — two near-copies; the textbook mass stays at
  coverage 0.30 forever.
- **`dcad` at `λ = 0.5`** (the discounted greedy): after `D1`, the score of `D2` is
  `1.0·(1 − 0.5·0.95) = 0.525` vs `T1`'s `0.2·(1 − 0.5·0.30) = 0.17` — it **still picks the
  duplicate `D2`**; only `λ = 1` flips it. The subtractive discount needs its knob tuned per
  regime, while FL's demand structure gets it right knob-free — exactly the at-scale ordering
  (`dcad λ=.5` 24.24 vs FL 26.08, §10).
- **Coverage-only FL** (`w ≡ 1`): picks a **textbook cat first** (`F({T1}) = 3.50 >
  F({D1}) = 2.85`), then a dog-ish one — same final coverage, but with no principled tiebreak
  among the easy near-duplicates and, at scale, no extra budget spent where learning actually
  happens.

> Why it's safe: deterministic (no RNG anywhere in the greedy), `O(IPC · n²)` per class on CPU,
> and the `(1−1/e)` guarantee bounds how much the greedy can leave on the table.

---

## 6. R-CAD — the code (mapped to the math)

The whole method is ~40 lines in `rcad.py`. The greedy *is* the hand trace:

```python
def greedy_fl(kappa, w, budget):
    """Greedy max of F(A) = sum_i w_i * max_{j in A} kappa(i,j)."""
    cur = torch.zeros(n)                                   # cur_i = max_{j in A} kappa(i,j)
    for _ in range(budget):
        gains = ((kappa - cur.unsqueeze(1)).clamp(min=0) * w.unsqueeze(1)).sum(0)
        gains[sel] = -1.0
        j = int(torch.argmax(gains))                       # best marginal gain
        sel.append(j)
        cur = torch.maximum(cur, kappa[:, j])              # update coverage
```

`select_indices` runs it once per class on `κ = cos_gram(E)` (the clipped cosine) with
`w = cad[idx]` — the per-class decomposition is exact because both `κ` and `w` only involve
same-class images. The **kernel** (`--kernel`) chooses `E`:

| kernel | embedding `E_i` | what similarity means |
|---|---|---|
| `feat` **(headline)** | L2-normalized observer penultimate features | "sits in the same place" (static geometry) |
| `softtraj` | `[p_1(i)−y_i; …; p_K(i)−y_i] ∈ R^{K·100}` from the probe | "makes the same mistakes over training" |
| `traj` | centered EL2N row `S[:,i]` (cosine ⇒ Pearson) | "learns on the same schedule" (zero extra compute) |

The at-scale answer (§10): **static feature geometry wins the relational term decisively**. The
dynamics information is already carried by the CAD *weights*; the kernel's job is geometric
coverage, and `K = 6` trajectory points are too coarse to beat 2048-d features at it.

Artifacts are path-keyed by the full config (`sets/rcad_feat_fl_ipc10.pt`,
`results/rcad_feat_fl_ipc10.json`); baseline caches are never touched. `rcad.py synth` reuses the
same selector to build a 300-image *pool* per class and hands it to the stock RDED crop/stitch
pipeline (the CA2D head-to-head; §10).

---

## 7. SHARP — the alignment identity (the math, gently)

`[HT Sec.3]`'s headline finding: with **soft labels**, subset quality barely matters; with **hard
labels**, it decides everything. SHARP turns that around into a selection principle: *keep the
images whose hard label already delivers the teacher's supervision.*

**The static trap.** The per-logit gradient difference between hard- and soft-label training is

$$ (p - y) - (p - q) \;=\; q - y, $$

which **does not involve the student at all**. Ranking by "hard ≈ soft" statically is ranking by
`‖q − y‖` — i.e. by teacher confidence — which is RDED's own selection criterion `[RDED Eq.8]`.
Nothing new. The signal must be **dynamic**.

**The identity.** Along the scoring trajectory `θ_1 … θ_K`, the cross-entropy gradient at the
**last layer** factors as an outer product with the shared feature vector:

$$ \nabla_{W} \ell_t \big|_{\theta_k} \;=\; (p_k - t)\, \varphi(x)^\top, \qquad t \in \{y, q\} $$

(and with the bias absorbed via `φ → [φ; 1]`). Because both gradients share the factor `φ(x)`,
the feature norms cancel in the cosine and

$$ \boxed{\;\cos\big(\nabla \ell_{HL},\, \nabla \ell_{SL}\big)\Big|_{\theta_k}
\;=\; \cos\big(p_k - y,\; p_k - q\big)\;} $$

— **student-dependent, epoch-dependent, and computable from the cached probe softmaxes alone**
(no gradient is ever materialized; the identity, including the bias term, is verified against
autograd in selftest 1 of `sharp.py`). SHARP's main score averages it over the run:

$$ A_{\text{grad}}(i) \;=\; \frac{1}{K} \sum_{k=1}^{K}
\cos\big(p_k(i) - y_i,\; p_k(i) - q_i\big), $$

with teacher `q = softmax(observer(x))`, `τ = 1` `[ASSUME P1]`. Two ablation scores probe the
same idea with norms instead of angles: `A_pred = −mean_k ‖p_k − q‖` (EL2N against the teacher,
`[HT Def.1]`) and `A_2run = −mean_k ‖p_k^{HL} − p_k^{SL}‖` (the literal counterfactual: distance
to a **twin run** trained on `q` with an *identical RNG stream*, `probe.py --twin`
`[ASSUME P2]`). §10 shows why the cosine beats both.

**Representativeness.** Alignment alone can pile onto one aligned region, so SHARP intersects it
with class representativeness — in two forms: `R(i) = cos(φ̂(x_i), μ̂_c)` (proximity to the
normalized class centroid), combined as per-class z-scores `α·z(A) + (1−α)·z(R)` and top-IPC'd
(`--mode topk`); or, taking §5's lesson seriously, **representativeness as coverage**: facility
location on the observer feature kernel with the (shifted) alignment score as the demand weight
(`--mode fl`) — R-CAD's skeleton with `A_grad` in place of CAD.

---

## 8. SHARP — traced by hand

Three cat images (`y = [1, 0, 0]` on 🐱🐶🚗), one epoch `k` mid-training. Columns: student `p_k`,
teacher `q`, the two error vectors, and the two scores.

**Image A — the info-rich boundary cat.** Teacher hedges toward dog, student still confused:

```
 p_k = [.40, .45, .15]      q = [.70, .25, .05]
 e_h = p_k − y = [−.60, +.45, +.15]        e_s = p_k − q = [−.30, +.20, +.10]
 cos(e_h, e_s) = 0.285 / (0.765 · 0.374) = +0.996        A_pred = −‖e_s‖ = −0.374
```

The hard label's push ("more cat, less dog") is **the same direction** the teacher would push —
just larger. This image's one-hot label delivers the soft supervision for free. `A_grad ≈ +1`.

**Image B — the teacher disagrees.** A labeled cat the teacher reads as a dog
(`q = [.25, .70, .05]`), same student state:

```
 e_h = [−.60, +.45, +.15]        e_s = [+.15, −.25, +.10]
 cos(e_h, e_s) = −0.188 / (0.765 · 0.308) = −0.80
```

The hard label **fights** the teacher — training on it with a one-hot pushes the student away
from where soft-label training would go. Strongly negative alignment; SHARP drops it.

**Image C — the easy prototype.** Student already slightly *past* the teacher's confidence
(`p_k = [.95, .04, .01]`, `q = [.90, .08, .02]`):

```
 e_h = [−.05, +.04, +.01]        e_s = [+.05, −.04, −.01]
 cos(e_h, e_s) = −1.00                                    A_pred = −‖e_s‖ = −0.065
```

Here is the whole `grad`-vs-`pred` story in one row. **`A_pred` loves C** (its prediction sits
right on the teacher — best of the three) and would fill the budget with easy prototypes. But
there is nothing left to teach: once the student crosses the teacher's confidence, the hard label
says "push further" while the teacher says "pull back" — opposite directions, cosine `−1`. Early
in training (before the crossing) the same image scores `+1`; **averaged over the K epochs**, easy
images — learned in epoch 1, hovering around the teacher thereafter — wash out toward zero or
negative, while boundary images that spend the whole run approaching the teacher from the
confused side keep `A_grad` high. The cosine normalizes magnitude away and asks only:
*does the hard label push in the teacher's direction while the sample is still being learned?*

Final ranking on this pool: `A_grad`: **A ≫ B > C**. `A_pred`: **C ≫ B > A** — the norm score's
favorite is the cosine score's least favorite. The at-scale numbers agree (§10: grad 22.02,
pred 17.22).

---

## 9. SHARP — the code (mapped to the math)

`alignment_scores` is the boxed identity, evaluated on the cached probe in 5000-sample chunks:

```python
p   = P[:, i:i+5000].float()                              # (K, n, C) softmax trajectory
e_h = p - F.one_hot(y[i:i+5000], C).float()               # p_k − y
e_s = p - q[i:i+5000].float()                             # p_k − q
cos = (e_h * e_s).sum(-1) / (e_h.norm(dim=-1) * e_s.norm(dim=-1) + EPS)
A[i:i+5000] = cos.mean(0)                                 # mean over the K epochs
```

`select_sharp` then works per class: `topk` sorts `α·z(A) + (1−α)·z(R)` and keeps IPC;
`fl` shifts the score to a nonnegative demand weight and reuses **R-CAD's greedy verbatim**:

```python
if mode == "fl":
    w = s - s.min() + 1e-6                                # nonneg demand for facility location
    sel = np.array(greedy_fl(cos_gram(emb(idx)), w, ipc)) # coverage weighted by alignment
```

That single import (`from rcad import greedy_fl, ...`) is the architectural point: the two
methods share one selection skeleton and differ only in what a sample's *demand* means —
"still being learned" (CAD) vs "hard label already teaches the teacher's lesson" (`A_grad`).

---

## 10. Did it actually work? (the real numbers)

CIFAR-100, ConvNet-D3, hard-label protocol `[HT Tab.4]`, seeds 42/43/44, mean ± population std.
All baselines are the locally cached reproductions (which land 0.7–2.2 points *below* their paper
numbers, so local-vs-published comparisons are conservative).

**R-CAD — the interior beats both poles, at both IPCs:**

| config | IPC 10 | IPC 50 |
|---|---|---|
| **R-CAD: feat kernel, CAD-weighted FL** | **26.08 ± 0.21** | **41.61 ± 0.20** |
| feat, dcad λ=0.5 (subtractive discount) | 24.24 ± 0.08 | 38.49 ± 0.08 |
| feat, FL, uniform w (coverage-only pole) | 23.93 ± 0.14 | 37.55 ± 0.31 |
| softtraj, dcad λ=0.5 | 22.52 ± 0.11 | 37.60 ± 0.32 |
| CAD-Prune (= dcad λ=0, difficulty-only pole) | 21.92 ± 0.21 | 37.14 ± 0.08 |
| published best prior selection `[HT Tab.6]` | K-centers 25.04 | 38.64 |

Read it in three steps: **(1)** relative to coverage-only, the CAD demand weights add **+2.15 /
+4.06**; **(2)** relative to difficulty-only, the coverage term adds **+4.16 / +4.47** — neither
ingredient explains the result alone; the *product* structure is causal. **(3)** The static
`feat` kernel beats every trajectory kernel for the relational term: dynamics belong in the
weights, geometry in the kernel. R-CAD also clears the published K-centers bar (+1.04 / +2.97)
on a protocol where every reproduced baseline lands *below* its published number; among all of
`[HT Tab.6]` only the bi-level synthesis methods TM and (at IPC 10) DM remain above.

**SHARP — the component ablation is clean (IPC 10):**

| config | top-1 |
|---|---|
| **SHARP-FL: coverage weighted by A_grad** | **24.18 ± 0.12** (IPC 50: 38.09 ± 0.35) |
| A_grad alone (α=1, top-k) | 22.02 ± 0.16 |
| A_grad + centroid-R (α=0.5, top-k) | 20.78 ± 0.19 |
| centroid-R alone (α=0) | 20.51 ± 0.27 |
| A_pred + centroid-R (α=0.5) | 19.82 ± 0.10 |
| 2run counterfactual (α=1) | 18.92 ± 0.12 |
| A_pred alone (α=1) | 17.22 ± 0.13 |

Three findings: **(a)** the gradient-alignment cosine is the value carrier — alone it already
matches CAD-Prune (22.02 vs 21.92) while selecting almost **disjoint** data (Jaccard 0.04).
**(b)** "Representative" must mean *coverage*, not *centroid proximity*: mixing in the centroid
score **hurts** (20.78 < 22.02) — it drags the set toward the class center, §8's image-C
territory — while alignment-weighted facility location **helps by +2.16**. **(c)** Among the
three alignment definitions, the cheap single-run **cosine beats both norm forms** — including
the literal two-run counterfactual, at half its scoring cost — for exactly the reason image C
shows: norms reward sitting *on* the teacher (easy prototypes); the cosine rewards *moving
toward* the teacher.

**Orthogonality.** Spearman(CAD, A_grad) = 0.39 over all 50k images; set-level Jaccard at IPC 10:
R-CAD ∩ CAD-Prune = 0.15, SHARP ∩ CAD-Prune = 0.04, **SHARP ∩ R-CAD = 0.05**. R-CAD genuinely
interpolates its poles rather than re-ranking either; SHARP taps a near-orthogonal signal at
matching accuracy — the natural future combination is SHARP alignment × CAD demand in one FL
objective.

**The synthesis coda.** Feeding the R-CAD pool into RDED's crop/stitch (`rcad.py synth`) beats
CA2D like-for-like (21.42 vs 20.94 at IPC 10) but both lose badly to their pruning counterparts:
at 32×32 the crop/stitch stage destroys more information than pool quality adds. **The pruning
form is the method.**

---

## 11. Three honest framings (a reviewer will ask)

- **Is the FL objective new?** Weighted facility location is textbook coreset machinery (CRAIG,
  submodular selection); the contribution is *which weight and which kernel*: compute-aware
  difficulty as demand over static teacher geometry, shown to strictly dominate both of its own
  degenerate poles — the two strongest published baselines — under an exact compute match
  (R-CAD feat/FL costs nothing beyond CAD-Prune's own scoring run plus one observer pass).
- **Is `A_grad` really about gradients?** Only the *last layer's* — the closed form is exact for
  `∇_W` (and bias) but says nothing about earlier layers. That is the standard EL2N/GraNd-style
  approximation `[DD]`, made honest here by the autograd-verified identity and by the `2run`
  ablation, which replaces the approximation with the literal behavioral counterfactual (twin
  run, identical RNG) — and *loses*, so the cheap form is not just cheap, it is better.
- **What did not reproduce.** The published "K-centers 25.04" `[HT Tab.6]` could not be
  reproduced as farthest-point traversal (8.13 cosine / 14.23 Euclidean — it selects outliers at
  this budget); the strongest coverage-only method found is uniform FL at 23.93. The number is
  treated as a bar, not a baseline, and R-CAD clears it on the stricter local protocol. All added
  choices are tagged (`[ASSUME P1/P2/R1/R2]`) in the module docstrings.

---

## 12. Recap — the whole story in five sentences

1. CAD-Prune ranks each image's learning trajectory in a vacuum, so at tiny budgets it buys
   near-copies of one difficult mode and leaves whole regions of the class unrepresented.
2. **R-CAD** makes selection set-level: maximize CAD-weighted facility-location coverage of the
   class's (observer-feature) geometry — a monotone submodular objective whose greedy carries the
   `(1−1/e)` guarantee and whose two degenerate poles are exactly CAD-Prune and K-centers.
3. The interior beats both poles at both IPCs (26.08 / 41.61), with dynamics in the *weights* and
   static geometry in the *kernel* — and it costs nothing beyond the already-cached scoring run.
4. **SHARP** scores each image by whether its hard label pushes the student in the teacher's
   direction while the sample is still being learned — `mean_k cos(p_k−y, p_k−q)`, the exact
   last-layer hard/soft gradient cosine, read off cached softmax trajectories (the static version
   degenerates to RDED's teacher-confidence rule, so the signal must be dynamic).
5. Alignment-weighted coverage (SHARP-FL, 24.18) beats published CAD-Prune while selecting
   near-disjoint data from R-CAD (Jaccard 0.05) — two orthogonal demand signals on one selection
   skeleton, and a natural future combination.

---

## 13. Where to look in the repo

| piece | file | symbol |
|---|---|---|
| R-CAD: kernels, FL/dcad/kcenter greedies, eval | `rcad.py` | `build_embeddings`, `greedy_fl`, `greedy_dcad`, `select_indices` |
| SHARP: alignment scores, topk/fl selection | `sharp.py` | `alignment_scores`, `representativeness_scores`, `select_sharp` |
| the probe (softmax trajectories + teacher q + twin) | `probe.py` | `run_probe`, determinism contract |
| EL2N, CAD, compute-matched K, baselines | `ca2d.py` | `el2n_scores`, `cad_from_S`, `compute_matched_epochs` |
| the gradient-cosine identity vs autograd | `sharp.py` | selftest 1 |
| dcad λ=0 ≡ CAD-Prune, index-for-index | `rcad.py` | selftest 5 |
| all numbers, ablations, TIN/SL generalization | `REPORT.md` | §2–§9 |
| the four CVPR tables (cache-first) | `bench.py` | `run --table ...` |

Run it yourself (everything below CPU-greedy-cheap once the caches exist):

```bash
# 0) shared instrumentation (only needed for softtraj / SHARP; feat/FL needs no probe)
python probe.py --ipc 10 --twin

# 1) R-CAD headline: CAD-weighted facility location on observer features
python rcad.py  eval --ipc 10 --kernel feat --selector fl          # 26.08
python rcad.py  eval --ipc 50 --kernel feat --selector fl          # 41.61

# 2) SHARP headline: coverage weighted by the hard/soft gradient cosine
python sharp.py eval --ipc 10 --align grad --alpha 1.0 --mode fl   # 24.18

# 3) the poles, for contrast
python rcad.py  eval --ipc 10 --kernel feat --selector dcad --lam 0    # == CAD-Prune
python rcad.py  eval --ipc 10 --kernel feat --selector fl --weight uniform  # coverage-only
```
