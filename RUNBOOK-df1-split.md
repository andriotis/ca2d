# Runbook: bring ALL four tables to n=5 (seeds 42-46), split across 3 servers

Goal: every cell of all four tables has seeds **42,43,44,45,46**. This is
**~2.3x** the original 45/46-only effort (783 GPU-h vs 345): the ipc100 tier and
several kd cells never had the 42-44 baselines either, and conv-tin ipc100 needs
backfilling too. Split across all three servers it finishes in **~5.9 days**
(vs ~11 days on df1 alone). conv-cifar100 is already complete and untouched.

**The split (verified: 0 overlaps, every incomplete cell covered):**

| Server | GPUs | Slice | Plan(s) | ~wall |
|---|---|---|---|---|
| deepfinance1 | 3× 3090 | rn18-**tin**, methods {random,el2nbest,cadprune,sharp,sharp2d,rcad2d} | `deepfinance1-final.yaml` | 5.6 d |
| symphony2 | 1× 4090 | rn18-**tin** {ca2d,rded,rcad}, then **conv-tin** (all) | `symphony2-df1tin.yaml` && `symphony2-convtin.yaml` | 5.6 d |
| deepstream1 | 2× 3080 | rn18-**cifar100**, all cells | `deepstream1-df1cifar.yaml` | **5.9 d** |

Every job runs `seeds: [42,43,44,45,46]` over the full grid; `eval_cell` runs only
the seeds each cell is missing and seed-unions them (already-n=5 cells cached-skip).
No two servers share a `(table,method,regime,ipc,seed)`, so `merge` never conflicts.

All commands run from `ca2d/`. `<LAPTOP>` = the collection machine (holds `incoming/`).

## 1. Prebuild + ship artifacts (before launching s2/ds1)

The 3080s can't build rn18 artifacts (full-dataset scoring OOMs 10GB), and
s2/ds1 only ever built *conv* artifacts. Build the missing ones on a **df1** 24GB
GPU and rsync out. df1 already has tin rn18 ipc100 sets from its earlier runs;
**cifar rn18 and conv-tin ipc100 need building.** el2n window searches are the
long pole — start them first.

On **df1** (spare GPU; run before/around the relaunch):
```bash
# cifar rn18: scores + sets (all ipcs)
for i in 10 50 100; do
  python bench.py score --ds cifar100 --arch rn18 --ipc $i --device cuda:1
  for m in cadprune sharp rcad sharp2d rcad2d rded ca2d; do
    python bench.py select --ds cifar100 --arch rn18 --method $m --ipc $i --device cuda:1
  done
done
# conv-tin ipc100 sets (for s2's phase 2)
for m in cadprune sharp rcad sharp2d rcad2d rded ca2d; do
  python bench.py select --ds tin --arch conv --method $m --ipc 100 --device cuda:1
done
# el2nbest: the 3080s OOM on full-dataset EL2N *scoring* only. Build just those
# shared scores here, PARALLEL across all 3 GPUs (one call per ds/arch), and ship
# them; ds1 then does its own window search + eval seeds during launch (set-sized,
# fits 10GB). Do NOT run `bench.py cell` here — it serializes to ~1 GPU (search &
# scores are cache-first/shared, leaving only 1-2 eval seeds) and the launch plans
# run those eval seeds anyway. tin el2nbest runs on df1/s2 (24GB) -> scores inline.
python bench.py el2n --ds cifar100 --arch rn18 --devices cuda:0,cuda:1,cuda:2
```
`python bench.py selftest`, then ship:
```bash
# -> symphony2 (rn18-tin + conv-tin sets/scores + el2nsearch)
rsync -av artifacts/scores/ symphony2:nikos/ca2d/artifacts/scores/
rsync -av artifacts/sets/   symphony2:nikos/ca2d/artifacts/sets/
rsync -av --include='bench_*_el2nsearch_*.json' --exclude='*' \
    artifacts/results/ symphony2:nikos/ca2d/artifacts/results/
# -> deepstream1 (cifar rn18 sets/scores + el2nsearch; the 3080s only load)
rsync -av artifacts/scores/    deepstream1:nikos/ca2d/artifacts/scores/
rsync -av artifacts/sets/      deepstream1:nikos/ca2d/artifacts/sets/
rsync -av artifacts/rded_sets/ deepstream1:nikos/ca2d/artifacts/rded_sets/
rsync -av --include='bench_*_el2nsearch_*.json' --exclude='*' \
    artifacts/results/ deepstream1:nikos/ca2d/artifacts/results/
```

## 2. Stop df1's current run and relaunch

```bash
tmux attach -t bench      # Ctrl-C the current launch, Ctrl-b d to detach
tmux new -s bench 'python bench.py launch plans/deepfinance1-final.yaml'
```
Interrupting keeps all finished cells (atomic writes); only in-flight seeds rerun.

## 3. Launch the two idle servers (after step 1 ships)

```bash
# symphony2 — rn18-tin, then conv-tin (chained)
python bench.py launch plans/symphony2-df1tin.yaml --dry-run     # expect 0 conflicts
tmux new -s bench 'python bench.py launch plans/symphony2-df1tin.yaml && \
  python bench.py launch plans/symphony2-convtin.yaml'

# deepstream1 — --dry-run MUST report no uncached scoring (else step 1 missed a set)
python bench.py launch plans/deepstream1-df1cifar.yaml --dry-run
tmux new -s bench 'python bench.py launch plans/deepstream1-df1cifar.yaml'
```
If a 3080 OOMs on a cifar cell, its set wasn't shipped — re-check step 1; never
let it fall back to scoring.

## 4. Collect + merge (on <LAPTOP>)

```bash
rsync -av symphony2:nikos/ca2d/artifacts/results/    incoming/s2new/
rsync -av deepstream1:nikos/ca2d/artifacts/results/  incoming/ds1new/
rsync -av deepfinance1:nikos/ca2d/artifacts/results/ incoming/df1/
python bench.py merge incoming/*/ --dry-run
python bench.py merge incoming/*/
```

## 5. Verify n=5 and render

```bash
python bench.py table    # header should read n=5; no cell should show n<5
python bench.py tex
```
`bench.py table` is the authoritative n check (it applies the legacy-JSON
resolution). If any cell still shows n<5, that cell's seeds didn't all land —
rerun the owning server's plan (resume fills only the gap).

## Notes

- **conv-cifar100** is already n=5 and appears in no plan — nothing to do.
- The ~5.9 d estimate leans on extrapolating tin rn18 ipc100 hl/sl at 2x the
  measured ipc50 (no such cell has finished yet); the first ipc100 hl/sl cells to
  land will confirm it.
- deepstream1 (cifar) is the bottleneck; symphony2 finishes ~0.3 d earlier. To
  shave the tail, after s2 drains you can hand it a cifar method off ds1 as a
  further chained phase (ship cifar artifacts to s2, remove that method from
  `deepstream1-df1cifar.yaml` to stay disjoint). Worth it only if the tail matters.
