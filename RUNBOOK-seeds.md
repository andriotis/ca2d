# Runbook: extending the tables to seeds 42–46 on multiple servers

Goal: add seeds **45, 46** to every cell of the four tables. Existing 3-seed
results (canonical and legacy JSONs) are **kept and never re-run** — servers
compute only the new seeds, and `merge` unions them on the laptop.

All commands run from `ca2d/`. `<LAPTOP>` = this machine.

## 1. Server setup (once per server)

1. Clone the repo (bench.py imports `ca2d.py`/`xarch.py`/`rcad.py`/`tin.py`,
   and KD cells run the official RDED repo, expected as a **sibling** directory
   `../RDED` — it is *not* just one file), create the env, and bootstrap:

   ```bash
   conda env create -f environment.yml && conda activate rded
   python bench.py prepare        # downloads datasets + teachers, builds trees
   ```
2. Prewarm the deterministic caches from the laptop. For CIFAR this is an
   optimization; **for TIN on ≤10GB GPUs (3080) it is required**: the
   full-dataset scoring/EL2N runs hold the whole 100k-image train set in VRAM
   (~4.9GB + activations, OOMs a 3080), while set-sized cells peak at ~4.1GB
   and are safe. `launch` warns if a manifest would trigger uncached scoring.
   Prewarming also stops servers duplicating selection/synthesis and racing
   on shared sets:

   ```bash
   rsync -av <LAPTOP>:nikos/ca2d/artifacts/scores/          artifacts/scores/
   rsync -av <LAPTOP>:nikos/ca2d/artifacts/sets/            artifacts/sets/
   rsync -av <LAPTOP>:nikos/ca2d/artifacts/rded_sets/       artifacts/rded_sets/   # optional (cheap to rebuild)
   # el2nbest only: ship the seed-42 window-search results so servers skip the
   # 10-window search (they would otherwise redo it — deterministic but costly):
   rsync -av --include='bench_*_el2nsearch_*.json' --exclude='*' \
       <LAPTOP>:nikos/ca2d/artifacts/results/ artifacts/results/
   ```
   **df1 prebuild (run once on df1, before any launches).** Builds every
   still-missing deterministic artifact on the 24GB GPUs, so the ds boxes
   never score (3080-unsafe) and concurrent regime-split jobs never race on
   shared set builds. The el2nbest cells also produce needed 42–44 baselines:

   ```bash
   # missing scoring blob (full-dataset run — 24GB GPU only)
   python bench.py score --ds tin --arch rn18 --ipc 100 --device cuda:1
   # missing selection/synthesis sets
   for m in cadprune sharp rcad sharp2d rcad2d; do
     python bench.py select --ds tin      --arch conv --method $m --ipc 100 --device cuda:1
     python bench.py select --ds cifar100 --arch rn18 --method $m --ipc 100 --device cuda:1
     python bench.py select --ds tin      --arch rn18 --method $m --ipc 100 --device cuda:1
   done
   for m in cadprune sharp rcad; do
     python bench.py select --ds tin --arch rn18 --method $m --ipc 50 --device cuda:1
   done
   for m in rded ca2d; do
     python bench.py select --ds tin --arch rn18 --method $m --ipc 100 --device cuda:1
   done
   # missing el2nbest window searches for the ds workloads (also 42-44 cells)
   python bench.py cell --ds tin --arch conv --method el2nbest --regime sl --ipc 100 \
       --seeds 42,43,44 --devices cuda:0,cuda:1,cuda:2
   python bench.py cell --ds tin --arch conv --method el2nbest --regime kd --ipc 100 \
       --seeds 42,43,44 --devices cuda:0,cuda:1,cuda:2
   ```

   then re-run the step-2 rsyncs so `sets/` and the el2nsearch files reach the
   servers.
3. Sanity: `python bench.py selftest`.

## 2b. df1 runs its own share (rn18 tables) — staging results

df1 is the live checkout: running new seeds directly would **overwrite** the
existing seed-42–44 cell JSONs (a `--seeds 45,46` run is a cache miss that
rewrites the same filename). `CA2D_RESULT_DIR` redirects result writes to a
staging dir; scores/sets stay shared (they are caches, not results):

```bash
cd ~/nikos/ca2d
export CA2D_RESULT_DIR=$PWD/artifacts/results_s4546
mkdir -p $CA2D_RESULT_DIR
cp artifacts/results/bench_*_el2nsearch_*.json $CA2D_RESULT_DIR/  # reuse seed-42 searches
tmux new -s bench 'export CA2D_RESULT_DIR=$PWD/artifacts/results_s4546; \
  python bench.py launch plans/deepfinance1-phase1.yaml && \
  python bench.py launch plans/deepfinance1-phase2.yaml'
```

At merge time the staging dir is just another input dir (run with the env
var **unset** so merges land in the real results):

```bash
unset CA2D_RESULT_DIR
python bench.py merge artifacts/results_s4546 ~/incoming/*/ --dry-run
```

## 2. Write the manifest and launch

Copy `plans/serverA.yaml` / `plans/serverB.yaml` and edit `table`/`seeds`/
`devices` to the server's GPUs. Do **not** pin single experiments to single
GPUs — give a job all its devices and let the pool balance (cells/seeds have
very unequal durations).

```bash
python bench.py launch plans/serverX.yaml --dry-run   # validate + preview
tmux new -s bench 'python bench.py launch plans/serverX.yaml'
```

- `launch` refuses to start if any two jobs share a cell or a device.
- Per-job output goes to `artifacts/logs/launch_<name>.log`; the terminal shows
  `[name done/total]`-prefixed cell completions.
- **Crash / Ctrl-C → rerun the same command.** Finished cells are cached and
  skipped. Resume granularity is the *cell*: a cell interrupted after 4/5 seeds
  reruns all its seeds (accepted cost; result JSONs are written atomically, so
  an interrupt can never corrupt them).

## 3. Collect results on df1

Copy each ds server's results into its **own staging dir** — never directly
into the repo's `artifacts/results/`:

```bash
rsync -av dvspanos@deepstream1.csd.auth.gr:nikos/ca2d/artifacts/results/ ~/incoming/ds1/
rsync -av dvspanos@deepstream2.csd.auth.gr:nikos/ca2d/artifacts/results/ ~/incoming/ds2/
```

## 4. Merge (df1, with CA2D_RESULT_DIR unset)

The df1 staging dir from §2b is just another merge input:

```bash
python bench.py merge artifacts/results_s4546 ~/incoming/*/ --dry-run
python bench.py merge artifacts/results_s4546 ~/incoming/*/
```

Any number of dirs; order-independent; safe to rerun (already-merged cells
report "unchanged"). Duplicate (cell, seed) pairs with identical accs are
deduped; **differing accs abort the merge** naming the cell — that means two
sources disagree (e.g. a server accidentally re-ran seeds 42–44 under a
different env) and must be resolved by hand.

## 5. Render

```bash
python bench.py table   # header/footnote show n=5 (or n=3-5 while in transition)
python bench.py tex
```

Welch †/‡ marks are recomputed from the raw per-seed accs, so they use n=5
automatically.

## Caveats

- The `--seeds` default is now `42,43,44,45,46`: a local `run`/`cell` before
  merging treats existing 3-seed files as cache misses and would re-run the
  cell — merge first, or pass `--seeds 42,43,44` explicitly.
- `seed_secs` merged from different machines are not comparable wall-times;
  `bench.py timing` averages across hardware.
- GPU0 on the current lab box is ~3.5× slower under load; the pool tolerates it
  but a job's wall-clock is bounded by its slowest device.
