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
2. Prewarm the deterministic caches from the laptop (this is what makes remote
   selection/synthesis a no-op — without it every server re-derives scores,
   selection sets and RDED/CA2D synthesis on GPU, and concurrent jobs could
   race building shared sets):

   ```bash
   rsync -av <LAPTOP>:nikos/ca2d/artifacts/scores/          artifacts/scores/
   rsync -av <LAPTOP>:nikos/ca2d/artifacts/sets/            artifacts/sets/
   rsync -av <LAPTOP>:nikos/ca2d/artifacts/rded_sets/       artifacts/rded_sets/   # optional (cheap to rebuild)
   # el2nbest only: ship the seed-42 window-search results so servers skip the
   # 10-window search (they would otherwise redo it — deterministic but costly):
   rsync -av --include='bench_*_el2nsearch_*.json' --exclude='*' \
       <LAPTOP>:nikos/ca2d/artifacts/results/ artifacts/results/
   ```
3. Sanity: `python bench.py selftest`.

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

## 3. Collect results on the laptop

Copy each server's results into its **own staging dir** — never directly into
the repo's `artifacts/results/`:

```bash
rsync -av serverA:.../ca2d/artifacts/results/ ~/incoming/serverA/
rsync -av serverB:.../ca2d/artifacts/results/ ~/incoming/serverB/
# ... one dir per server
```

## 4. Merge (laptop)

```bash
python bench.py merge ~/incoming/*/ --dry-run   # preview: seeds [42,43,44] -> [42..46] per cell
python bench.py merge ~/incoming/*/
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
