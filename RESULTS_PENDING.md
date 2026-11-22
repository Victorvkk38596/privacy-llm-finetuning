# Second-half core grid: pending runs

Status on 2026-11-22: **no corrected runs have been executed.** The environment used for the code
update had no GPU and no PyTorch / Transformers / Opacus, and package installs were blocked. All code
changes and the checks that do not need those libraries are done (see "Checks run" below).

Everything below runs from the repository root. Each stage is resumable: swap pairs whose two records
in `results/grid/records/` have `status: ok` are skipped. Failed runs get a record with
`status: failed` (plus a log in `results/grid/logs/`) and the loop continues.

## 0. Environment

```bash
pip install -r requirements.txt        # torch, transformers, datasets, opacus, scikit-learn, pandas, pyyaml, ...
pip install matplotlib                 # figures in scripts/aggregate_grid.py
python tests/test_privacy_guard.py     # pure-Python ledger tests (already pass)
python tests/test_aggregate_grid.py    # aggregation on synthetic records (already pass)
```

The first run for seeds 43 and 44 writes `manifests/manifest_seed43.json` and `manifest_seed44.json`
(the seed-42 manifest is checked against the existing file and never overwritten). The partition code
asserts length balance (SMD < 0.10, KS p > 0.01) for each seed; if seed 43 or 44 fails that check, the run
fails visibly and the record says so.

## 1. Smoke test: accounting starts before the first private-data update

```bash
python scripts/run_core_grid.py --stage smoke
```

Runs a 3-step late-3 DP swap pair (seed 42) and checks, for both models: accounting activated before the
first private read; zero private reads before accounting; accountant at 0 steps at the first private read;
accountant steps == logical optimizer steps == 3; accounted ε ≤ target. Also checks that
`head_warmup_source: private` is rejected. Output: `results/grid/smoke/smoke_result.json`.

## 2. Step-1 gate: do early-3, late-3, all-12 learn at ε = 5 without warmup? (seed 42)

```bash
python scripts/run_core_grid.py --stage gate --warmup none
```

Pre-registered rule (`scripts/grid_records.py`): a model has collapsed if val acc < 55% or its
predicted-positive rate is outside [5%, 95%]. Decision: `results/grid/warmup_decision_none.json`.

Only if that decision says `collapse`:

```bash
python scripts/run_core_grid.py --stage gate --warmup public_imdb
```

Use whichever variant learns as `<W>` below, for **every** DP configuration. Keep both sets of records.

## 3. Validate one run at each remaining setting (seed 42)

```bash
python scripts/run_core_grid.py --stage validate --warmup <W>
```

Covers middle-3, late-3 ε=1, late-3 ε=8, last-1, last-6, and the clean baseline.

## 4. Remaining seeds

```bash
python scripts/run_core_grid.py --stage grid --warmup <W>                 # everything still pending
python scripts/run_core_grid.py --stage grid --warmup <W> --seeds 43      # or one seed per session
python scripts/run_core_grid.py --stage grid --warmup <W> --plan          # list pending pairs + exact commands
```

## 5. Analysis, figures, report

```bash
python scripts/aggregate_grid.py --records results/grid/records --out results/grid/analysis --warmup <W> --estimate
pdflatex report.tex && pdflatex report.tex
```

`report.tex` picks up `results/grid/analysis/table_core.tex` and `fig_{location,budget,count}.pdf`
automatically. After the runs, write the interpretation paragraphs in the Results section by hand
(only call an accuracy difference "repeated" if it has the same sign in all three matched seeds; see
`matched_diffs.csv`).

## Kaggle

`kaggle_runner/run_grid.py` runs one stage per push. Set `GRID_STAGE`, `GRID_WARMUP`, optionally `GRID_SEEDS`
(edit the defaults at the top of section 3 in that file), then `kaggle kernels push -p kaggle_runner`.
Outputs land in `/kaggle/working/grid`. To resume in a later session, attach the previous output as a
dataset named `privacy-llm-grid-results`; the runner copies it back before starting.

## Pending configurations and seeds

Swap pairs (2 models each). `<W>` = chosen warmup; DP experiment names are `<config>__w-<W>`.

| Config | Blocks (+ head) | ε | Stage | Seeds pending |
|---|---|---|---|---|
| dp_early3_e5 | 0,1,2 | 5 | gate | 42, 43, 44 |
| dp_late3_e5 | 9,10,11 | 5 | gate | 42, 43, 44 |
| dp_all_e5 | all 12 | 5 | gate | 42, 43, 44 |
| dp_mid3_e5 | 4,5,6 | 5 | validate | 42, 43, 44 |
| dp_late3_e1 | 9,10,11 | 1 | validate | 42, 43, 44 |
| dp_late3_e8 | 9,10,11 | 8 | validate | 42, 43, 44 |
| dp_last1_e5 | 11 | 5 | validate | 42, 43, 44 |
| dp_last6_e5 | 6–11 | 5 | validate | 42, 43, 44 |
| clean_all | all 12 | ∞ | validate | 42, 43, 44 |

Total: 27 swap pairs = 54 models, plus the smoke pair (3 steps) and, only if needed, 3 extra gate pairs
with the public warmup.

## Compute estimate

No measured runtime exists yet (the pilot logs did not record wall time). After the gate stage, run
`python scripts/aggregate_grid.py ... --estimate`. It estimates the remaining time only from measured
per-pair wall times of completed configurations, and lists configurations with no measured run
separately instead of guessing.

## Checks run for this update (no GPU)

- `tests/test_privacy_guard.py`: 9 tests on the private-data ledger (ordering, blocking, Opacus
  shape-probe window, step-count matching). Pass.
- `tests/test_aggregate_grid.py`: swap-pair averaging, mean/SD, matched-seed differences, failed and
  pending visibility, collapse flag, partition-hash mismatch rejection, figure and table output, on
  synthetic records. Pass.
- `scripts/grid_spec.py`: static checks on the grid (anchoring, matched layer sets, parameter counts).
- `scripts/rdp_crosscheck.py`: independent RDP accountant reproduces the pilot's logged ε within 0.5%.
- All Python files byte-compile; undefined-name scan clean; `report.tex` compiles with and without results.
- **Not run:** anything importing torch / transformers / opacus, including the smoke test.
