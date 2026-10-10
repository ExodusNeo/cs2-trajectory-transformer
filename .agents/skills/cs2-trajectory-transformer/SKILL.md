---
name: cs2-trajectory-transformer
description: Coding logic, scientific-integrity rules, decision checklists and runbooks for the CS2 Trajectory Transformer thesis (server-side aimbot and smurf detection from Counter-Strike 2 replays). Use whenever writing, reviewing or debugging code, data pipelines, training or evaluation runs, reports, or thesis text in this repository.
---

# CS2 Trajectory Transformer: Engineering & Research Skill

Thesis: *Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic
Trajectory Transformers* (Medel & Gutierrez, USeP CIC, 2026–2027). Outline defense November 2026.

This file is the **canonical** skill. Copies elsewhere (`skills/…`, `.claude/skills/…`) only point here.
`AGENTS.md` holds project context and the discrepancy registry; `THESIS_TRACKER.md` holds status.
When they disagree, trust in this order: **code + tests → THESIS_TRACKER.md → AGENTS.md → manuscript**.

---

## 0. Orientation (do this first, every session)

1. Paths: git root is the outer `cs2-trajectory-transformer/`. The project lives in the inner
   `cs2-trajectory-transformer/` (`src/`, `tests/`, CLIs). `data/` there is a symlink to `D:\cs2_replay_data`.
2. Python: `cs2-trajectory-transformer\venv\Scripts\python.exe` (3.14). On the GPU workstation:
   `D:\cs2_thesis_env\Scripts\python.exe`. Never use a bare `python` on PATH for project code.
3. Baseline: run the tests (Section 9). Record the pass count before changing anything.
4. Read the "Project state" below and the tracker phase table before proposing work.

### Project state (2026-10-10; update this block when it changes)
- **Real data:** one FACEIT demo (`data/raw_demos/clean/1-1cfcda8f-…dem`); CS2CD pilot subset (40 + 40 matches, seed 0)
  in `data/raw_demos/cs2cd/` → `data/processed_parquet_cs2cd/` (64,014 ATWs). Full CS2CD (795) not yet ingested.
- **Real evidence so far = the CS2CD tabular pilot** (`reports/pilot_cs2cd/`, 80 matches, 563 sessions):
  session AUROC 0.817 [0.749, 0.876]; +0.070 over raw angles and +0.043 from target-relative channels (both CIs > 0);
  confounds map/rank ≈ 0.63; TPR at 1% FPR ≈ 0.02; mouse-input consistency adds nothing. Other `reports/*.csv`
  are synthetic and **not evidence**.
- **Feature set:** 9 channels in `features.kinematics.MODEL_FEATURE_COLUMNS`. Checkpoints trained on
  the old 8-channel set (`models/checkpoints/best_model.pt`) are incompatible: retrain.
- **Empirical finding:** on the real demo, small view-angle steps are single mouse counts (0.022° × sens),
  >50% of live ticks show zero motion, and real-player tremor band power (median 0.023) is *below*
  white noise (0.16). Treat tremor as a weak candidate channel. Re-check with `probe_replay_signal.py`.
- **Manuscript:** aligned with the code on 2026-10-11 (text-only edit). Still open: typed TOC/list page numbers
  (out of sync since before that edit), smurf scope (authors' decision), whether to restate Table 8 targets
  relative to baselines (pilot 0.82 vs target 0.98), AntiCheatPT reproduction, ONNX export.

---

## 1. Golden rules

1. **Evidence before claims.** A claim about cheaters or humans needs a measurement on real data.
   Otherwise it is a hypothesis, and the text must say so.
2. **Synthetic data never counts as a result.** Keep it in `data/synthetic_parquet/`, tag it in every report.
3. **One preprocessing path.** Ingestion and inference both call `extract_player_feature_windows`.
4. **One feature contract.** Import `MODEL_FEATURE_COLUMNS`; never re-type a feature list.
5. **Zero leakage, always.** Splits are player- *and* match-disjoint. Never split randomly by window.
6. **Fail closed, loudly.** Raise with counts and context; never silently substitute defaults
   (no fake ELO, no 0.5 "calibrated" threshold, no random weights without `--allow_untrained`).
7. **Never destroy data.** No `rmtree`/bulk delete on `data/` or anything you did not create in this task.
8. **Tests prove behaviour, not just shapes.** Every bug fix gets a regression test that fails on the old code.
9. **Update the tracker** (`THESIS_TRACKER.md` audit rows + change log) and the AGENTS.md registry with every change.
10. **The manuscript `.docx` lives outside git.** Edits there cannot be undone, so get the user's
    explicit go-ahead first and never create backup copies yourself.

---

## 2. Scientific-integrity logic

### 2.1 Claim levels: label every statement
| Level | Wording to use | Example |
| :--- | :--- | :--- |
| Hypothesis | "we hypothesize", "candidate feature", "may" | "TBP may separate injected noise from human motion" |
| Synthetic check | "on synthetic pipeline-verification data" | "the pipeline trains end to end on synthetic windows" |
| Real measurement | numbers + N + data source + split | "AUROC 0.91 (95% CI …) on 212 held-out matches" |

**Replace on sight:** "invariant", "immune", "impossible", "definitive", "proves", "guarantees detection",
"consistently violate", "cannot be faked" → hypothesis wording or a measured number.

### 2.2 Labels are weak
`is_aimbot = 1` means **"this account later received a FACEIT cheating ban"**. It does not mean aim
assistance was active in that window. Bans include wallhack-only cheaters (outside scope) and mostly
come from kernel AC detections, so DMA/hardware cheats are under-represented. Consequences:
- Report **player-match session metrics as primary**; window metrics are secondary.
- Clean negatives should come from the *same* matches as positives (the other 9 players) so the
  model cannot learn data source, map pool or patch instead of cheating.
- Never claim DMA detection unless DMA-labelled data was evaluated.

### 2.3 Evaluation logic
- The threshold τ\* is calibrated on **validation**, applied once to **test**. Never tune on test.
- An FPR target is meaningless alone (τ = 1.0 gives FPR 0). Always report **TPR at FPR ≤ target**.
- Report observed FP count, clean N, and the 95% upper bound (Rule of Three only when FP = 0;
  Clopper–Pearson otherwise) plus cluster-bootstrap bounds. Never certify 0.01% without N_clean_test ≥ 30,000 and 0 FPs.
- Compare against baselines on the **identical split and seed**. State targets relative to baselines.
- Negative results (e.g. "tremor adds nothing") are results. Report them; do not tune them away.
- Model selection: AUROC on a single-class validation split is undefined. Select on validation loss then (see `train.py`).

---

## 3. Data & feature contract

### 3.1 CS2 replay conventions (verified against real demos)
| Fact | Consequence in code |
| :--- | :--- |
| 64 ticks/s, dt = 1/64 s | `tick_rate=64.0`; Nyquist 32 Hz |
| Yaw 0° = +X, 90° = +Y; **positive pitch looks down** | look vector z = −sin(pitch) (`view_target_angles_rad`) |
| Positions are feet origins | add `EYE_HEIGHT_UNITS = 64` to both eye and enemy head |
| `team_num` 2 = T, 3 = CT; 0/1 = unassigned/spectator | filter to `PLAYING_TEAMS`; never treat spectators as players or enemies |
| **Teams swap at halftime** | resolve opponents **per tick**; never fix a team from the first tick |
| Dead players keep emitting rows | filter `is_alive` for both player and enemies |
| Small angle steps = one mouse count (0.022° × sens); >50% ticks static | high-order derivatives are quantization-dominated; tremor nearly invisible |

### 3.2 Model channels (`MODEL_FEATURE_COLUMNS`, 9)
| Channel | Unit | Notes |
| :--- | :--- | :--- |
| `pitch` | rad [−π/2, π/2] | absolute **yaw is excluded** (map-orientation shortcut); yaw stays in Parquet for the raw baseline |
| `angular_velocity` ω | rad/s | great-circle speed, Eq (5) |
| `angular_accel` α | rad/s² | dω/dt |
| `angular_jerk` j | rad/s³ | d²ω/dt²: a scalar smoothness proxy, not limb jerk |
| `trajectory_curvature` κ_g | – | intrinsic geodesic curvature; latitude circles have κ_g = |tan p| |
| `curvature_entropy` S_c | bits | 32-tick window, 10 bins |
| `tremor_power_8_12hz` TBP | ratio | on **signed** rates (never |ω|: rectification doubles frequency) |
| `aim_error` | rad [0, π] | crosshair → nearest living enemy head; π = no target in range |
| `aim_error_rate` | rad/s | zeroed next to "no target" ticks |

ATW: FOV ≤ 30°, range ≤ 3500 u, ±64-tick combat buffers (`weapon_fire`, `player_hurt`, `player_death`),
L_min = 64 ticks (one FFT window), L_max = 512 with 50%-stride chunking.

### 3.3 Checklist: adding, removing or changing a feature
1. Implement in `src/features/kinematics.py` (vectorized, units in docstring, thesis equation tag).
2. Add it to `MODEL_FEATURE_COLUMNS`. Nothing else should hold a feature list.
3. Add a normalization branch in `dataset.normalize_kinematic_features` (heavy tails → signed `log1p`).
4. If it needs match context (enemies, events), compute it inside `extract_player_feature_windows`.
5. Update `ablation.py` configs, `TabularMLP` default `input_dim = 6 × n_channels`, tests, the AGENTS registry, the tracker, and note the manuscript tables to change.
6. Old checkpoints become incompatible: say so in the change log.

### 3.4 CS2CD specifics (src/data/cs2cd_adapter.py)
- Per match: `N.parquet` (demoparser2 ticks, 10 rows/tick, 64 Hz) + `N.json` (events, `cheaters`, `CSstats_info`).
- `Player_k` IDs repeat across matches: always namespace as `cs2cd_<folder>_<N>:<Player_k>` before hashing.
- Positives = listed cheaters. Negatives = players in `no_cheater_present` only (unlabeled players in cheater
  matches have ~55.6% label precision and are skipped). No FACEIT ELO (NaN). Valve MM source: report per source,
  never pair CS2CD positives with FACEIT negatives.
- Both CS2CD and FACEIT demos carry `usercmd_mouse_dx/dy` (raw mouse input). Mouse-input/view consistency is a
  session-level candidate signal for software aim assistance; hardware input emulators stay consistent.

### 3.5 ELO and identity
- Unknown ELO → `NaN` in Parquet → masked by `elo_mask` / `masked_smooth_l1_loss`. Never default to 1500.
- FACEIT API ELO may be *current* ELO, not ELO at match time. Record the source.
- SteamIDs and match IDs are salted-SHA-256 pseudonyms (`CS2_PSEUDONYMIZATION_SALT` required, RA 10173).
  Never log, commit or print raw SteamIDs or the salt.

---

## 4. Leakage & splitting logic

`partition_dataset_files(strategy=...)`:
- `"components"`: whole match–player connected components are atomic; allocation is size-aware and the
  largest component always goes to train.
- `"match_drop"`: split matches per class, then drop windows of players already in a higher-priority split
  (test > val > train). Still P- and M-disjoint; only windows are lost.
- `"auto"` (default): components, unless the graph is degenerate (one giant component, or a class with
  ≥ 3 matches but < 3 components). Expect `match_drop` on real FACEIT/pro data, where players recur.

Rules: keep the six disjointness assertions; report realized split sizes; raise `ValueError` with clean/cheater
counts when a valid split is impossible; never "fix" an infeasible split by allowing overlap.
For InfoNCE, train with `samples_per_player > 0` (P×K batches); random batches have almost no positive pairs.

---

## 5. Coding logic standards

### 5.1 Control flow & error handling
- **Validate at boundaries, trust inside.** Check inputs where data enters (parser, CLI args, Parquet load), then rely on invariants.
- **No silent fallbacks** that change meaning: no default labels, ELOs, thresholds or weights. Use explicit flags and log the reason.
- **No bare `except: pass`** around anything that decides a label, a split or a metric. If you catch, log what was lost and why.
- **Fail-closed audits:** if a prerequisite (partition, test split) is unavailable, report "unmet" with the reason, never a partial count as success.
- **Deterministic randomness:** every sampler/split takes a `seed`; use `np.random.default_rng(seed)`, never global `np.random.*` in library code.
- **Edge cases to always consider:** empty DataFrame, single tick, all-static window, no enemies alive,
  single-class split, NaN/inf, player present on both teams (halftime), duplicate ticks, tick gaps.

### 5.2 Numerics
- Divisions: `+ 1e-6`/`1e-9`. Probabilities: clamp to `[1e-6, 1 − 1e-6]` before `log`.
- Angles: wrap differences with `wrap_angle_rad`. Small angles: `arctan2(‖a×b‖, a·b)`; `arccos` only after `clip(-1, 1)` and never where sub-milliradian precision matters.
- Heavy-tailed channels (α, j, aim_error_rate): signed `log1p` scaling, not z-scores alone.
- Prefer `float64` for geometry, then cast to `float32` for tensors.

### 5.3 Vectorization with an oracle
Never loop over ticks in Python. When vectorizing an existing loop, keep the loop **in the test** as an
oracle and assert equality (`test_vectorized_tbp_matches_reference_loop` is the pattern).
Useful tools: `sliding_window_view`, `np.split` on run boundaries, `merge` + `groupby(...).min()` for per-tick nearest target.

### 5.4 PyTorch
- Mask-aware mean pooling; `src_key_padding_mask = ~attention_mask` (True = ignore).
- `torch.load(..., weights_only=True)`; `model.eval()` + `torch.no_grad()` for inference.
- Losses must handle "nothing valid in this batch" by returning `pred.sum() * 0.0` (keeps the graph), not a Python float.
- Build models with `feature_dim=len(FEATURE_COLUMNS)`; never hard-code 8 or 9.

### 5.5 Style
Full type hints; NumPy docstrings with **units and shapes**; thesis equation tag above every formula;
match the surrounding code's comment density. Windows: guard multiprocessing with `if __name__ == "__main__":`;
`core.autocrlf=true` handles line endings. Python 3.14 evaluates annotations lazily, but still import every
`typing` name you use (the GPU environment may run an older Python).

### 5.6 Never
`shutil.rmtree` on data dirs · synthetic files in `processed_parquet` · filename-based labels ·
`'substr' in reason` label logic (use word-boundary regex) · duplicate feature lists · window-random splits ·
reporting synthetic scores as findings · `.bak`/copy files.

---

## 6. Decision checklists

**Before any change:** run the tests → read the relevant module end to end → find the owning layer
(feature math → `kinematics.py`; match context → `batch_processor.extract_player_feature_windows`;
splits/batching → `dataset.py`; metrics → `evaluate.py`) → change it there, not in a caller.

**After any change:** tests pass (count ≥ before) → new regression test added → a short real-demo
smoke run if ingestion changed → tracker row + change log → AGENTS registry row → list the manuscript sections affected.

**Real-data sanity checks** (run after any ingestion change; investigate if violated):
| Statistic on processed real ATWs | Healthy | If violated, suspect |
| :--- | :--- | :--- |
| median `aim_error` | well above 0 (≈ 0.1–0.3 rad) | enemy resolution (self/teammates counted as enemies) |
| ATW rows / live ticks | well below 1 | FOV filter always true; halftime bug |
| share `aim_error == π` | small | enemies missing (team/alive filter too strict) |
| ELO NaN share | known and reported | manifest not found |
| zero-motion share of ticks | ≈ 0.5 | unit errors (degrees vs radians) |

**Adding a baseline:** same split, seed, feature normalization and metrics function (`compute_metrics`);
record its config in Table 7; never give it fewer epochs or less tuning than ST-Trans without saying so.

**Editing the manuscript:** confirm with the user first (no git, no backups) → change wording to match
claim levels (2.1) → keep equation numbers consistent with code tags → log the change in the tracker.

---

## 7. Runbooks (PowerShell, from the git root)

```powershell
$py = "cs2-trajectory-transformer\venv\Scripts\python.exe"

# Tests
& $py -m pytest "cs2-trajectory-transformer\tests" -q

# Synthetic pipeline check (never writes to processed_parquet)
& $py "cs2-trajectory-transformer\generate_benchmark_dataset.py" --output_dir "cs2-trajectory-transformer\data\synthetic_parquet"

# Empirical signal probe on real demos
& $py "cs2-trajectory-transformer\probe_replay_signal.py" (Get-ChildItem "cs2-trajectory-transformer\data\raw_demos\clean\*.dem")

# Ingest real demos (requires CS2_PSEUDONYMIZATION_SALT in .env)
& $py -c "import sys; sys.path.append('cs2-trajectory-transformer/src'); from data.batch_processor import batch_process_demos; batch_process_demos('cs2-trajectory-transformer/data/raw_demos/clean', 'cs2-trajectory-transformer/data/processed_parquet')"

# CS2CD (Hugging Face Parquet + JSON) -> separate store data\processed_parquet_cs2cd, then pilot study
& $py "cs2-trajectory-transformer\ingest_cs2cd.py" --download_per_folder 40 --seed 0
& $py "cs2-trajectory-transformer\pilot_study.py"

# Train (P x K batches for InfoNCE; auto leakage-free split)
& $py "cs2-trajectory-transformer\train.py" --data_dir "cs2-trajectory-transformer\data\processed_parquet" --epochs 50 --samples_per_player 4

# Evaluate (calibrates tau* on val, reports test + session + bounds + latency)
& $py "cs2-trajectory-transformer\evaluate.py" --data_dir "cs2-trajectory-transformer\data\processed_parquet" --model_path "cs2-trajectory-transformer\models\checkpoints\best_model.pt"

# Benchmark and ablation (write reports/ in the current directory)
& $py "cs2-trajectory-transformer\benchmark.py" --data_dir "cs2-trajectory-transformer\data\processed_parquet"
& $py "cs2-trajectory-transformer\ablation.py" --data_dir "cs2-trajectory-transformer\data\processed_parquet"

# Single-match audit
& $py "cs2-trajectory-transformer\analyze_match.py" --demo "<path>.dem"
```

---

## 8. Reporting targets (Table 8) with the honest reading

| Metric | Proposal target | How to report it |
| :--- | :--- | :--- |
| AUROC / AUPRC | > 0.98 / > 0.95 | vs AntiCheatPT (0.93 reported) and raw-angle baseline on the same split; with CIs |
| Window FPR | < 0.01% | **TPR at FPR ≤ 0.01%**, FP count, N_clean (≥ 30,000), bound type |
| Session FPR | ≤ 0.16% | primary operational number; N_clean_sessions and bound |
| ELO MAE / Spearman | < 150 / > 0.85 | known-ELO players only; ambitious, so report honestly if missed |
| Biometric P@1 | > 0.80 | state gallery size and protocol (held-out players, cross-match) |
| Latency | < 500 ms/match | device, batch size, windows per match |

---

## 9. Verification commands for agents

```bash
cd cs2-trajectory-transformer
venv/Scripts/python.exe -m pytest tests -q          # must not drop below the recorded count
venv/Scripts/python.exe probe_replay_signal.py data/raw_demos/clean/*.dem
```
State results plainly: pass counts, what was not run, and what remains unverified.
