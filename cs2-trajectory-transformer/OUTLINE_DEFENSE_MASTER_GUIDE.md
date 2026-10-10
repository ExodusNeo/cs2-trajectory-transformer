# 🎓 CS2 Trajectory Transformer — Outline Defense Master Guide

> **Thesis Title:** Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers
> **Authors:** Judah Ben Hur L. Medel & Dishann G. Gutierrez · **Adviser:** Vera Kim S. Tequin
> **Panel:** College of Information and Computing (CIC), University of Southeastern Philippines (USeP)
> **Outline Defense:** November 2026 · **Rewritten:** 2026-10-11 (replaces the earlier guide, whose results were synthetic)

---

## 0. Read this first: what changed and why

The previous guide claimed results we do not have: 128-tick data, "15–45% tremor in humans", "< 0.1% FPR achieved",
an ablation "proving" invariants, and a line-of-sight check. Each would be easy to catch: the replays run at 64 Hz,
the ablation ran on synthetic data built to separate, and there is no occlusion check. **Do not reuse any of it.**

The defense is now built on three things a panel respects:
1. **A falsifiable central question:** do physics-informed and target-relative features improve *leakage-free*
   cheat detection over a model fed raw view angles?
2. **Measured facts about the data,** including inconvenient ones (tremor is barely visible in replays).
3. **A real-data pilot on CS2CD** (Section 5), honestly scoped as feasibility evidence, not the final result.

An outline defense examines whether the *plan* is sound. Showing that you have already found and fixed the
weak points is the strongest position you can be in.

---

## 1. The one-paragraph pitch (memorize)

> Kernel anti-cheats cannot see cheats that run on separate hardware, and aggregate statistics are easy to
> game. Every aim-assist, however it is delivered, must eventually move the crosshair, and the game server
> records that movement 64 times per second. We extract combat windows from CS2 replays, describe them with
> nine candidate channels (motion smoothness, curvature, a tremor-band measure and, crucially, the crosshair's
> angle to the nearest enemy), and test with leakage-free splits whether these channels beat a raw-angle model.
> A pilot on 80 real CS2CD matches already shows that our engineered channels separate cheaters from clean
> players at the player-match level (AUROC 0.82) and beat a raw-angle model by +0.07 AUROC, with a 95% interval
> that excludes zero. It also shows what is still hard: catching cheaters at very low false-positive rates.

---

## 2. Format & speaker split

- **Presentation:** 15–18 minutes (~1 minute per slide). **Q&A:** 20–30 minutes.
- **Dishann:** problem, related work, theory-as-hypotheses, research questions (Slides 1–8).
- **Judah:** data, pipeline, model, evaluation, pilot, threats, timeline (Slides 9–18).
- Both: Q&A. Whoever owns the slide answers first; the other adds only if something is missing.

---

## 3. Slide blueprint (18 slides)

| # | Slide | Key content | Avoid |
| :-: | :--- | :--- | :--- |
| 1 | Title | Title, authors, adviser | — |
| 2 | Problem: client-side limits | DMA / external-HID cheats run outside the OS a kernel driver can inspect; privacy & stability costs of Ring-0 | "kernel AC is useless", "immune" |
| 3 | Problem: server-side is hard too | Aggregate stats are gameable; GAN-Aimbots [1] and Witschel & Wressnegger [9] show humanized aimbots evade learned detectors | Implying prior work failed by being naive |
| 4 | Smurfing (secondary aim) | Skill estimation from mechanics; evaluated by simulated rank discrepancy | Claiming solved smurf detection |
| 5 | Research questions | RQ1 features · RQ2 ATW segmentation · RQ3 dual-head model · RQ4 vs baselines, TPR at strict FPR | — |
| 6 | Conceptual framework | IPO: replays → ATWs → 9 channels → ST-Trans → aimbot score, embedding, ELO | "8D" |
| 7 | Theory as hypotheses | Fitts → aim error; Flash & Hogan → jerk; tremor → TBP; curvature. Each with "what a cheat would change" | "laws", "invariants", "cannot be faked" |
| 8 | What replays can physically show | Small steps are single mouse counts; >50% of live ticks static; real TBP 0.023 < white noise 0.16 → tremor is a weak candidate, tested not assumed | Hiding this. It is a strength. |
| 9 | Data sources & labels | FACEIT (ban-verified, ELO) + CS2CD (795 Valve MM matches). Labels are account-level. CS2CD negatives only from no-cheater matches; per-source reporting | "ground truth" without qualification |
| 10 | ATW extraction | 64 Hz; FOV ≤ 30°, ≤ 3500 u; ±64-tick event buffers; L_min 64, L_max 512; opponents per tick (halftime swap) | "line-of-sight check" |
| 11 | Nine channels | Table 5: pitch, ω, α, j, κ_g, S_c, TBP, aim error, aim-error rate; absolute yaw excluded (map shortcut) | — |
| 12 | ST-Trans | 9→128, 4 layers × 8 heads, mask-aware pooling, dual heads | — |
| 13 | Training objective | Focal + 0.5·InfoNCE (P×K batches) + 0.2·masked Smooth-L1 ELO | — |
| 14 | Leakage-free evaluation | Player- *and* match-disjoint splits (components, or match-level fallback); τ* on validation; TPR at FPR ≤ 0.01%; session-level primary; bounds under label noise | "0.01% FPR achieved" |
| 15 | Pilot on CS2CD | Section 5 table: ablation at session level, grouped CV | Over-reading small-n numbers |
| 16 | What the pilot does *not* show | Weak TPR at 1% FPR; map/rank confound ≈ 0.63; mouse-input consistency weak; tabular model, not ST-Trans | Hiding it |
| 17 | Threats to validity | Label noise, source confound, no occlusion, quantization, DMA under-representation, adaptive adversaries | — |
| 18 | Timeline & close | Ingest full corpus → train ST-Trans + baselines → ablation → Chapter 4 | — |

**Slide 8 is your best slide.** Script:
> *"Before trusting any feature we measured what replays can physically show. View angles move in steps of one
> mouse count, more than half of live ticks have no motion, and the 8–12 Hz band holds less power than white
> noise. So we do not assume tremor separates humans from bots. We keep it as one candidate channel and let the
> ablation decide. That measurement is also why we added target-relative features."*

---

## 4. What we claim, and what we do not

| We claim | We do not claim |
| :--- | :--- |
| A leakage-free pipeline from real CS2 replays to session-level cheat scores | That cheats "cannot fake" human motion |
| Candidate channels grounded in motor-control theory, tested by ablation | That tremor or jerk are invariants |
| A pilot on real CS2CD matches (Section 5) | Final accuracy, or any achieved 0.01% FPR |
| A plan to compare against raw-angle and tabular baselines on identical splits | That we beat AntiCheatPT (not yet reproduced) |
| Server-side analysis without client software | Detection of DMA cheats specifically (labels under-represent them) |

---

## 5. Real-data pilot (CS2CD subset)

Run: `python ingest_cs2cd.py --download_per_folder 40 --seed 0` then `python pilot_study.py`.
Outputs: `reports/pilot_cs2cd/` (`summary.json`, `session_ablation.csv`).

**Setup.** 40 `with_cheater_present` + 40 `no_cheater_present` matches (seed 0) → 64,014 ATWs →
**563 player-match sessions (163 cheaters, 400 clean)**. Negatives only from no-cheater matches. Gradient-boosted
trees on per-session means of 6 statistics per channel; stratified **grouped 5-fold CV by match** (players exist
in one match only, so folds are also player-disjoint), 5 seeds. This is a fast feasibility probe, not ST-Trans.

### 5.1 Feature-group ablation (session level)

| Feature group | AUROC | AUPRC | TPR @ 1% FPR | TPR @ 5% FPR |
| :--- | :-: | :-: | :-: | :-: |
| **All 9 model channels** | **0.820** | 0.559 | 0.02 | 0.19 |
| Self-kinematics only (no aim error) | 0.787 | 0.530 | 0.02 | 0.18 |
| Target-relative only (aim error + rate) | 0.763 | 0.516 | 0.02 | 0.19 |
| Raw angles only (yaw, pitch) | 0.742 | 0.543 | 0.08 | 0.21 |
| Tremor band power only | 0.649 | 0.459 | 0.08 | 0.18 |
| Mouse-input consistency only | 0.678 | 0.459 | 0.04 | 0.15 |
| All 9 + input consistency | 0.806 | 0.535 | 0.00 | 0.13 |
| *Confound: map only* | 0.637 | 0.369 | 0.00 | 0.05 |
| *Confound: Valve average rank only* | 0.625 | 0.380 | 0.00 | 0.04 |

AUPRC chance level is 0.29 (163 / 563). Seed-to-seed spread is ≤ 0.026 AUROC.

### 5.2 Are the differences real? (paired bootstrap over matches, 2,000 resamples)

| Comparison | AUROC (full) [95% CI] | Δ AUROC [95% CI] |
| :--- | :-: | :-: |
| All 9 vs raw angles | 0.817 [0.749, 0.876] | **+0.070 [+0.027, +0.116]** |
| All 9 vs self-kinematics only | 0.817 [0.749, 0.876] | **+0.043 [+0.020, +0.066]** |
| All 9 vs map only (confound) | 0.817 [0.749, 0.876] | **+0.151 [+0.020, +0.279]** |

### 5.3 How to say it (and not over-say it)

- ✅ "On real CS2CD data, engineered channels beat raw view angles, and the target-relative channels add
  significant signal. Both intervals exclude zero."
- ✅ "Tremor band power alone is only slightly above the map confound, consistent with our signal probe."
- ✅ "Detection at strict false-positive rates is not solved: at 1% FPR the pilot catches about 2% of cheaters.
  That is exactly why the full study reports TPR at strict thresholds rather than AUROC alone."
- ✅ "Mouse-input consistency was a promising single-match anomaly, but across 163 cheaters it adds nothing at
  the session level; most CS2CD cheaters' view rotation is explained by mouse input."
- ❌ Do not compare 0.82 to AntiCheatPT's 0.9336: different model, protocol and unit (session vs window).
- ❌ Do not present 0.82 as the thesis result; it is a tabular pilot on 10% of CS2CD.

### 5.4 Data facts discovered while building the pilot (use in Q&A)
- CS2CD "cheater-present" matches have a **median of 4 labeled cheaters** (one match has all 10). Valve's
  trust-factor matchmaking appears to group cheaters, which is why the "not cheater" label there is unreliable.
- Maps differ by class (e.g. hostage maps and de_train/de_edin/de_anubis appear only in clean matches), hence
  the confound baselines.
- One downloaded file was silently truncated; the downloader now verifies byte counts and file integrity.

---

## 6. Q&A bank (honest answers)

**Q1. "Is 8–12 Hz tremor even visible at 64 Hz in a replay?"**
> Barely, and we measured it. 64 Hz resolves 8–12 Hz (Nyquist 32 Hz), but small view changes are single mouse
> counts and real players' tremor-band power sits below white noise. So tremor is a candidate channel tested by
> ablation, not a pillar of the method. `probe_replay_signal.py` reproduces the measurement.

**Q2. "A cheat developer can add fake tremor and minimum-jerk smoothing. Then what?"**
> Agreed; prior work (GAN-Aimbots, Witschel & Wressnegger) shows humanized aimbots evade detectors. That is why
> we do not rely on motion shape alone. Aim assistance has to act on the crosshair–target relationship, which
> our aim-error channels measure, and our synthetic stress test includes a humanized aimbot that mimics human
> motion but locks onto targets.

**Q3. "How do you know your labels are correct?"**
> They are not perfect, and we say so. A label means "this account was banned for cheating", not "this window
> was assisted". CS2CD's own audit gives 55.6% precision for "not cheater" in cheater matches, so we exclude
> those players and take negatives only from no-cheater matches (~97% clean). We report session-level metrics
> as primary and manually review flagged clean-labeled sessions.

**Q4. "Can you detect DMA cheats?"**
> Server-side analysis does not depend on the client, so a DMA card does not blind it. Most ban labels come
> from existing anti-cheat systems, though, so DMA cheats are under-represented in training data. We claim
> detection of aim *assistance*, not specifically DMA. A hardware mouse emulator produces real mouse input, so
> the input-consistency signal would not catch it; target-relative behavior still could.

**Q5. "How is there no data leakage?"**
> Splits are disjoint in both players and matches. We group matches that share players into connected
> components. If the graph collapses into one giant component, we switch to a match-level split that drops the
> overlapping players' windows. CS2CD player IDs only exist within a match, so they are namespaced per match.
> The disjointness assertions run on every split, and unit tests cover the giant-component case.

**Q6. "Why a transformer? Wouldn't XGBoost do?"**
> Maybe, and we test exactly that. Tabular baselines (RF, XGBoost, MLP on 54 window statistics) and a BiLSTM
> run on identical splits. The pilot already uses boosted trees as a fast probe. If the transformer does not
> beat them, that is a reportable result.

**Q7. "How do you compare with AntiCheatPT?"**
> AntiCheatPT reports AUROC 0.9336 on CS2CD with raw inputs. We use the same dataset, and our raw-angle
> ablation is the closest internal analogue. A faithful reproduction under our leakage-free protocol is planned
> before the final defense. Until then we do not claim to beat it, because numbers under different splits are
> not comparable.

**Q8. "Can you really show an FPR below 0.01%?"**
> Only if we observe zero false positives in at least 30,000 clean held-out windows, and even then only as a
> bound. Label noise sets a floor: about 3% of CS2CD "clean" players may be cheaters, so a flagged clean player
> may be a missed cheater. We report TPR at the calibrated threshold, FP counts and bounds per source, never
> "FPR achieved".

**Q9. "Your aim error ignores walls. Isn't that wrong?"**
> Demos do not contain map collision geometry, so we cannot raycast. We approximate engagement with a 30° cone,
> a 3500-unit range and combat events, and state the limitation. Note that tracking enemies through walls is
> itself cheat-like (wallhack). Prefiring by legitimate players is the main confound, and the model sees it in
> both classes.

**Q10. "Smurf detection has no ground truth."**
> Correct. We evaluate it with controlled rank-discrepancy simulations (a high-tier player's sessions tested
> against a low-tier nominal rank) and report skill-regression error only on players with known ELO. Smurf
> detection is our secondary objective; aimbot detection is primary.

**Q11. "Why is your dataset balanced when real cheating is rare?"**
> Training balance aids learning; evaluation reports AUPRC, TPR at strict FPR, and calibrated thresholds, which
> are what matter at real-world prevalence.

**Q12. "Privacy?"**
> We use only replay telemetry. SteamIDs and match IDs are salted-SHA-256 pseudonyms (RA 10173), the salt is a
> managed secret, and no client software or personal files are involved.

**Q13b. "Your pilot AUROC is 0.82 but your target is 0.98. Isn't the target unrealistic?"**
> The 0.82 comes from a tabular model on 80 matches with noisy account-level labels; the full study uses
> ST-Trans on the whole corpus. We treat 0.98 as aspirational and judge success by improvement over raw-input
> and tabular baselines on identical splits, plus TPR at strict FPR. *(Discuss with the adviser whether to
> restate Table 8 targets relative to baselines before the defense.)*

**Q13c. "Could your model just be learning the map or rank, not cheating?"**
> We tested that. Map alone gives AUROC 0.64 and rank alone 0.63. Our channels reach 0.82, and the bootstrap
> interval for the gain over map-only excludes zero. Absolute yaw, which encodes map orientation, is already
> excluded from the model input.

**Q13. "What if your features turn out useless?"**
> Then the thesis reports a rigorously measured negative result on real data with a leakage-free protocol,
> which is still a contribution, since most published anti-cheat numbers lack that rigor.

---

## 7. Live demo (only real, reproducible steps)

```powershell
cd cs2-trajectory-transformer
# 1. Tests (state the count shown)
venv\Scripts\python.exe -m pytest tests -q
# 2. What replays can show (Slide 8 numbers, live)
venv\Scripts\python.exe probe_replay_signal.py data\raw_demos\clean\1-1cfcda8f-0d0c-46ee-8863-f746235e48e7-1-1.dem
# 3. Pilot tables (pre-computed; re-running takes ~15 minutes, so show the CSVs)
type reports\pilot_cs2cd\session_ablation.csv
type reports\pilot_cs2cd\session_bootstrap.csv
```
Do **not** demo `demo_sample.py` as detection evidence: it runs on synthetic inputs. Do not run
`analyze_match.py` without a checkpoint trained on real data.

---

## 8. Readiness checklist

| Item | Status |
| :--- | :---: |
| Manuscript Ch. 1–3 aligned with code (2026-10-11 edits); adviser has read the new wording | 🔲 |
| TOC / lists page numbers regenerated in Word (they were out of sync before the edits) | 🔲 |
| Pilot run reproduced on the presentation laptop (or results exported) | 🔲 |
| Slides use only numbers from `reports/pilot_cs2cd/` and the probe; no synthetic numbers | 🔲 |
| Each speaker can answer Q1–Q13 without notes | 🔲 |
| Two timed dry runs under 18 minutes | 🔲 |
| PDF backup of slides on USB | 🔲 |
