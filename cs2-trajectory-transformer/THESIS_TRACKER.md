# 📌 CS2 Trajectory Transformer — Thesis Master Tracker & Roadmap

**Thesis Title:** Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers  
**Authors:** Medel & Gutierrez  
**Degree Program:** Bachelor of Science in Computer Science (Major in Data Science)  
**Academic Year:** 2026–2027  
**Outline Defense Target:** November 2026 (1st Semester)  
**Final Oral Defense Target:** May–June 2027 (2nd Semester)  
**Last Updated:** October 6, 2026  
**Post-Completion Guide:** See [POST_COMPLETION_GUIDE.md](POST_COMPLETION_GUIDE.md)  

---

## 🚦 Master Timeline & Phase Overview (Target: June 2027)

| Phase | Description | Target Window | Status | Completion % |
| :--- | :--- | :--- | :---: | :---: |
| **Phase 1** | Mathematical Foundations, 8 Features & ST-Trans Architecture | Aug – Sep 2026 | 🟢 COMPLETED | 100% |
| **Phase 2** | Real Replay Ingestion, Crawler & Autonomous Staging Buffer | Aug – Oct 2026 | 🟢 COMPLETED | 100% |
| **Phase 3** | Baselines, Feature Ablations & Experimental Proofs | Oct 2026 | 🟢 COMPLETED | 100% |
| **Phase 4** | Outline Defense Presentation & Proposal Examination | November 2026 | 🟡 IN PROGRESS | 85% |
| **Phase 5** | Extended Rolling Dataset Accumulation & GPU Scale-Up | Dec 2026 – Mar 2027 | ⚪ SCHEDULED | 15% |
| **Phase 6** | Final 5-Chapter Manuscript, ONNX Export & Final Oral Defense | Apr – June 2027 | ⚪ SCHEDULED | 10% |

---

## 🔍 Critical Technical Flaws Resolution Audit

| # | Identified Critical Flaw / Gap | Resolution Implementation | Verification Test | Status |
| :-: | :--- | :--- | :--- | :---: |
| **1** | **Euler Angle Discontinuity:** Boundary jumps across $\pm 180^\circ$ caused false $358^\circ$ velocity/jerk spikes. | `wrap_angle_rad` using $((\Delta \theta + \pi) \pmod{2\pi}) - \pi$ in `src/features/kinematics.py`. | `tests/test_kinematics.py::test_euler_angle_wrapping` | 🟢 Verified Fixed |
| **2** | **Ad-Hoc Planar Curvature Proxy:** 2D curvature formula invalid on spherical viewing coordinates. | Intrinsic 3D geodesic sight vector curvature $\kappa_g = \frac{|\mathbf{v} \cdot (\mathbf{v}' \times \mathbf{v}'')|}{\|\mathbf{v}'\|^3 + \epsilon}$ on unit sphere $S^2$ (yielding $\kappa_g = 0$ for unperturbed great circles). | `tests/test_kinematics.py::test_spherical_curvature_straight_vs_curved` | 🟢 Verified Fixed |
| **3** | **Missing 8–12 Hz Tremor PSD:** Proposal claimed neuromuscular frequency extraction, but code lacked FFT. | `compute_tremor_band_power` using sliding-window Hanning-windowed FFT relative power in $[8, 12]$ Hz. | `tests/test_kinematics.py::test_tremor_band_power_detection` | 🟢 Verified Fixed |
| **4** | **Replay Idle Walking Noise:** Analyzing entire 45-min matches diluted combat aimbot signals with 70%+ navigation noise. | Active Tracking Window (`src/data/atw_filter.py`) extracting $30^\circ$ enemy visual cones and $\pm 64$-tick combat event buffers. | `tests/test_parser.py::test_relative_fov_geometry` & `test_extract_active_tracking_windows` | 🟢 Verified Fixed |
| **5** | **Lack of Contrastive Smurf Embedding:** Initial model only performed scalar regression without biometric latent clustering. | 32-dim unit-normalized projection head optimized via `SupervisedInfoNCELoss` in `src/models/losses.py`. | `tests/test_model.py::test_infonce_contrastive_loss` | 🟢 Verified Fixed |
| **6** | **Aimbot Class Imbalance:** Sparse cheater engagement windows lead standard BCE to majority-class collapse. | Implemented `FocalLoss` ($\alpha=0.25, \gamma=2.0$) in `src/models/losses.py` down-weighting easy background samples. | `tests/test_model.py::test_focal_loss` | 🟢 Verified Fixed |
| **7** | **Data Leakage in Splits:** Splitting randomly across ticks or rounds of the same player causes memorization. | `create_partitioned_dataloaders` enforcing pairwise disjoint player-ID and match-ID splits. | `tests/test_dataset.py::test_zero_data_leakage_splits` | 🟢 Verified Fixed |
| **8** | **Variable Length Attention Distortion:** Padded zeros corrupted global temporal pooling. | Mask-aware temporal pooling and `src_key_padding_mask` attention in `src/models/st_transformer.py`. | `tests/test_dataset.py::test_batch_collate_and_masks` & `tests/test_model.py::test_st_transformer_forward_with_mask` | 🟢 Verified Fixed |
| **9** | **Sampling Rate Inconsistency:** Abstract/Ch 1 cited legacy 128 Hz while CS2 replay simulation operates at 64 Hz sub-tick. | Surgical manuscript audit replaced all 128 Hz / 7.8125 ms mentions with native 64 Hz / 15.625 ms; updated `generate_benchmark_dataset.py` to 64.0 Hz. | Document audit script (`audit_docx.py`) + `test_kinematics.py` | 🟢 Verified Fixed |
| **10** | **ATW Duration Calibration at 64 Hz:** Table 4 duration conversions held legacy 128 Hz times (0.50s, 0.25s, 4.0s). | Updated Table 4 (Table 7) & text: Buffer $\pm 64$ ticks = 1.00s, $L_{\min}=32$ ticks = 0.50s, $L_{\max}=512$ ticks = 8.00s; added $L_{\max}$ chunking in `atw_filter.py`. | `tests/test_parser.py::test_extract_active_tracking_windows_max_capping` | 🟢 Verified Fixed |
| **11** | **Feature Units Normalization Mismatch:** Table 5 specified angles in radians, but kinematics output raw degrees into dataset `feats / np.pi`. | Output `yaw` wrapped in $[-\pi, \pi]$ and `pitch` clamped in $[-\pi/2, \pi/2]$ in radians in `src/features/kinematics.py`, aligning with dataset normalizer. | `tests/test_kinematics.py` & `test_dataset.py` | 🟢 Verified Fixed |
| **12** | **Multi-Event Combat Buffers & Ban Manifest:** Extractor checked only `weapon_fire`, and cheater tagging failed on match-ID filenames. | `batch_processor.py` aggregates `weapon_fire`, `player_hurt`, `player_death`, and resolves cheaters via `banned_steamids.json` manifest. | `tests/test_parser.py` & `batch_processor.py` | 🟢 Verified Fixed |
| **13** | **Bipartite Graph Zero-Leakage Guarantee:** Fallback in cluster split partitioned only by player, risking match cross-over. | `dataset.py` fallback partitions by match ID and filters cross-match player segments, strictly enforcing $M_{\text{train}} \cap M_{\text{test}} = \emptyset$ AND $P_{\text{train}} \cap P_{\text{test}} = \emptyset$. | `tests/test_dataset.py::test_zero_data_leakage_matches_and_players_clusters` | 🟢 Verified Fixed |
| **14** | **XGBoost & Tabular MLP Baseline Alignment:** Benchmark used sklearn GB and default MLP instead of proposal specs. | Integrated native `xgboost` (XGBClassifier) into `ClassicalBaselines` and implemented PyTorch `TabularMLP` matching Table 28 ($48 \to 128 \to 64 \to 1$). | `tests/test_baselines.py` | 🟢 Verified Fixed |
| **15** | **Smurf Detection Decision Logic:** Model produced raw ELO and embeddings without an operational rank audit decision rule. | Implemented `SmurfDetector` class in `src/models/st_transformer.py` evaluating $\Delta_{\text{ELO}} = \hat{\text{ELO}} - \text{ELO}_{\text{reported}}$ and biometric tier centroid similarities. | `tests/test_model.py::test_smurf_detector_decision_rules` | 🟢 Verified Fixed |
| **16** | **Inference Latency & Low-FPR Evaluation:** Lacked latency profiling and fixed-FPR verification. | Implemented `profile_inference_latency` (streaming batch=1 & match throughput) and low-FPR operating point evaluation in `evaluate.py`. | `evaluate.py` execution (~7.3ms CPU streaming, ~526ms match audit) | 🟢 Verified Fixed |
| **17** | **Intrinsic Geodesic Curvature on $S^2$:** Ambient 3D curvature formula yielded $\kappa=1$ for great circles instead of zero. | Implemented intrinsic geodesic curvature $\kappa_g(t) = \frac{|\mathbf{v} \cdot (\mathbf{v}' \times \mathbf{v}'')|}{\|\mathbf{v}'\|^3 + \epsilon}$, yielding $\kappa_g = 0$ identically for unperturbed great-circle flicks. | `tests/test_kinematics.py::test_spherical_curvature_straight_vs_curved` | 🟢 Verified Fixed |
| **18** | **Batch Processor Typing & Ban/ELO Manifest Discovery:** Missing `Union`/`Iterable` imports and disconnected downloader staging manifests. | Added typing imports and automatic multi-path discovery of `staging/staging_manifest.json`, `banned_steamids.json`, and `player_elos.json`. | `src/data/batch_processor.py` & `src/data/demo_downloader.py` | 🟢 Verified Fixed |
| **19** | **CUDA Queue Latency Bias & Biometric Retrieval Metrics:** Asynchronous CUDA execution biased timing, and embedding lacked retrieval metrics. | Added `torch.cuda.synchronize()` in `evaluate.py` latency profiler, and added Top-1 ($P@1$) & Top-5 ($P@5$) nearest neighbor player retrieval evaluation. | `evaluate.py` & `tests/test_baselines.py` | 🟢 Verified Fixed |
| **20** | **Biometric Cluster Visualization & Centroid Smurf Logic:** Single t-SNE plot lacked player identity separation, and centroid similarities did not inform decision rule. | Generated dual t-SNE plots (`tsne_latent_space.png` colored by player ID and `tsne_aimbot_space.png` by aimbot label); updated `SmurfDetector` with biometric tier mismatch flags. | `benchmark.py` & `tests/test_model.py::test_smurf_detector_decision_rules` | 🟢 Verified Fixed |
| **21** | **Inference Preprocessing Pipeline Mismatch:** `analyze_match.py` & `demo_sample.py` used 128 Hz and per-window z-scores. | Standardized both scripts on native 64 Hz and domain-aware `normalize_kinematic_features` from `dataset.py`. | `analyze_match.py` & `demo_sample.py` | 🟢 Verified Fixed |
| **22** | **Strict Joint Zero-Leakage Guarantee:** Fallback in `dataset.py` previously reverted to player-only split, risking match overlap. | Replaced with iterative joint search and explicit `ValueError` if dataset cannot satisfy $M_{\text{train}} \cap M_{\text{test}} = \emptyset$ AND $P_{\text{train}} \cap P_{\text{test}} = \emptyset$. | `tests/test_dataset.py` & `src/data/dataset.py` | 🟢 Verified Fixed |
| **23** | **Cryptographic Salted SHA-256 Pseudonymization:** `batch_processor.py` wrote raw SteamIDs to Parquet metadata and filenames. | Implemented `pseudonymize_steamid(steamid, salt)` using salted SHA-256 mapped to 60-bit integers, fulfilling Philippine RA 10173. | `src/data/batch_processor.py` | 🟢 Verified Fixed |
| **24** | **Validation Threshold Calibration & Session Evaluation:** Evaluator derived threshold post-hoc from test ROC without session unit aggregation. | Added validation partition threshold calibration ($\tau^*$), test evaluation at $\tau^*$ with Rule of Three 95% upper bounds, and Match-Player session aggregation. | `evaluate.py` | 🟢 Verified Fixed |

---

## 📋 Month-by-Month Milestone Checklist

### August 2026: Foundations & Verified Core Architecture (100% COMPLETED)
- [x] **Task 1.1:** Fix Euler Angle Wrapping in Kinematics (`src/features/kinematics.py`).
- [x] **Task 1.2:** Spherical Geodesic Trajectory Curvature on unit sphere.
- [x] **Task 1.3:** 8–12 Hz Physiological Hand Tremor Power Spectrum via FFT.
- [x] **Task 1.4:** CS2 Demo Parser Module (`src/data/demo_parser.py` using `demoparser2`).
- [x] **Task 1.5:** Active Tracking Window (ATW) Extractor (`src/data/atw_filter.py`).
- [x] **Task 1.6:** PyTorch ST-Trans Architecture with dual heads (Focal Loss Aimbot + InfoNCE Smurf).
- [x] **Task 1.7:** Comprehensive Unit Testing (18/18 unit tests passing).

### September 2026: Ingestion Architecture, Rolling Buffer & CS2CD Integration (100% COMPLETED)
- [x] **Task 2.1:** Automated Faceit Open API & HLTV Multi-Tier Scraper in `src/data/demo_downloader.py` and `crawl_replays.py` CLI (supporting Beginner [L1-3], Intermediate [L4-6], Advanced [L7-8], and Pro [L9-10] with .dem.zst Backblaze stream & auto-decompression).
- [x] **Task 2.2:** Real replay ingestion pipeline + integration with public IEEE CoG 2025 CS2CD benchmark dataset (795 matches) as ground-truth cheater corpus. Enhanced with official API ban verification (`GET /players/{id}/bans`) and autonomous rolling buffer.
- [x] **Task 2.3:** Multi-threaded `src/data/batch_processor.py` to extract ATW Parquet telemetry stores at native 64 Hz with 30-degree spatial FOV cones and verified cheater-only tagging.
- [x] **Task 2.4:** Zero-leakage Train/Validation/Test splits (80/10/10) partitioned strictly by connected Match-ID and Player-ID clusters ($M_{\text{train}} \cap M_{\text{test}} = \emptyset, P_{\text{train}} \cap P_{\text{test}} = \emptyset$).

### October 2026: Full-Scale Model Training & Tuning
- [x] **Task 3.1:** PyTorch CUDA 13.2 setup for NVIDIA RTX 5060 (`sm_120`) in `D:\cs2_thesis_env` and initial GPU training on real Level 10 FACEIT dataset (1,758 segments) with zero false flags verified on pro match audit.
- [ ] **Task 3.2:** Extended 25–50 epoch GPU convergence run with AdamW and Cosine Annealing scheduler ($T_0=5, T_{mult}=2$).
- [ ] **Task 3.3:** Optimize InfoNCE contrastive temperature ($\tau \in [0.05, 0.15]$) and multi-task loss balance ($\lambda_1, \lambda_2$).
- [x] **Task 3.4:** Generate thesis publication-quality t-SNE and UMAP biometric latent cluster visualizations in `reports/` (Generated: `reports/tsne_latent_space.png`).

### November 2026: Benchmarks, Ablations & Manuscript Drafting
- [x] **Task 4.1:** Comparative benchmark study vs. XGBoost, Bi-LSTM, and MLP (Completed: `benchmark.py`, `reports/benchmark_summary.csv` using 48-stat feature engine).
- [x] **Task 4.2:** Feature ablation studies quantifying impact of 8–12 Hz Tremor, Minimum Jerk, Geodesic Curvature, and Raw Angles (Completed: `ablation.py`, `reports/ablation_study_summary.csv`, `reports/ablation_study.png`).
- [x] **Task 4.3:** Profile server-side inference throughput (Completed: `evaluate.py` profile benchmark validating ~7.3ms per ATW window and ~526ms CPU / <50ms GPU per 100-window match audit).
- [x] **Task 4.4:** Generate publication figures (ROC/PR curves, t-SNE latent skill clusters, multi-panel ablation bar charts).
- [ ] **Task 5.1:** Outline Defense (Nov 2026): Present Chapters 1–3 methodology, mathematical formulations, and Phase 2 synthetic pipeline verification proofs.
- [ ] **Task 5.2:** Final Defense (May–June 2027): Ingest full 2,000-match multi-tier corpus (795 CS2CD + FACEIT downloads) for complete empirical validation in Chapters 4–5.

### December 2026: Defense Presentation & Final Release
- [x] **Task 6.1:** Build 15–20 slide defense presentation deck blueprint & Master Guide (Completed: `OUTLINE_DEFENSE_MASTER_GUIDE.md`).
- [ ] **Task 6.2:** Rehearse outline defense presentation and live demo script (`demo_sample.py`).
- [ ] **Task 6.3:** Complete outline defense examination and incorporate panel recommendations.

---

## 📌 Post-Completion & Future Deployment Roadmap
*For detailed instructions, see the complete guide in [POST_COMPLETION_GUIDE.md](POST_COMPLETION_GUIDE.md).*
- [x] **Sub-500ms ONNX Export & Production Integration Protocol** (Documented)
- [x] **Oral Defense Rehearsal & Live Demo Runbook** (Documented)
- [x] **Academic Journal/Conference Publication Strategy** (IEEE CoG / ACM CHI PLAY) (Documented)
- [x] **HuggingFace & GitHub Open-Source Packaging** (Documented)

---

## 🔄 Protocol for Agents & Developers (Markdown-First Documentation Rule)
1. **Mandatory Documentation:** Whenever ANY new script, pipeline, feature, or analysis module is created or modified in this repository, it MUST be recorded immediately in `THESIS_TRACKER.md` and documented in `README.md` or `POST_COMPLETION_GUIDE.md`.
2. **Update Checklist:** Mark completed items with `[x]` and update the task status to `✅ Done (YYYY-MM-DD)`.
3. **Recalculate Progress:** Update the Phase Completion % table.
4. **Log Changes:** Add an entry in the Change Log below.

---

## 📝 Change Log & Milestone History
* **2026-10-06:** Implemented fully autonomous rolling buffer ingestion and deferred ban auditor pipeline (`--autonomous`, `--stage_recent`, `--audit_staging`) in `crawl_replays.py` and `src/data/demo_downloader.py`. Solved FACEIT's 30-day permanent CDN expiration constraint by caching fresh match candidates (< 48h old) into a local staging buffer (`staging/` with `staging_manifest.json`) and asynchronously auditing player accounts via `GET /players/{id}/bans` to automatically promote confirmed cheaters to `cheaters/` and graduate clean matches (> 21 days without infractions) to `clean/`. Updated Section 3.2.2 of the thesis proposal manuscript (`Thesis_Proposal_Medel-Gutierrez_Ch1-3.docx`) to formally document this rolling buffer ingestion protocol. Added `test_staging_manifest_and_audit` in `tests/test_downloader.py` (24/24 unit tests passing).
* **2026-10-03:** Completed Feature Ablation Study (`ablation.py`) across 6 conditions: Full Model (8D), w/o Tremor (7D), w/o Jerk (7D), w/o Curvature & Tortuosity (6D), Kinematics Only (6D), and Raw Coordinates Only (2D). Discovered major empirical finding: removing biomechanical invariants (Raw Coordinates Only) causes AUROC to collapse to 0.525 and FPR@95% to spike to 85.0%, proving that raw camera angles alone cannot discriminate aimbots without domain-specific biomechanical features. Generated publication figure `reports/ablation_study.png` and `reports/ablation_study_summary.csv`. Authored comprehensive `OUTLINE_DEFENSE_MASTER_GUIDE.md` containing 18-slide presentation blueprint, panel defensive Q&A matrix, and terminal demo protocols for November 2026 outline defense. All 23/23 unit tests passing.
* **2026-10-03:** Completed thesis proposal alignment audit (`Thesis_Proposal_Medel-Gutierrez_Ch1-3.docx` vs codebase). Implemented 4 critical code-level fixes: (1) Added skewness and kurtosis to `benchmark.py` tabular feature extraction reaching all 48 summary statistics per Table 7; (2) Aligned BiLSTM baseline to 64 hidden units (128 bidirectional) in `benchmark.py`; (3) Constrained 8-12 Hz Tremor Band Power denominator to voluntary motor bandwidth [1.0, 30.0] Hz in `kinematics.py` per Eq 10; (4) Set contrastive loss weight to 0.5 in `benchmark.py`. Created persistent `AGENTS.md` and `SKILL.md` (in root, `.agents/skills/`, and `skills/`) establishing coding standards, mathematical citations, and discrepancy registry. All 22/22 unit tests passing.
* **2026-09-04:** Implemented official FACEIT ban verification (`GET /players/{id}/bans`) and pre-ban match filtering (`match.finished_at <= ban.starts_at`) in `src/data/demo_downloader.py`. Added `--banned_file` and `--scan_cheaters` CLI options in `crawl_replays.py`. Calibrated `dataset.py` split default to 80/10/10, switched ELO loss to Smooth L1 (Eq 18), aligned model defaults to $d_{model}=128, H=8, d_{ff}=512$, added Spearman rank correlation to `evaluate.py`. Verified with 22/22 unit tests passing.
* **2026-08-21:** Implemented multi-tier scraping across all 10 FACEIT skill tiers (`beginner` [L1-3], `intermediate` [L4-6], `advanced` [L7-8], `pro` [L9-10]) in `crawl_replays.py` with direct Backblaze CDN streaming and auto ATW extraction.
* **2026-08-21:** Configured native PyTorch CUDA 13.2 for NVIDIA RTX 5060 on D: drive (`D:\cs2_thesis_env\`). Performed initial GPU training on real Level 10 FACEIT dataset (1,758 segments) and verified 0 false positives across 1.53M ticks in pro match audit. Phase 3 initiated.
* **2026-08-20:** Master roadmap updated and synchronized with approved Thesis Concept Paper (Medel & Gutierrez, 2026). All 6 phases structured with targeted deadlines to ensure oral defense by mid-December 2026.
