# 📌 CS2 Trajectory Transformer — Thesis Master Tracker & Roadmap

**Thesis Title:** Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers  
**Authors:** Medel & Gutierrez  
**Degree Program:** Bachelor of Science in Computer Science (Major in Data Science)  
**Academic Year:** 2026  
**Target Defense Date:** Mid-December 2026  
**Last Updated:** August 20, 2026  
**Post-Completion Guide:** See [POST_COMPLETION_GUIDE.md](POST_COMPLETION_GUIDE.md)  

---

## 🚦 Master Timeline & Phase Overview (Target: Dec 2026)

| Phase | Description | Target Window | Status | Completion % |
| :--- | :--- | :--- | :---: | :---: |
| **Phase 1** | Mathematical Foundations, 6 Features & ST-Trans Architecture | Aug 20 – Aug 21 | 🟢 COMPLETED | 100% |
| **Phase 2** | Real Replay Ingestion & Scraper Pipeline | Aug 22 – Sep 30 | 🟢 COMPLETED | 100% |
| **Phase 3** | Full-Scale GPU Training & InfoNCE Contrastive Tuning | Oct 01 – Oct 31 | 🟡 IN PROGRESS | 25% |
| **Phase 4** | Comparative Benchmarks (XGBoost/BiLSTM), Ablation & Latency | Nov 01 – Nov 18 | ⚪ SCHEDULED | 0% |
| **Phase 5** | Complete Thesis Manuscript Compilation (Chapters 1–5) | Nov 10 – Dec 05 | ⚪ SCHEDULED | 0% |
| **Phase 6** | Slide Deck, Mock Rehearsals & Final Oral Defense | Dec 06 – Dec 20 | ⚪ SCHEDULED | 0% |

---

## 🔍 Critical Technical Flaws Resolution Audit

| # | Identified Critical Flaw / Gap | Resolution Implementation | Verification Test | Status |
| :-: | :--- | :--- | :--- | :---: |
| **1** | **Euler Angle Discontinuity:** Boundary jumps across $\pm 180^\circ$ caused false $358^\circ$ velocity/jerk spikes. | `wrap_angle_rad` using $((\Delta \theta + \pi) \pmod{2\pi}) - \pi$ in `src/features/kinematics.py`. | `tests/test_kinematics.py::test_euler_angle_wrapping` | 🟢 Verified Fixed |
| **2** | **Ad-Hoc Planar Curvature Proxy:** 2D curvature formula invalid on spherical viewing coordinates. | True 3D geodesic sight vector cross product curvature $\kappa = \frac{\|\mathbf{v}' \times \mathbf{v}''\|}{\|\mathbf{v}'\|^3 + \epsilon}$ on unit sphere $S^2$. | `tests/test_kinematics.py::test_spherical_curvature_straight_vs_curved` | 🟢 Verified Fixed |
| **3** | **Missing 8–12 Hz Tremor PSD:** Proposal claimed neuromuscular frequency extraction, but code lacked FFT. | `compute_tremor_band_power` using sliding-window Hanning-windowed FFT relative power in $[8, 12]$ Hz. | `tests/test_kinematics.py::test_tremor_band_power_detection` | 🟢 Verified Fixed |
| **4** | **Replay Idle Walking Noise:** Analyzing entire 45-min matches diluted combat aimbot signals with 70%+ navigation noise. | Active Tracking Window (`src/data/atw_filter.py`) extracting $30^\circ$ enemy visual cones and $\pm 64$-tick combat event buffers. | `tests/test_parser.py::test_relative_fov_geometry` & `test_extract_active_tracking_windows` | 🟢 Verified Fixed |
| **5** | **Lack of Contrastive Smurf Embedding:** Initial model only performed scalar regression without biometric latent clustering. | 32-dim unit-normalized projection head optimized via `SupervisedInfoNCELoss` in `src/models/losses.py`. | `tests/test_model.py::test_infonce_contrastive_loss` | 🟢 Verified Fixed |
| **6** | **Aimbot Class Imbalance:** Sparse cheater engagement windows lead standard BCE to majority-class collapse. | Implemented `FocalLoss` ($\alpha=0.25, \gamma=2.0$) in `src/models/losses.py` down-weighting easy background samples. | `tests/test_model.py::test_focal_loss` | 🟢 Verified Fixed |
| **7** | **Data Leakage in Splits:** Splitting randomly across ticks or rounds of the same player causes memorization. | `create_partitioned_dataloaders` enforcing pairwise disjoint player-ID and match-ID splits. | `tests/test_dataset.py::test_zero_data_leakage_splits` | 🟢 Verified Fixed |
| **8** | **Variable Length Attention Distortion:** Padded zeros corrupted global temporal pooling. | Mask-aware temporal pooling and `src_key_padding_mask` attention in `src/models/st_transformer.py`. | `tests/test_dataset.py::test_batch_collate_and_masks` & `tests/test_model.py::test_st_transformer_forward_with_mask` | 🟢 Verified Fixed |

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

### September 2026: Data Ingestion & Batch Store (2,000 Matches) (100% COMPLETED)
- [x] **Task 2.1:** Automated Faceit Open API & HLTV Replay Scraper in `src/data/demo_downloader.py` and `download_demos.py` CLI (with .dem.zst support).
- [x] **Task 2.2:** Ingest clean demos (Faceit Level 10 Pro / FPL) + cheater dataset (Faceit Ban Registry & High-Fidelity Synthetic Benchmark Suite).
- [x] **Task 2.3:** Run multi-threaded `src/data/batch_processor.py` to extract ATW Parquet telemetry stores (Verified: 879 ATWs extracted & validated with PyTorch DataLoader).
- [x] **Task 2.4:** Generate zero-leakage Train/Validation/Test splits (80/10/10) partitioned strictly by Player/Match IDs.

### October 2026: Full-Scale Model Training & Tuning
- [ ] **Task 3.1:** Train ST-Trans on GPU with AdamW and Cosine Annealing scheduler.
- [ ] **Task 3.2:** Optimize InfoNCE contrastive temperature ($\tau \in [0.05, 0.15]$) and multi-task loss weights.
- [ ] **Task 3.3:** Save finalized production model checkpoint (`models/checkpoints/best_model.pt`).

### November 2026: Benchmarks, Ablations & Manuscript Drafting
- [ ] **Task 4.1:** Comparative benchmark study vs. XGBoost, Bi-LSTM, and MLP on real data.
- [ ] **Task 4.2:** Feature ablation studies (quantifying impact of 8–12 Hz Tremor and Minimum Jerk).
- [ ] **Task 4.3:** Profile server-side inference throughput (validating $< 500$ ms per match latency).
- [ ] **Task 4.4:** Generate publication figures (ROC/PR curves, t-SNE latent skill clusters).
- [ ] **Task 5.1:** Draft complete 5-chapter thesis manuscript (Intro, Lit Review, Methodology, Results, Discussion).

### December 2026: Defense Presentation & Final Release
- [ ] **Task 6.1:** Build 15–20 slide defense presentation deck.
- [ ] **Task 6.2:** Rehearse defense presentation and live demo script (`demo_sample.py`).
- [ ] **Task 6.3:** Conduct oral defense and submit camera-ready thesis documentation.

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
* **2026-08-20:** Master roadmap updated and synchronized with approved Thesis Concept Paper (Medel & Gutierrez, 2026). All 6 phases structured with targeted deadlines to ensure oral defense by mid-December 2026.
