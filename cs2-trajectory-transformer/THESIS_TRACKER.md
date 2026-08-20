# 📌 CS2 Trajectory Transformer — Thesis Master Tracker & Roadmap

**Thesis Topic:** Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers  
**Author:** Dishann Gutierrez  
**Degree:** Bachelor of Science in Computer Science (Major in Data Science)  
**Target Completion:** December 2026  
**Last Updated:** August 20, 2026  

---

## 🚦 Overall Project Status Summary

| Phase | Description | Target Window | Status | Completion % |
| :--- | :--- | :--- | :---: | :---: |
| **Phase 1** | Mathematical Foundations & Parser Engineering | Aug 21 – Sep 12 | 🟢 COMPLETED | 100% |
| **Phase 2** | Dataset Assembly & Telemetry Store (2,000 Demos) | Sep 13 – Oct 03 | 🟡 IN PROGRESS | 75% |
| **Phase 3** | ST-Trans Architecture & InfoNCE Training | Oct 04 – Oct 31 | 🟡 IN PROGRESS | 85% |
| **Phase 4** | Evaluation, Ablation Studies & Benchmarks | Nov 01 – Nov 20 | ⚪ NOT STARTED | 0% |
| **Phase 5** | Manuscript Writing, Defense Prep & Release | Nov 21 – Dec 20 | ⚪ NOT STARTED | 0% |

---

## 📋 Detailed Task Checklist

### Phase 1: Mathematical Foundations & Parser Engineering (Aug 21 – Sep 12)
- [x] **Task 1.1: Fix Euler Angle Wrapping in Kinematics**
  - Implemented circular difference `(Δθ + π) % (2π) - π` in `src/features/kinematics.py`.
  - Verified with unit tests that $\pm 180^\circ$ boundary wraps produce clean geodesic deltas without artificial jerk spikes.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 1.2: Spherical Trajectory Curvature**
  - Implemented 3D sight-line vector cross product curvature on unit sphere.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 1.3: 8–12 Hz Physiological Tremor Power Spectrum**
  - Implemented FFT/Welch sliding-window band power extraction in `src/features/kinematics.py`.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 1.4: CS2 Demo Parser Module (`demoparser2`)**
  - Built `src/data/demo_parser.py` (`CS2DemoParser`) to extract 128-tick coordinates, viewangles, and weapon fire events from `.dem` files.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 1.5: Active Tracking Window (ATW) Extractor**
  - Implemented 3D geometric visual cone ($30^\circ$ enemy FOV) and temporal combat buffers ($\pm 64$ ticks around shot events) in `src/data/atw_filter.py`.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 1.6: Unit Testing & Verification**
  - Created test suite in `tests/test_kinematics.py` and `tests/test_parser.py` (7/7 tests passing).
  - Status: ✅ Done (2026-08-20)

---

### Phase 2: Dataset Assembly & Telemetry Store (Sep 13 – Oct 03)
- [ ] **Task 2.1: Demo Replay Collection (2,000 Matches)**
  - Source 1,000 clean demos (Faceit / HLTV across rank tiers: Silver to Pro).
  - Source 1,000 cheater demos (Faceit banned player registry).
  - Status: ⏳ In Progress (Ready for download)
- [x] **Task 2.2: Batch Telemetry Processing**
  - Built `src/data/batch_processor.py` for parallel extraction of `.dem` replays into Parquet format.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 2.3: Data Leakage Prevention Partitioning**
  - Strict Player-ID and Match-ID split algorithm in `src/data/dataset.py` (Zero Data Leakage).
  - Status: ✅ Done (2026-08-20)
- [x] **Task 2.4: PyTorch Dataset & DataLoader Pipeline**
  - Implemented `CS2TrajectoryDataset` and `collate_trajectory_batch` with attention masks and sequence padding.
  - Status: ✅ Done (2026-08-20)

---

### Phase 3: Spatial-Temporal Transformer Implementation & Training (Oct 04 – Oct 31)
- [x] **Task 3.1: Upgraded ST-Trans Architecture**
  - 4-layer Transformer Encoder, 8 attention heads, $d_{model} = 128$, GELU activations, mask-aware pooling in `src/models/st_transformer.py`.
  - Status: ✅ Done (2026-08-20)
- [x] **Task 3.2: Dual-Head Implementation**
  - Aimbot Head: Binary classification with Sigmoid and Focal Loss.
  - Smurf Head: 32-dim unit-normalized latent projection with **Supervised InfoNCE Contrastive Loss** + calibrated ELO estimator in `src/models/losses.py`.
  - Status: ✅ Done (2026-08-20)
- [ ] **Task 3.3: Model Training on Full Replay Dataset**
  - `train.py` pipeline ready with AdamW and Cosine Annealing. Awaiting 2,000 match dataset batching.
  - Status: ⏳ Pending Data

---

### Phase 4: Evaluation, Ablation Studies & Benchmarking (Nov 01 – Nov 20)
- [ ] **Task 4.1: Quantitative Benchmarks vs. Classical Baselines**
  - Compare ST-Trans against Random Forest, XGBoost, LSTM, and MLP.
  - Target: False Positive Rate $< 0.01\%$, AUROC $> 0.98$.
  - Status: ⚪ Not Started
- [ ] **Task 4.2: Feature Ablation Studies**
  - Evaluate model without Tremor band power.
  - Evaluate model without Angular Jerk.
  - Evaluate model without Active Tracking Window filter.
  - Status: ⚪ Not Started
- [ ] **Task 4.3: Latency & Server-Side Feasibility Test**
  - Benchmark inference latency per match (Target $< 500$ ms).
  - Status: ⚪ Not Started
- [ ] **Task 4.4: Latent Space & Attention Visualizations**
  - Generate t-SNE / UMAP plots of player skill embeddings and cheat clusters.
  - Generate cross-attention trajectory heatmaps.
  - Status: ⚪ Not Started

---

### Phase 5: Manuscript Writing, Defense Prep & Release (Nov 21 – Dec 20)
- [ ] **Task 5.1: Thesis Manuscript Chapters (1–5)**
  - Draft Chapters: Introduction, Literature Review, Methodology, Results, Discussion & Conclusion.
  - Status: ⚪ Not Started
- [ ] **Task 5.2: Defense Slide Deck & Mock Presentations**
  - Build slide deck with high-res figures and rehearsed speaking scripts.
  - Status: ⚪ Not Started
- [ ] **Task 5.3: Codebase Polish & Final Defense**
  - Prepare reproducible environment, pre-trained weights, and conduct oral defense.
  - Status: ⚪ Not Started

---

## 📝 Change Log & Milestone History
* **2026-08-20:** Phase 1 100% completed. Phase 2 (75%) and Phase 3 (85%) pipelines fully implemented and tested (13/13 unit tests passing).
