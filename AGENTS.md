# 🤖 CS2 Trajectory Transformer — Agent Persona, Coding Standards & Thesis Knowledge Base

> **Thesis Title:** Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers  
> **Authors:** Judah Ben Hur L. Medel & Dishann G. Gutierrez  
> **Adviser:** Vera Kim S. Tequin  
> **Institution:** College of Information and Computing (CIC), University of Southeastern Philippines (USeP), Bo. Obrero, Davao City  
> **Degree Program:** Bachelor of Science in Computer Science (Major in Data Science)  
> **Academic Year:** 2026–2027 | **Outline Defense:** November 2026 | **Target Final Defense:** May–June 2027 (2nd Semester)  
> **Tracking Documentation:** [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md) | [`POST_COMPLETION_GUIDE.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/POST_COMPLETION_GUIDE.md)

---

## 🎯 1. Core Mission & Research Philosophy

### 1.1 The Research Problem
1. **Failure of Kernel-Level Anti-Cheats (Ring 0):** Modern cheat ecosystems have shifted to external hardware: Direct Memory Access (DMA) PCIe cards and microcontroller-based HID mouse smoothers (Arduino/Raspberry Pi). Because cheat code runs entirely on secondary machines or firmware, client-side kernel drivers (e.g., Riot Vanguard, Easy Anti-Cheat) suffer from fundamental architectural blindness while introducing severe system instability (BSODs) and user privacy liabilities.
2. **Evasion of Coarse Server Heuristics:** Existing server heuristics rely on aggregate post-match metrics (K/D ratio, headshot percentage). Humanized, low-FOV aimbots easily bypass these by applying subtle micro-corrections solely during decisive 200ms flick moments.
3. **Smurfing and Matchmaking Degradation:** Rank spoofing ruins competitive fairness, but competitive matchmaking engines lack continuous biometric skill profilers capable of evaluating a player's true motor proficiency from fine-grained kinematic execution.

### 1.2 The Solution & Biomechanical Invariants
The human motor system is governed by immutable biological and neuromuscular constraints:
- **Flash & Hogan Minimum Jerk Optimization:** Biological motor control plans trajectories that minimize the third time derivative of position ($\int j(t)^2 dt$), generating bell-shaped velocity curves and continuous acceleration. Algorithmic aimbots display instantaneous torque changes and extreme jerk spikes.
- **8–12 Hz Physiological Hand Tremor:** Involuntary motor-unit firing causes a natural micro-tremor in the $8.0\text{--}12.0\text{ Hz}$ frequency band. Legitimate human aiming exhibits significant power in this band ($15\%\text{--}45\%$ of active motor spectrum). Aimbots either completely lack this resonance ($\text{TBP} < 2\%$) or inject artificial Gaussian noise without biological phase coherence.
- **Spherical Geodesic Curvature ($\kappa_t$):** Gaze trajectories operate on a spherical surface $S^2$, not Euclidean 2D space. Curvature is computed via 3D sight-line vector cross products.
- **Shortest-Path Euler Angle Wrapping:** View angles across coordinate boundaries ($\pm 180^\circ$) must be wrapped to $[-\pi, \pi]$ to prevent false $358^\circ$ coordinate jump spikes.
- **Spatial-Temporal Trajectory Transformer (ST-Trans):** A dual-head deep transformer architecture combining:
  - **Head A (Aimbot Classification):** Trained with Binary Focal Loss ($\alpha=0.25, \gamma=2.0$) to overcome the extreme class imbalance ($< 1\%$ cheater windows).
  - **Head B (Smurf Biometric Embedding):** 32-dimensional unit-normalized embedding trained with Supervised InfoNCE Contrastive Loss ($\tau=0.07$) + auxiliary continuous ELO regression head with Smooth L1 loss.

---

## 📁 2. Repository Architecture & Directory Mapping

```
cs2-trajectory-transformer/
├── AGENTS.md                               <- This file: Project context, coding rules, & persistent agent memory
├── README.md                               <- User-facing project documentation
├── THESIS_TRACKER.md                       <- Master roadmap, milestone checklist, & audit history
├── POST_COMPLETION_GUIDE.md                <- Deployment runbook, ONNX export, & defense strategies
├── requirements.txt                        <- Python package dependencies
├── setup_env.ps1 / .bat / .sh              <- Automated 1-click virtual environment setup scripts
│
├── crawl_replays.py                        <- Automated FACEIT API scraper & Backblaze CDN downloader
├── analyze_match.py                        <- Single-match forensic audit CLI
├── demo_sample.py                          <- End-to-end pipeline verification demo
├── train.py                                <- ST-Trans training pipeline (AdamW + Cosine Annealing)
├── evaluate.py                             <- Comprehensive evaluation (AUROC, AUPRC, FPR@95%TPR, ELO MAE)
├── benchmark.py                            <- Comparative benchmark suite (Random Forest, XGBoost, MLP, BiLSTM)
├── generate_benchmark_dataset.py           <- High-fidelity synthetic trajectory generator
├── inspect_checkpoint.py                   <- Weight inspection utility for saved PyTorch checkpoints
├── visualize.py                            <- Publication-ready ROC/PR curves & t-SNE latent cluster visualizer
│
├── src/                                    <- Core Python package source modules
│   ├── features/
│   │   └── kinematics.py                   <- Biomechanical feature engine (Wrapping, Velocity, Jerk, Curvature, Tremor)
│   ├── models/
│   │   ├── st_transformer.py               <- PyTorch Spatial-Temporal Trajectory Transformer (ST-Trans)
│   │   ├── losses.py                       <- Supervised InfoNCE Loss & Binary Focal Loss
│   │   └── baselines.py                    <- BiLSTM & Classical baseline models (RF, GradientBoosting, MLP)
│   └── data/
│       ├── demo_parser.py                  <- High-speed CS2 replay parser wrapping demoparser2
│       ├── atw_filter.py                   <- Active Tracking Window (ATW) spatial-temporal extractor
│       ├── dataset.py                      <- Zero-leakage PyTorch Dataset & batch collator
│       ├── demo_downloader.py              <- FACEIT API polite scraper & archive decompressor
│       └── batch_processor.py              <- Multiprocessing ATW Parquet extraction pipeline
│
├── tests/                                  <- Pytest automated test suite (26/26 verified passing)
│   ├── test_kinematics.py                  <- Euler wrapping, curvature, & tremor PSD unit tests
│   ├── test_model.py                       <- ST-Trans forward pass, masks, & loss function tests
│   ├── test_dataset.py                     <- Zero data leakage splits & batch collation tests
│   ├── test_parser.py                      <- ATW geometry & demoparser2 integration tests
│   ├── test_downloader.py                  <- Download, decompression, & rate limit backoff tests
│   └── test_baselines.py                   <- Baseline model architectures & inference tests
│
├── data/                                   <- Local and external data directories
│   ├── raw_demos/                          <- Downloaded .dem replays (clean/ & cheaters/)
│   └── processed_parquet/                  <- Extracted 8D ATW Parquet feature files
│
├── models/checkpoints/                     <- Saved PyTorch model checkpoint weights (best_model.pt)
└── reports/                                <- Generated figures (roc_pr_curve.png, tsne_latent_space.png)
```

---

## 🛠️ 3. Execution Protocols & Environment

- **Python Virtual Environment:**  
  Always use the dedicated virtual environment located at:  
  `cs2-trajectory-transformer\venv\Scripts\python.exe` (or `D:\cs2_thesis_env\Scripts\python.exe` if on the primary GPU workstation).
- **Run Unit Tests:**
  ```powershell
  & "cs2-trajectory-transformer\venv\Scripts\python.exe" -m pytest "cs2-trajectory-transformer\tests"
  ```
- **Train ST-Trans Model:**
  ```powershell
  & "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\train.py" --data_dir "cs2-trajectory-transformer\data\processed_parquet" --epochs 50 --batch_size 32
  ```
- **Run Evaluation:**
  ```powershell
  & "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\evaluate.py" --data_dir "cs2-trajectory-transformer\data\processed_parquet" --model_path "cs2-trajectory-transformer\models\checkpoints\best_model.pt"
  ```
- **Run Comparative Benchmarks:**
  ```powershell
  & "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\benchmark.py"
  ```

---

## 💻 4. Proper Coding Techniques & Standards

All code contributions MUST adhere strictly to the following standards:

### 4.1 Mathematical & Computational Rigor
1. **Vectorization Over Python Loops:** Never iterate over simulation ticks with `for i in range(len(ticks))`. Use vectorized NumPy, SciPy, or PyTorch tensor operations.
2. **Defensive Numerical Stability:**
   - Always add small numerical epsilons to division denominators: `denom = denom + 1e-6` or `1e-9`.
   - Before taking `arccos(dot_product)`, strictly clamp the dot product to the valid range: `np.clip(dot, -1.0, 1.0)` or `torch.clamp(dot, -1.0, 1.0)`.
   - In probability functions, clamp inputs away from $0$ and $1$ to prevent `log(0)` / `NaN`: `inputs.clamp(min=1e-6, max=1.0 - 1e-6)`.
3. **Euler Coordinate Wrapping:** Any numerical differentiation on camera yaw or pitch must pass through `wrap_angle_rad(d_angle)` using `((d_angle + np.pi) % (2.0 * np.pi)) - np.pi`.

### 4.2 PyTorch Deep Learning Best Practices
1. **Mask-Aware Pooling:** Sequence lengths in ATWs vary ($32 \le L \le 512$). All temporal pooling over transformer representations MUST multiply by the attention mask and normalize by valid sequence length:
   ```python
   mask_expanded = attention_mask.unsqueeze(-1).float()
   pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
   ```
2. **Key Padding Mask:** Remember PyTorch's `nn.TransformerEncoder` convention: `src_key_padding_mask` requires `True` for positions that should be **ignored** (padding) and `False` for valid tokens. Use `src_key_padding_mask = ~attention_mask`.
3. **Safe Checkpoint Loading:** Always pass `weights_only=True` when invoking `torch.load()` to mitigate arbitrary code execution risks.
4. **Evaluation Isolation:** Always wrap evaluation inference blocks in `with torch.no_grad():` and call `model.eval()`.

### 4.3 Code Structure & Clean Architecture
1. **Strict Type Annotations:** Every function signature must include complete type hints from `typing` (`List`, `Dict`, `Tuple`, `Optional`, `Union`) and explicit return types.
2. **Standardized Docstrings:** Follow NumPy/SciPy docstring style. Specify:
   - Brief summary of the algorithm or mathematical role.
   - Exact physical units of inputs and outputs (e.g., `rad/s`, `rad/s^2`, `degrees`, `normalized [0.0, 1.0]`).
   - Tensor shapes for multidimensional inputs (e.g., `[batch_size, seq_len, 8]`).
3. **Windows Multiprocessing Safety:** Any script invoking `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` MUST be protected by `if __name__ == "__main__":` to prevent recursive process spawning on Windows.

---

## 📝 5. Mathematical Commenting & Citation Standards

Every mathematical formula implemented in code MUST be explicitly tagged with its corresponding Chapter 3 thesis equation number and scientific justification:

```python
# ==============================================================================
# Thesis Reference: Chapter 3, Equation (5) — Great-Circle Angular Velocity
# omega_t = (1 / dt) * sqrt((wrap(Delta p_t))^2 + (cos(p_t) * wrap(Delta y_t))^2)
# Units: rad/s on unit sphere S^2 (dt = 1.0 / 64.0 s, native CS2 sub-tick simulation)
# ==============================================================================
```

```python
# ==============================================================================
# Thesis Reference: Chapter 3, Equation (10) — 8-12 Hz Tremor Band Power (TBP)
# TBP = (sum_{f=8}^{12} |X(f)|^2) / (sum_{f=1}^{30} |X(f)|^2 + eps)
# Ratio of power in human neuromuscular tremor band relative to total motor band.
# ==============================================================================
```

```python
# ==============================================================================
# Thesis Reference: Chapter 3, Equation (16) — Binary Focal Loss
# FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
# Down-weights easy background negative samples (gamma=2.0, alpha=0.25).
# ==============================================================================
```

---

## 🔍 6. Comprehensive Proposal Alignment & Discrepancy Registry

The following table documents the audited alignment between the approved thesis proposal (`Thesis_Proposal_Medel-Gutierrez_Ch1-3.docx`) and the current codebase implementation. **Maintain this registry when refactoring or adding modules:**

| # | Feature / Module | Proposal Specification (Ch 1–3) | Codebase Implementation | Current Status & Action Item |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Tabular Baseline Features** | **48 summary statistics** across 8 channels: mean, std, min, max, skewness, kurtosis ($6 \times 8 = 48$). (Sec 3.2.8, Table 7) | `benchmark.py` extracts 48 statistics using `scipy.stats.skew` and `kurtosis` with `nan_to_num` defense. | 🟢 **Resolved (2026-10-03):** Aligned with Table 7. All 48 features active. |
| **2** | **BiLSTM Baseline Hidden Dimension** | 2 BiLSTM layers, `hidden_dim = 64` (128 bidirectional units). (Sec 3.2.8, Table 7) | `benchmark.py` instantiates `BiLSTMBaseline(hidden_dim=64)`. | 🟢 **Resolved (2026-10-03):** Aligned with Table 7 and `baselines.py`. |
| **3** | **Tremor Band Power Denominator** | Ratio of $[8, 12]\text{ Hz}$ power relative to active motor bandwidth $[1.0, 30.0]\text{ Hz}$. (Sec 3.2.4, Eq 10) | `kinematics.py` enforces `total_mask = (freqs >= 1.0) & (freqs <= 30.0)`. | 🟢 **Resolved (2026-10-03):** Aligned with Eq (10). Native 64 Hz sub-tick sampling active. |
| **4** | **Contrastive Loss Weight $\lambda_{\text{con}}$** | Composite loss specifies $\lambda_{\text{focal}}=1.0$, $\lambda_{\text{infonce}}=0.5$, $\lambda_{\text{elo}}=0.2$. (Sec 3.2.6, Eq 14) | `benchmark.py` and `train.py` both use `0.5 * loss_con`. | 🟢 **Resolved (2026-10-03):** Aligned with Eq (14). Consistent across scripts. |
| **5** | **Normalization Strategy** | Global dataset standardization or domain-aware physical scaling. (Sec 3.2.5) | `dataset.py` computes global training statistics; prevents per-window amplitude collapse. | 🟢 **Resolved (2026-10-08):** Per-window z-score replaced by global scaler. |
| **6** | **ATW Max Length Capping** | Capped at $L_{\max} = 512$ ticks ($\approx 8.0$ s at 64 Hz). (Sec 3.2.3, Table 4) | `atw_filter.py` chunks long duels at $L_{\max}=512$; DataLoader collator enforces tensor cap. | 🟢 **Resolved (2026-10-08):** Enforced at both extraction and collation stages. |
| **7** | **Zero-Leakage Split Guarantee** | Disjoint by both Player ID ($P_{\text{train}} \cap P_{\text{test}} = \emptyset$) AND Match ID ($M_{\text{train}} \cap M_{\text{test}} = \emptyset$). (Sec 3.2.7, Eq 18) | `dataset.py` partitions by connected Match/Player clusters with fallback cross-player filtering. | 🟢 **Resolved (2026-10-08):** Joint cluster partitioning strictly guarantees zero leakage. |
| **8** | **Thesis Manuscript Status** | Chapters 1–3 approved. Outline Defense: November 2026; Target Final Defense: May–June 2027. | Proposal Ch 1–3 updated with 64 Hz sub-tick physics, CS2CD dataset (795 matches), and AntiCheatPT baseline. | 🟢 **Aligned:** Follow [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md) milestone schedule. |
| **9** | **Feature Units Normalization** | View angles specified in radians in Table 5/16; degrees would corrupt `feats / np.pi`. | `kinematics.py` outputs wrapped radians $[-\pi, \pi]$ and $[-\pi/2, \pi/2]$, matching Table 5 and dataset normalizer. | 🟢 **Resolved (2026-10-08):** Feature units and normalizer perfectly aligned. |
| **10** | **Multi-Event Combat Buffering** | Buffers anchored around `weapon_fire`, `player_hurt`, `player_death` with ground truth cheater tracking. | `batch_processor.py` aggregates all 3 combat events and queries `banned_steamids.json` manifest. | 🟢 **Resolved (2026-10-08):** Full combat event coverage and robust cheater tagging. |
| **11** | **XGBoost & Tabular MLP Baseline** | Benchmark specified XGBoost and $48 \to 128 \to 64 \to 1$ MLP. | Native `xgboost` integrated into `ClassicalBaselines` and PyTorch `TabularMLP` implemented. | 🟢 **Resolved (2026-10-08):** Exact architecture alignment with Table 28. |
| **12** | **Smurf Detection Decision Logic** | Continuous ELO discrepancy decision rule $\Delta_{\text{ELO}} \ge \tau$ and latent biometric cluster matching. | Implemented `SmurfDetector` class in `src/models/st_transformer.py` classifying smurf severity tiers. | 🟢 **Resolved (2026-10-08):** Operational decision rule and centroid matching fully implemented. |
| **13** | **Inference Latency Profiling** | Server-side throughput validation ($< 500$ ms per match, $< 5$ ms per ATW window). | `evaluate.py` profiles batch=1 streaming latency (~7.3ms CPU) and match audit throughput (~526ms CPU / <50ms GPU). | 🟢 **Resolved (2026-10-08):** Formal profiling benchmark active in `evaluate.py`. |
| **14** | **ATW Duration Calibration at 64 Hz** | Table 4 duration conversions held legacy 128 Hz times (0.50s, 0.25s, 4.0s). | Table 4 (Table 7) & text updated: Buffer $\pm 64$ ticks = 1.00s, $L_{\min}=32$ ticks = 0.50s, $L_{\max}=512$ ticks = 8.00s. | 🟢 **Resolved (2026-10-08):** All time constants consistent across manuscript. |
| **15** | **Intrinsic Geodesic Curvature on $S^2$** | Great-circle flicks should possess $\kappa_g = 0.0$; ambient 3D curvature formula yielded $\kappa=1.0$. (Sec 3.2.4, Eq 9) | `kinematics.py` implements $\kappa_g(t) = \frac{|\mathbf{v} \cdot (\mathbf{v}' \times \mathbf{v}'')|}{\|\mathbf{v}'\|^3 + \epsilon}$, yielding $\kappa_g = 0$ for great circles. | 🟢 **Resolved (2026-10-09):** Intrinsic geodesic curvature aligned with differential geometry and Eq (9). |
| **16** | **Batch Processor Manifest Resolution** | Ingestion must discover downloader staging manifests and link real SteamIDs and player ELO ratings. | `batch_processor.py` dynamically resolves `staging/staging_manifest.json`, `banned_steamids.json`, and `player_elos.json`. | 🟢 **Resolved (2026-10-09):** Real match labels and ELO profiles automatically ingested. |
| **17** | **Biometric Identification Evaluation** | Supervised InfoNCE metric learning must evaluate held-out player retrieval and biometric clustering. | `evaluate.py` computes Top-1 ($P@1$) & Top-5 ($P@5$) retrieval; `benchmark.py` generates player-colored t-SNE plot. | 🟢 **Resolved (2026-10-09):** Publication-grade retrieval metrics and latent visualizations active. |
| **18** | **Ethics & RA 10173 Pseudonymization** | Raw SteamIDs identify human accounts; Philippine Data Privacy Act requires pseudonymization. (Sec 3.3.1) | Proposal specifies salted SHA-256 one-way hashing ($h = \text{SHA256}(\text{SteamID} \,\|\, \text{salt})$) severing link to PII. | 🟢 **Resolved (2026-10-09):** RA 10173 compliance and research ethics explicitly codified. |
| **19** | **Inference Preprocessing Pipeline Mismatch** | Inference must strictly match training normalization and 64 Hz sub-tick sampling. | Standardized `analyze_match.py` & `demo_sample.py` on native 64 Hz and domain-aware `normalize_kinematic_features` from `dataset.py`. | 🟢 **Resolved (2026-10-09):** Unified feature extraction and physical scaling. |
| **20** | **Strict Joint Zero-Leakage Guarantee** | Joint disjoint constraint ($M_{\text{train}} \cap M_{\text{test}} = \emptyset$ AND $P_{\text{train}} \cap P_{\text{test}} = \emptyset$). (Sec 3.2.7, Eq 18) | Replaced fallback with iterative search and explicit `ValueError` if dataset cannot satisfy bipartite partition without overlap. | 🟢 **Resolved (2026-10-09):** Absolute zero-leakage guarantee enforced. |
| **21** | **Cryptographic Salted SHA-256 Pseudonymization** | Philippine Data Privacy Act (RA 10173) requires one-way hashing of SteamIDs before storage. (Sec 3.3.1) | Implemented `pseudonymize_steamid(steamid, salt)` using salted SHA-256 mapped to positive 60-bit integers in `batch_processor.py`. | 🟢 **Resolved (2026-10-09):** RA 10173 compliance enforced at Parquet generation. |
| **22** | **Validation Threshold Calibration & Session Evaluation** | Operational decision threshold must be calibrated on validation set, and false alarms aggregated at session unit level. | `evaluate.py` calibrates $\tau^*$ on validation data, evaluates test at $\tau^*$ with Rule of Three 95% upper bounds, and computes Match-Player session metrics. | 🟢 **Resolved (2026-10-09):** Statistically sound thresholding and cluster-aware evaluation. |


---

## 🔄 7. Protocol for Starting a New Conversation

When starting work in a new conversation:
1. **Orient Immediately:** Read `AGENTS.md` and check [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md) for current phase and completed milestones.
2. **Preserve Documentation Integrity:** Whenever modifying or adding any code, update `THESIS_TRACKER.md` immediately. Never leave code undocumented.
3. **Verify Before and After:** Run `pytest tests/` before making changes to confirm baseline functionality, and run `pytest tests/` after changes to ensure zero regressions.
4. **Adhere to Mathematical Notation:** Use the symbols and equations defined in Chapter 3 ($\omega_t, \alpha_t, j_t, \kappa_t, S_c, \text{TBP}, \mathcal{L}_{\text{total}}$).
