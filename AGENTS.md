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

### 1.2 The Solution & Candidate Biomechanical Features
These are **hypotheses tested by ablation**, not invariants. Write them as hypotheses in code, docs and the manuscript (see the `cs2-trajectory-transformer` skill, Section 3):
- **Flash & Hogan Minimum Jerk (smoothness proxy):** Voluntary reaching tends to minimize $\int \|\dddot{x}\|^2 dt$ (bell-shaped speed). We use the speed-derived scalar $j_t = d^2\omega_t/dt^2$ as a proxy. A humanized aimbot can emit minimum-jerk profiles (our own synthetic generator does), so jerk alone cannot separate them.
- **8–12 Hz Tremor Band Power (weak on real data):** Physiological tremor is real, but on the first real FACEIT demo small view-angle steps are single mouse counts (0.022° × sens), >50% of live ticks show no motion, and real-player TBP (median 0.023) is *below* white noise (0.16). The old "humans 15–45%, aimbots < 2%" claim is unsupported. Compute TBP on signed rates, never on $|\omega_t|$; verify with `probe_replay_signal.py`.
- **Spherical Geodesic Curvature ($\kappa_g$):** Computed intrinsically on $S^2$. Great circles have $\kappa_g = 0$, but a horizontal mouse swipe traces a circle of latitude with $\kappa_g = |\tan p|$, so "humans follow geodesics" is false; treat as an empirical channel.
- **Shortest-Path Euler Angle Wrapping:** View angles across coordinate boundaries ($\pm 180^\circ$) must be wrapped to $[-\pi, \pi]$ to prevent false $358^\circ$ jump spikes.
- **Target-Relative Aim Error (Fitts-style acquisition):** `aim_error` = angle from crosshair to the nearest living enemy head; `aim_error_rate` its derivative. Aim assistance acts on this relationship, which self-kinematics cannot see. Opponents are resolved **per tick** (teams swap at halftime).
- **Spatial-Temporal Trajectory Transformer (ST-Trans):** A dual-head deep transformer architecture combining:
  - **Head A (Aimbot Classification):** Binary Focal Loss ($\alpha=0.25, \gamma=2.0$). Labels are **account-level weak labels** ("account later banned for cheating"), not per-window ground truth; report player-match session metrics as primary.
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
├── generate_benchmark_dataset.py           <- Synthetic PIPELINE-VERIFICATION generator (writes syn_* to data/synthetic_parquet; not evidence)
├── inspect_checkpoint.py                   <- Weight inspection utility for saved PyTorch checkpoints
├── probe_replay_signal.py                  <- Empirical probe: mouse-count quantization & tremor-band signal vs white noise
├── ablation.py                             <- Feature ablation (central RQ: engineered vs raw-angle inputs)
├── ingest_cs2cd.py                         <- CS2CD download (verified, resumable) + ATW extraction to data/processed_parquet_cs2cd
├── pilot_study.py                          <- CS2CD session-level grouped-CV pilot (feature groups + mouse-input consistency)
├── visualize.py                            <- Publication-ready ROC/PR curves & t-SNE latent cluster visualizer
│
├── src/                                    <- Core Python package source modules
│   ├── features/
│   │   └── kinematics.py                   <- Feature engine + MODEL_FEATURE_COLUMNS (single source of truth, 9 channels)
│   ├── models/
│   │   ├── st_transformer.py               <- PyTorch Spatial-Temporal Trajectory Transformer (ST-Trans)
│   │   ├── losses.py                       <- Supervised InfoNCE Loss & Binary Focal Loss
│   │   └── baselines.py                    <- BiLSTM & Classical baseline models (RF, GradientBoosting, MLP)
│   └── data/
│       ├── demo_parser.py                  <- High-speed CS2 replay parser wrapping demoparser2
│       ├── atw_filter.py                   <- Active Tracking Window (ATW) spatial-temporal extractor
│       ├── dataset.py                      <- Zero-leakage partitioning (components / match_drop), P×K sampler, collator
│       ├── demo_downloader.py              <- FACEIT API polite scraper & archive decompressor
│       ├── batch_processor.py              <- extract_player_feature_windows (shared by ingestion & analyze_match) + Parquet export
│       └── cs2cd_adapter.py                <- CS2CD Parquet+JSON loader, label policy, verified downloader
│
├── tests/                                  <- Pytest suite (77/77 passing on 2026-10-11)
│   ├── test_kinematics.py                  <- Euler wrapping, curvature, & tremor PSD unit tests
│   ├── test_model.py                       <- ST-Trans forward pass, masks, & loss function tests
│   ├── test_dataset.py                     <- Zero data leakage splits & batch collation tests
│   ├── test_parser.py                      <- ATW geometry & demoparser2 integration tests
│   ├── test_downloader.py                  <- Download, decompression, & rate limit backoff tests
│   ├── test_baselines.py                   <- Baseline model architectures & inference tests
│   ├── test_feature_contract_and_splits.py <- Feature contract, TBP, aim error, halftime swap, P×K, match_drop, ELO mask
│   └── test_cs2cd_adapter.py               <- CS2CD loading, label policy, namespacing, file validation, resumability
│
├── data/                                   <- Local and external data directories
│   ├── raw_demos/                          <- Downloaded .dem replays (clean/ & cheaters/)
│   ├── processed_parquet/                  <- Real ATW Parquet files (9 model channels + yaw + metadata)
│   └── synthetic_parquet/                  <- Synthetic syn_* windows (never mix into processed_parquet)
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
   - Before taking `arccos(dot_product)`, strictly clamp the dot product to `[-1, 1]`. For small angles (aim error) prefer `arctan2(||a x b||, a . b)`, which stays accurate near 0 where `arccos` loses ~1e-6 rad.
   - In probability functions, clamp inputs away from $0$ and $1$ to prevent `log(0)` / `NaN`: `inputs.clamp(min=1e-6, max=1.0 - 1e-6)`.
3. **Euler Coordinate Wrapping:** Any numerical differentiation on camera yaw or pitch must pass through `wrap_angle_rad(d_angle)` using `((d_angle + np.pi) % (2.0 * np.pi)) - np.pi`.

### 4.2 PyTorch Deep Learning Best Practices
1. **Mask-Aware Pooling:** Sequence lengths in ATWs vary ($64 \le L \le 512$). All temporal pooling over transformer representations MUST multiply by the attention mask and normalize by valid sequence length:
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
4. **In-Place File Updates (Zero Backup Copies):** Never generate `.bak`, `.backup`, or duplicate copy files on the host system or desktop. Directly update canonical target files in-place and rely on Git version control.

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
| **11** | **XGBoost & Tabular MLP Baseline** | Benchmark specified XGBoost and $48 \to 128 \to 64 \to 1$ MLP. | Native `xgboost` integrated into `ClassicalBaselines` and PyTorch `TabularMLP` implemented. | 🟢 **Resolved (2026-10-08):** Exact architecture alignment with Table 7 (DOCX internal table index 28). |
| **12** | **Smurf Detection Decision Logic** | Continuous ELO discrepancy decision rule $\Delta_{\text{ELO}} \ge \tau$ and latent biometric cluster matching. | Implemented `SmurfDetector` class in `src/models/st_transformer.py` classifying smurf severity tiers. | 🟢 **Resolved (2026-10-08):** Operational decision rule and centroid matching fully implemented. |
| **13** | **Inference Latency Profiling** | Server-side throughput validation ($< 500$ ms per match, $< 5$ ms per ATW window). | `evaluate.py` profiles batch=1 streaming latency (~7.3ms CPU) and match audit throughput (~526ms CPU / <50ms GPU). | 🟢 **Resolved (2026-10-08):** Formal profiling benchmark active in `evaluate.py`. |
| **14** | **ATW Duration Calibration at 64 Hz** | Table 4 duration conversions held legacy 128 Hz times (0.50s, 0.25s, 4.0s). | Table 4 (Table 7) & text updated: Buffer $\pm 64$ ticks = 1.00s, $L_{\min}=32$ ticks = 0.50s, $L_{\max}=512$ ticks = 8.00s. | 🟢 **Resolved (2026-10-08):** All time constants consistent across manuscript. |
| **15** | **Intrinsic Geodesic Curvature on $S^2$** | Great-circle flicks should possess $\kappa_g = 0.0$; ambient 3D curvature formula yielded $\kappa=1.0$. (Sec 3.2.4, Eq 9) | `kinematics.py` implements $\kappa_g(t) = \frac{|\mathbf{v} \cdot (\mathbf{v}' \times \mathbf{v}'')|}{\|\mathbf{v}'\|^3 + \epsilon}$, yielding $\kappa_g = 0$ for great circles. | 🟢 **Resolved (2026-10-09):** Intrinsic geodesic curvature aligned with differential geometry and Eq (9). |
| **16** | **Batch Processor Manifest Resolution** | Ingestion must discover downloader staging manifests and link real SteamIDs and player ELO ratings. | `batch_processor.py` dynamically resolves `staging/staging_manifest.json`, `banned_steamids.json`, and `player_elos.json`. | 🟢 **Resolved (2026-10-09):** Real match labels and ELO profiles automatically ingested. |
| **17** | **Biometric Identification Evaluation** | Supervised InfoNCE metric learning must evaluate held-out player retrieval and biometric clustering. | `evaluate.py` computes Top-1 ($P@1$) & Top-5 ($P@5$) retrieval; `benchmark.py` generates player-colored t-SNE plot. | 🟢 **Resolved (2026-10-09):** Publication-grade retrieval metrics and latent visualizations active. |
| **18** | **Ethics & RA 10173 Pseudonymization** | Raw SteamIDs identify human accounts; Philippine Data Privacy Act requires pseudonymization. (Sec 3.3.1) | Proposal specifies salted SHA-256 one-way hashing ($h = \text{SHA256}(\text{SteamID} \,\|\, \text{salt})$) severing link to PII. | 🟢 **Resolved (2026-10-09):** RA 10173 compliance and research ethics explicitly codified. |
| **19** | **Inference Preprocessing Pipeline Mismatch & Unified Normalization** | Training and inference previously risked scaling mismatch (dataset standardization vs domain scaling). | Standardized canonical domain scaling across training and inference; added `save_scaler_stats` and `load_scaler_stats` (`_scaler.npz`) for deterministic reuse when global standardization is enabled. | 🟢 **Resolved (2026-10-09):** Unified deterministic normalization and scaler reuse. |
| **20** | **Strict Joint Zero-Leakage Guarantee** | Joint disjoint constraint ($M_{\text{train}} \cap M_{\text{test}} = \emptyset$ AND $P_{\text{train}} \cap P_{\text{test}} = \emptyset$). (Sec 3.2.7, Eq 18) | Replaced fallback with iterative search and explicit `ValueError` if dataset cannot satisfy bipartite partition without overlap. | 🟢 **Resolved (2026-10-09):** Absolute zero-leakage guarantee enforced. |
| **21** | **Cryptographic Salted SHA-256 Pseudonymization & Managed Secret Salt** | Raw SteamIDs and match IDs in metadata/filenames; privacy claim overstated irreversible anonymization. (Sec 3.3.1) | Implemented `pseudonymize_steamid` and `pseudonymize_match_id` using salted SHA-256 with managed secret salt (`CS2_PSEUDONYMIZATION_SALT`); updated Sec 3.3.1 to state pseudonymous governance under RA 10173. | 🟢 **Resolved (2026-10-09):** Pseudonymous data governance and managed salt enforced. |
| **22** | **Validation Threshold Calibration & Session Evaluation** | Operational decision threshold must be calibrated on validation set at 0.01% target, and false alarms aggregated at Match-Player session unit level. | `evaluate.py` calibrates $\tau^*$ on validation data with `target_fpr=0.0001` (0.01%), evaluates test at $\tau^*$ with Rule of Three 95% upper bounds, and computes Match-Player session metrics grouping by `(match_id, player_id)`. | 🟢 **Resolved (2026-10-09):** Statistically sound thresholding and match-player session evaluation. |
| **23** | **Replay Telemetry Phrasing & Tremor Hypothesis Nuance** | Manuscript held legacy sub-tick replay wording and overstated universal continuous hand tremor. | Purged all "sub-tick" replay claims across Chapters 1–3 (standardizing on 64 Hz tick-sampled telemetry and tick-indexed combat events); refined tremor hypothesis in Sec 3.1.2/3.2.4 acknowledging biological variability (PMC3517187). | 🟢 **Resolved (2026-10-09):** Manuscript claims and physiological literature perfectly aligned. |
| **24** | **Enforcement of Managed Secret Salt** | RA 10173 data privacy governance requires a managed secret salt for irreversible one-way SHA-256 hashing. | `batch_processor.py` strictly enforces `CS2_PSEUDONYMIZATION_SALT` (raising `ValueError` if absent without public string fallback); both root and inner `.env.example` document the variable. | 🟢 **Resolved (2026-10-09):** Secret salt strictly enforced with unit tests. |
| **25** | **Window-Level vs Session-Level FPR Bounds & Clean-Sample Denominators** | Denominator in statistical bounds strictly requires clean negative samples ($N_{\text{clean}}$). Intra-cluster correlation and overlapping chunks require dependence-aware bounds. | Updated Abstract, Sec 3.2.9, RQ4, and Table 8 (DOCX index 29) to formalize clean-sample requirements ($N_{\text{clean\_ATW}} \ge 30,000$ from ~210–220 test matches, $N_{\text{clean\_sessions}} \approx 1,950 \implies \le 0.16\%$, and $M_{\text{clean\_matches}} = 180 \implies \le 1.67\%$), cleanly separating nominal Rule-of-Three bounds from cluster bootstrap bounds over independent match clusters in `evaluate.py`. | 🟢 **Resolved (2026-10-09):** Three-tier statistical bounds, clean denominators, and Table 8 labeling aligned. |
| **26** | **Speed-Derived Scalar Angular Jerk Feature vs Theory Language** | Second derivative of great-circle angular speed $\omega_t$ is a speed-derived scalar proxy, not unobservable 3D limb vector jerk. | Clarified `src/features/kinematics.py` docstrings, Table 5 (DOCX index 16), and Sec 3.2.4 as "Speed-Derived Scalar Angular Jerk ($j_t = d^2\omega_t/dt^2$)". | 🟢 **Resolved (2026-10-09):** Kinematic proxy and biomechanical theory aligned. |
| **27** | **Fail-Fast Checkpoint Loading** | Missing checkpoints previously silently fell back to uninitialized random weights in analysis/eval scripts. | `analyze_match.py`, `evaluate.py`, and `demo_sample.py` now raise `FileNotFoundError` unless explicit `--allow_untrained` flag is passed. | 🟢 **Resolved (2026-10-09):** Fail-fast checkpoint loading active. |
| **28** | **Graph Partition Target Ratios & TBP Phase Coherence Framing** | 80/10/10 previously framed as rigid 1,600/200/200 match counts; TBP previously claimed direct measurement of phase coherence. | Manuscript reframed 80/10/10 as nominal target ratios ($\approx 80\% / 10\% / 10\%$) determined by connected-component boundaries in bipartite graph partitioning; TBP clarified as relative spectral power ($|X(f)|^2$), framing phase-incoherent noise as a theoretical hypothesis. | 🟢 **Resolved (2026-10-09):** Graph partitioning and TBP spectral vs phase nuances fully aligned. |
| **29** | **Design-Effect, Player Dependence & Empirical Stopping Condition** | Manuscript claimed Design Effect and Match-level FPR not present in evaluator; sample plan asserted match counts unconditionally guaranteed 30k clean ATWs; zero-FP design effect fell back to independence. | Implemented `compute_design_effect` with continuous score ICC when binary FP=0, match & player clustering axes with conservative combined $N_{\text{eff}}$, and match-level evaluation ($3/M_{\text{clean}}$) in `evaluate.py`; reconciled planned corpus to 2,000–2,200 matches with $N_{\text{clean\_ATW}} \ge 30,000$ formalized as an empirical stopping condition dynamically audited during ingestion. | 🟢 **Resolved (2026-10-09):** Score-derived ICC, match/player clustering axes, and empirical stopping condition aligned. |
| **30** | **Empirical Stopping Condition on Test Split & Stratified Partitioning:** Overall corpus clean count previously used for stopping condition instead of held-out test split ($N_{\text{clean\_test\_ATW}} \ge 30,000$); cluster split lacked class stratification; score-derived ICC needed explicit sensitivity proxy framing. | Refactored `partition_dataset_files` in `dataset.py` with class-stratified bipartite graph partitioning; updated `audit_clean_atw_quota` in `batch_processor.py` to evaluate the partitioned held-out test split directly; updated `crawl_replays.py` (`--audit_clean_quota` & `--crawl_until_quota`) to stop only when test split clean quota is satisfied; clarified score-derived ANOVA ICC in docstrings and proposal Sec 3.2.9 strictly as an empirical sensitivity proxy rather than an exact binary confidence bound. | 🟢 **Resolved (2026-10-09):** Test-split quota auditing, class-stratified disjoint partitioning, and sensitivity proxy framing fully aligned. |
| **31** | **Fail-Closed Test Split Quota Audit & Minority Class Cluster Stratification** | If data could not be partitioned, audit caught error and fell back to overall corpus count, falsely reporting `quota_met=True` and allowing crawl loop to terminate without auditing test split; splitter previously only stratified when both classes had $\ge 3$ clusters, otherwise using unstratified split that dropped minority cheaters from test set. | `audit_clean_atw_quota` in `batch_processor.py` now strictly fails closed (`quota_met=False`, `clean_count=0`, records `partition_error`) when `partition_test_split=True` and partitioning fails; `crawl_replays.py` terminates with `sys.exit(1)` and explicit error diagnostics if test split is unavailable or quota unmet; `partition_dataset_files` in `dataset.py` always stratifies clean and cheater clusters independently whenever both classes exist (guaranteeing both classes in train and test when $\ge 2$ clusters per class exist); updated proposal manuscript (Sec 3.2.2, 3.2.7, 3.2.9). | 🟢 **Resolved (2026-10-09):** Fail-closed test split stopping condition and robust minority-class cluster stratification enforced with unit tests (36/36 passing). |
| **32** | **Empty-Validation Edge Case & Component Feasibility Enforcement** | When exactly 2 connected components existed in each class ($C=2, X=2$), partitioning assigned 1 to train and 1 to test for each class, silently leaving validation empty (`val_files=[]`) and breaking training/threshold calibration; legacy match-fallback previously broke connected components by filtering players across matches. | Refactored `partition_dataset_files` in `dataset.py` to preserve connected component integrity; eliminated component-breaking fallbacks; strictly guarantees non-empty partitions with both classes in Train and held-out Test when $C \ge 2, X \ge 2, C+X \ge 5$; raises descriptive `ValueError` listing clean and cheater counts when component counts make valid non-empty partitions infeasible (e.g. $C=2, X=2$ or single-class $N < 3$); reports actual split sizes and cluster allocations (approximating 80/10/10); added unit tests (42/42 passing). | 🟢 **Resolved (2026-10-09):** Empty-validation edge case fixed, component feasibility enforced, zero-leakage strictly preserved. |
| **33** | **Validation Operating Threshold Calibration Data Requirements & Split Boundary Behavior** | Proposal Sec 3.2.7/3.2.9 and tracker previously overstated when both classes are guaranteed in every split; `calibrate_operating_threshold` returned 0.5 on single-class validation or used percentiles vulnerable to ties where `score >= tau` exceeded target FPR; split-ratio was claimed by ATW weight rather than component count; and 30,000-window quota conflated sample size with empirical confidence bounds. | Refactored `calibrate_operating_threshold` in `evaluate.py` to evaluate candidates under the `score >= tau` rule accounting for ties; calculates actual validation FPR and marks calibration successful only if empirical rate meets target (marking unmet if scores equal 1.0 or ties exceed allowed FPs); keeps empirical calibration separate from statistical evidence by reporting observed FP count, clean sample count, and 95% upper confidence bounds (using the Rule-of-Three approximation for zero false positives and Clopper-Pearson bounds for nonzero counts); keeps validation TPR unavailable on clean-only splits; updated proposal Sec 3.2.7 (P315 conditional calibration, P316 component-count allocation) and Sec 3.2.9 (P331 distinguishing 30,000 clean test stopping condition from observed zero-FP result); added unit tests for tied scores, threshold equal to max score, scores at 1.0, and unachievable targets (50/50 tests passing). | 🟢 **Resolved (2026-10-09):** Tie-aware FPR calibration enforced, statistical evidence separated, component-count allocation aligned, and 30,000-window bound clarified. |
| **34** | **Tremor on Signed Rates & $L_{\min}$** | Eq (11) TBP over 8–12 Hz / 1–30 Hz with 1 Hz bins (64-tick FFT). | TBP on signed rates (not rectified $|\omega|$, which doubles frequency); vectorized; $L_{\min}$ = 64 ticks so every ATW holds one full FFT window. | 🟢 **Resolved (2026-10-10 code, 2026-10-11 manuscript Table 4 / Sec 3.2.3–3.2.4).** |
| **35** | **Tremor Premise vs Real Data** | Sec 2.3.3/3.2.4 assert human 8–12 Hz resonance. | `probe_replay_signal.py`: real TBP median 0.023 < white noise 0.16; count-quantized angles. | 🟡 **Manuscript reframed (2026-10-11);** report probe results on the full corpus in Chapter 4. |
| **36** | **Target-Relative Channels; No Absolute Yaw** | Table 5 lists 8 self-kinematic channels incl. absolute yaw; Fitts' Law in theory but not in features. | `MODEL_FEATURE_COLUMNS` = 9: pitch, ω, α, j, κ_g, S_c, TBP, aim_error, aim_error_rate. Yaw kept in Parquet for the raw baseline only. | 🟢 **Resolved (2026-10-11):** Tables 5, 6, 7, IPO table and Sec 3.2.4/3.2.5/3.2.8 updated (no new equation number added; aim error is defined in Sec 3.2.4 text). |
| **37** | **Halftime Side-Swap Enemy Bug** | — | Opponents resolved per tick; zero-distance targets excluded. Real demo ATW rows 1.52M → 0.83M. | 🟢 **Resolved (2026-10-10).** Older real-data audits are invalid. |
| **38** | **Shared Preprocessing (no train/inference skew)** | Same pipeline for training and server-side audit. | `extract_player_feature_windows` used by `batch_processor.py` and `analyze_match.py`. | 🟢 **Resolved (2026-10-10).** |
| **39** | **InfoNCE Positive Pairs** | Eq (17) requires same-player positives per batch. | `PlayerBalancedBatchSampler` (P×K), `--samples_per_player 4`. | 🟢 **Resolved (2026-10-10).** |
| **40** | **Giant-Component Partitioning** | Sec 3.2.7 assumes hundreds of independent components. | Size-aware allocation (largest component → train); `strategy="auto"` falls back to leakage-free `match_drop`. | 🟢 **Resolved (2026-10-11):** Sec 3.2.7 describes size-aware allocation and the match-level fallback. |
| **41** | **Weak Labels, Masked ELO, Ban Regex** | Labels as ground truth; ELO from API. | Account-level weak labels documented; unknown ELO masked (no 1500 default); `is_cheating_ban_reason` word-boundary regex. | 🟢 **Resolved (2026-10-11):** Sec 1.4 and 3.2.2 state account-level weak labels and within-match negatives; Sec 3.2.6/3.2.9 state ELO masking. Open: confirm whether FACEIT ELO is at match time. |
| **42** | **Synthetic Data Integrity** | Synthetic suite as stress test only. | Generator isolated to `data/synthetic_parquet`, adds quantization, moving targets and a humanized aimbot; reports tagged with provenance. Synthetic scores are never evidence. | 🟢 **Resolved (2026-10-10).** Old `reports/*.csv` are synthetic. |
| **43** | **Promised but Unimplemented** | CS2CD integration, AntiCheatPT baseline, ONNX export, HID/Bézier synthetic suite. | CS2CD adapter implemented (`src/data/cs2cd_adapter.py`, `ingest_cs2cd.py`); ONNX described as planned in Sec 3.2.9. | 🔴 **Open:** AntiCheatPT reproduction, ONNX export, HID/Bézier synthetic suite, full CS2CD ingestion. |

---

## 🔄 7. Protocol for Starting a New Conversation

When starting work in a new conversation:
1. **Orient Immediately:** Read `AGENTS.md` and check [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md) for current phase and completed milestones. Then load the project skill `.agents/skills/cs2-trajectory-transformer/SKILL.md` (Claude Code: `/cs2-trajectory-transformer`), which holds the coding logic rules and decision checklists.
2. **Preserve Documentation Integrity:** Whenever modifying or adding any code, update `THESIS_TRACKER.md` immediately. Never leave code undocumented.
3. **Verify Before and After:** Run `pytest tests/` before making changes to confirm baseline functionality, and run `pytest tests/` after changes to ensure zero regressions.
4. **Adhere to Mathematical Notation:** Use the symbols and equations defined in Chapter 3 ($\omega_t, \alpha_t, j_t, \kappa_t, S_c, \text{TBP}, \mathcal{L}_{\text{total}}$).
5. **Zero Backup Copies Policy:** Never create `.bak`, `.backup`, or duplicate copy files of the manuscript or codebase files. Directly update the active canonical file in-place and rely on Git version control.
