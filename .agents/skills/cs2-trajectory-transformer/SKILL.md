---
name: cs2-trajectory-transformer
description: Master guidance, operational runbooks, coding conventions, testing protocols, and thesis context for the CS2 Trajectory Transformer project. Use whenever developing, modifying, training, evaluating, or writing thesis documentation.
---

# 🎮 CS2 Trajectory Transformer — Developer & Agent Skill Guide

This skill equips any AI agent with complete context, operational runbooks, coding techniques, and commenting standards for the thesis research project: **"Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers"** (Medel & Gutierrez, 2026, USeP College of Information and Computing).

---

## 🧭 1. Rapid Context Restoration (New Conversation Checklist)

Whenever you enter a new conversation, execute this 30-second orientation sequence:
1. **Identify Project Structure:**
   - Source code resides in `src/` (`features/`, `models/`, `data/`).
   - Command-line entry points are in the project root (`train.py`, `evaluate.py`, `benchmark.py`, `crawl_replays.py`, `demo_sample.py`).
   - Tracked thesis progress is in [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md).
   - Deployment and post-completion runbooks are in [`POST_COMPLETION_GUIDE.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/POST_COMPLETION_GUIDE.md).
2. **Identify the Environment:**
   - Dedicated Python environment: `cs2-trajectory-transformer\venv\Scripts\python.exe` (or `D:\cs2_thesis_env`).
   - GPU: NVIDIA RTX with CUDA support.
3. **Verify Baseline State:**
   - Run tests: `& "cs2-trajectory-transformer\venv\Scripts\python.exe" -m pytest "cs2-trajectory-transformer\tests"` (All 22 tests must pass).
4. **Inspect Master Roadmap:**
   - Check which Phase in [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md) is currently active before proposing work.

---

## 🔬 2. Theoretical Foundations & Physics Invariants

Always ground code and analysis in the core biomechanical and mathematical principles from Chapter 3:

| Principle | Mathematical Formulation | Physical & Biomechanical Significance |
| :--- | :--- | :--- |
| **Euler Wrapping** | $\text{wrap}(\Delta\theta) = ((\Delta\theta + \pi) \pmod{2\pi}) - \pi$ | Prevents catastrophic $358^\circ$ velocity spikes when camera yaw crosses the $\pm 180^\circ$ coordinate boundary. |
| **Spherical Angular Velocity** | $\omega_t = \frac{1}{\Delta t}\sqrt{(\Delta p_t)^2 + (\cos(p_t)\Delta y_t)^2}$ | Computes geodesic great-circle velocity on the unit sphere $S^2$, eliminating planar Euclidean distortion near poles. |
| **Minimum Jerk Dynamics** | $j_t = \frac{\alpha_t - \alpha_{t-1}}{\Delta t}$ | Flash & Hogan model: Human neuromuscular motor control minimizes jerk $\int j(t)^2 dt$. Aimbots introduce step accelerations and jerk spikes. |
| **Spherical Curvature** | $\kappa_t = \frac{\|\mathbf{v}'(t) \times \mathbf{v}''(t)\|}{\|\mathbf{v}'(t)\|^3 + \epsilon}$ | Differential geometric curvature of 3D sight vector on $S^2$. Detects robotic linear or spline-interpolated aim smoothing. |
| **Curvature Entropy** | $S_c = -\sum_{k=1}^M p_k \log_2(p_k + 10^{-9})$ | Sliding-window Shannon entropy over 32 ticks (250ms). Low entropy = scripted determinism; high entropy = human neuromuscular variance. |
| **8–12 Hz Tremor PSD** | $\text{TBP} = \frac{\sum_{f=8}^{12} \|X(f)\|^2}{\sum_{f=1}^{30} \|X(f)\|^2 + \epsilon}$ | Fast Fourier Transform of velocity across 64 ticks (0.5s). Human aiming exhibits natural $8\text{--}12\text{ Hz}$ muscle tremor ($15\%\text{--}45\%$). |
| **Active Tracking Window** | $\theta_{\text{FOV}} \le 30^\circ, \|\mathbf{r}\| \le 3500\text{ units}, \pm 64\text{ ticks}$ | Dual spatial-temporal filter pruning $> 70\%$ passive walking noise and isolating decisive combat interactions. |

---

## ⚡ 3. End-to-End Operational Runbooks

### Runbook A: Verify Full Pipeline via Synthetic Benchmark
Use this when real replays are not available or for immediate functional verification:
```powershell
# 1. Generate 600 synthetic trajectories across clean skill tiers and aimbots
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\generate_benchmark_dataset.py"

# 2. Run unit test suite
& "cs2-trajectory-transformer\venv\Scripts\python.exe" -m pytest "cs2-trajectory-transformer\tests"

# 3. Execute end-to-end demo verification
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\demo_sample.py"
```

### Runbook B: Replay Ingestion & Multi-Tier Crawling
Use this to collect real match replays from FACEIT Open API and Backblaze CDN:
```powershell
# Crawl clean multi-tier matches (beginner, intermediate, advanced, pro)
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\crawl_replays.py" --auto --tier all --count 12

# Crawl confirmed banned cheaters with verified API ban records
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\crawl_replays.py" --banned_file "cs2-trajectory-transformer\data\banned_cheaters.txt" --matches_per_player 1

# Automated lobby spider to find newly banned cheater lobbies
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\crawl_replays.py" --scan_cheaters --count 5
```

### Runbook C: Batch Processing into ATW Parquet Stores
```powershell
& "cs2-trajectory-transformer\venv\Scripts\python.exe" -c "
import sys; sys.path.append('cs2-trajectory-transformer/src')
from data.batch_processor import batch_process_demos
batch_process_demos('cs2-trajectory-transformer/data/raw_demos/clean', 'cs2-trajectory-transformer/data/processed_parquet', is_cheater_dataset=False)
"
```

### Runbook D: ST-Trans Model Training
```powershell
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\train.py" `
  --data_dir "cs2-trajectory-transformer\data\processed_parquet" `
  --epochs 50 `
  --batch_size 32 `
  --lr 1e-4 `
  --d_model 128 `
  --nhead 8 `
  --num_layers 4 `
  --save_path "cs2-trajectory-transformer\models\checkpoints\best_model.pt"
```

### Runbook E: Model Evaluation & Benchmarking
```powershell
# Evaluate trained model checkpoint on test split
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\evaluate.py" `
  --data_dir "cs2-trajectory-transformer\data\processed_parquet" `
  --model_path "cs2-trajectory-transformer\models\checkpoints\best_model.pt"

# Run full baseline benchmark comparison (RF, GradientBoosting, MLP, BiLSTM, ST-Trans)
& "cs2-trajectory-transformer\venv\Scripts\python.exe" "cs2-trajectory-transformer\benchmark.py"
```

---

## 📐 4. Proper Coding Techniques & Code Review Standards

When writing, reviewing, or refactoring code in this project, enforce these rules:

### Rule 1: Vectorization & Performance
- **Prohibited:** Loop-based iteration over ticks: `for row in df.itertuples(): ...`.
- **Mandatory:** Vectorized NumPy operations:
  ```python
  d_yaw = wrap_angle_rad(np.diff(yaw_rad, prepend=yaw_rad[0]))
  d_pitch = wrap_angle_rad(np.diff(pitch_rad, prepend=pitch_rad[0]))
  arc_diff = np.sqrt(d_pitch**2 + (np.cos(pitch_rad) * d_yaw)**2)
  angular_velocity = arc_diff / dt
  ```

### Rule 2: Numerical Safeguards & Clamping
- Never perform unshielded divisions or square roots near zero:
  ```python
  curvature = cross_norm / (speed**3 + 1e-6)
  ```
- Before computing inverse trigonometric functions (`arccos`), clamp inputs:
  ```python
  dot_clamped = np.clip(dot_product, -1.0, 1.0)
  angle_rad = np.arccos(dot_clamped)
  ```
- In log operations or loss functions:
  ```python
  p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)
  ```

### Rule 3: PyTorch Mask-Aware Reductions
- Because sequence lengths vary ($32 \le T \le 512$), always apply mask-weighted pooling:
  ```python
  # Mask shape: [Batch, Seq_Len] (True for valid tokens, False for padding)
  mask_expanded = attention_mask.unsqueeze(-1).float()  # [Batch, Seq_Len, 1]
  valid_lengths = mask_expanded.sum(dim=1).clamp(min=1.0) # [Batch, 1]
  pooled = (encoded_tokens * mask_expanded).sum(dim=1) / valid_lengths
  ```

### Rule 4: Data Leakage Protection
- Never split data randomly across ticks or segments.
- Enforce strict player-level and match-level disjoint sets:
  ```python
  assert len(set(train_players).intersection(set(test_players))) == 0, "Data leakage detected across splits!"
  ```

### Rule 5: Multiprocessing Safety
- Guard all multiprocessing code blocks:
  ```python
  if __name__ == "__main__":
      # Entry point logic
  ```

---

## ✍️ 5. Commenting & Documentation Guidelines

Every function, class, and mathematical derivation must adhere to this commenting protocol:

### Function Docstring Template
```python
def compute_spherical_angular_velocity(
    pitch_rad: np.ndarray, 
    yaw_rad: np.ndarray, 
    dt: float = 1.0 / 128.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes true great-circle angular velocity on the unit sphere view model.

    Thesis Reference:
        Chapter 3, Equation (5):
        omega_t = (1 / dt) * sqrt((wrap(Delta p_t))^2 + (cos(p_t) * wrap(Delta y_t))^2)

    Parameters:
    -----------
    pitch_rad : np.ndarray
        Pitch gaze angles in radians [-pi/2, pi/2], shape (N,).
    yaw_rad : np.ndarray
        Yaw gaze angles in radians [-pi, pi], shape (N,).
    dt : float, optional
        Time step between ticks in seconds (default: 1.0 / 128.0).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - angular_velocity: Great-circle speed in rad/s, shape (N,).
        - d_pitch: Wrapped pitch differential, shape (N,).
        - d_yaw: Wrapped yaw differential, shape (N,).
    """
```

### Inline Thesis Equation Annotations
When writing complex loss functions or feature transformations, include the equation tag directly above the logic:
```python
# Thesis Equation (16): Binary Focal Loss
# FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
# Down-weights easy negatives to focus gradients on subtle cheater smoothers.
p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
loss = (focal_weight * bce_loss).mean()
```

---

## 📊 6. Thesis Target Benchmarks Reference (Table 8)

When running benchmarks or evaluations, verify results against these operational thresholds:

| Metric Category | Target Value | Operational Significance |
| :--- | :---: | :--- |
| **AUROC (ROC Area)** | $> 0.980$ | Global separation across all decision thresholds. |
| **AUPRC (PR Area)** | $> 0.950$ | High precision under severe cheater class imbalance. |
| **Strict False Positive Rate** | $< 0.01\%$ | Less than 1 false accusation per 10,000 clean evaluations. |
| **ELO Skill Prediction MAE** | $< 150\text{ ELO}$ | Predicts nominal skill within half a competitive tier. |
| **Spearman Rank Correlation** | $> 0.850$ | Monotonic alignment between latent space and skill. |
| **Server Inference Latency** | $< 500\text{ ms}$ | Sub-second match report generation via ONNX runtime. |

---

## 🛡️ 7. Discrepancy Prevention & Code Integrity Rule

Whenever you propose code edits or discover inconsistencies between the codebase and the thesis manuscript:
1. Cross-check against the **Proposal Alignment & Discrepancy Registry** in `AGENTS.md`.
2. Do not introduce silent changes to feature shapes or column definitions without updating both `dataset.py` and `test_kinematics.py`.
3. Immediately update [`THESIS_TRACKER.md`](file:///C:/Users/ddgut/OneDrive/Desktop/cs2-trajectory-transformer/cs2-trajectory-transformer/THESIS_TRACKER.md) with an entry in the Change Log.
