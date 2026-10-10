# Non-Invasive Server-Side Aimbot & Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers

**Degree:** Bachelor of Science in Computer Science (Major in Data Science)  
**Academic Year:** 2026  
**Author:** Dishann Gutierrez  

---

## 📌 Project Overview
This repository contains the official implementation for the thesis research: **"Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers"**.

The system processes 64 Hz tick-sampled view angles (pitch, yaw), player positions and combat events from Counter-Strike 2 (CS2) match replays (`.dem`) using `demoparser2`. It extracts nine **candidate** channels (`MODEL_FEATURE_COLUMNS` in `src/features/kinematics.py`); their value is tested by ablation, not assumed:
1. **Pitch** (absolute yaw is excluded because it encodes map orientation).
2. **Great-circle angular velocity ($\omega_t$), acceleration ($\alpha_t$) and speed-derived jerk ($j_t$)**: smoothness proxies inspired by the Flash & Hogan minimum-jerk model.
3. **Intrinsic geodesic curvature ($\kappa_g$) and its sliding-window Shannon entropy ($S_c$).**
4. **8–12 Hz tremor band power** on signed angular rates. On the first real demo it sits below white noise, see `probe_replay_signal.py`.
5. **Target-relative aim error** (crosshair → nearest living enemy head) **and its rate**.

> Results in `reports/` are currently from synthetic pipeline-verification data and are not evidence. See `THESIS_TRACKER.md`.

The extracted telemetry sequences are processed by a **Spatial-Temporal Trajectory Transformer (ST-Trans)** featuring a dual-head output (Aimbot Binary Classification + Smurf Contrastive Embedding).

---

## 📁 Repository Structure
```
cs2-trajectory-transformer/
├── requirements.txt         <- Package dependencies (PyTorch, demoparser2, scipy, etc.)
├── README.md                <- Project overview and setup instructions
├── THESIS_TRACKER.md        <- Master thesis progress and milestone checklist
├── setup_env.ps1            <- 1-Click Automated Setup for Windows (PowerShell)
├── setup_env.bat            <- 1-Click Automated Setup for Windows (CMD)
├── setup_env.sh             <- 1-Click Automated Setup for Linux / macOS
├── demo_sample.py           <- End-to-end verification pipeline
├── tests/
│   └── test_kinematics.py   <- Unit tests for biomechanical feature extraction
├── data/
│   ├── raw_demos/           <- Directory for downloaded .dem match replays
│   └── processed_csv/       <- Parsed trajectory parquet / CSV arrays
└── src/
    ├── features/
    │   └── kinematics.py    <- Biomechanical feature engine (Wrapping, Curvature, Jerk, Tremor)
    └── models/
        └── st_transformer.py<- PyTorch Spatial-Temporal Trajectory Transformer
```

---

## 🚀 1-Click Environment Setup (Any New Device)

### Option A: Windows (PowerShell) — Recommended
Clone the repository and run:
```powershell
.\setup_env.ps1
```

### Option B: Windows (Command Prompt)
Double-click or run:
```cmd
setup_env.bat
```

### Option C: Linux / macOS
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## 🧪 Running Unit Tests & Pipeline Verification

To verify that all kinematic calculations, wrapping boundaries, and model inferences pass:
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run pytest unit test suite
pytest tests/

# Synthetic pipeline check (writes syn_* files to data/synthetic_parquet only)
python generate_benchmark_dataset.py

# Probe what real replays can physically show (quantization, tremor band vs white noise)
python probe_replay_signal.py data/raw_demos/clean/*.dem
```

---

## 📥 Data Ingestion & Cheater Replay Harvesting

To populate clean and cheater datasets for training:

```bash
# 1. Multi-Tier Clean Matches (Beginner to Pro):
python crawl_replays.py --auto --tier all --count 12

# 2. Confirmed Banned Cheaters (Verified via official FACEIT API /bans):
python crawl_replays.py --banned_file data/banned_cheaters.txt --matches_per_player 1

# 3. Automated Match Spider (Auto-scan lobbies for active cheaters):
python crawl_replays.py --scan_cheaters --count 5

# 4. Train ST-Trans Dual-Head Model:
python train.py --data_dir data/processed_parquet --epochs 50 --batch_size 32 --samples_per_player 4
```

