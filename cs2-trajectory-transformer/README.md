# Non-Invasive Server-Side Aimbot & Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers

**Degree:** Bachelor of Science in Computer Science (Major in Data Science)  
**Academic Year:** 2026  
**Author:** Dishann Gutierrez  

---

## 📌 Project Overview
This repository contains the official implementation for the thesis research: **"Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers"**.

The system processes 128-tick spatial-temporal mouse view-angles $(\phi, 	heta)$ and player movement telemetry from Counter-Strike 2 (CS2) match replay files (`.dem`) using `demoparser2`. It extracts neuromuscular kinematic features:
1. **Euler-Wrapped Angular Velocity ($\omega_t$):** Geodesic great-circle angular speed on the unit sphere sight vector.
2. **Angular Acceleration ($lpha_t$) and Minimum Jerk ($j_t$):** Higher-order derivatives of motor coordination (Flash & Hogan model).
3. **Spherical Geodesic Trajectory Curvature ($\kappa_t$):** Curvature computed via 3D sight-line vector cross products.
4. **Trajectory Curvature Shannon Entropy ($S_c$):** Sliding-window spatial entropy.
5. **8–12 Hz Physiological Hand Tremor Band Power:** Relative spectral power via FFT/Welch's PSD to distinguish biological muscle tremor from algorithmic cursor updates.

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

# Run end-to-end pipeline verification
python demo_sample.py
```
