# Non-Invasive Server-Side Aimbot & Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers

**Degree:** Bachelor of Science in Computer Science (Major in Data Science)  
**Academic Year:** 2026  

---

## 📌 Project Overview
This repository contains the official codebase for the thesis project: **"Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers"**.

The project processes 128-tick spatial-temporal mouse view angles $(\phi, \theta)$ and player movement telemetry from Counter-Strike 2 (CS2) match replay files (`.dem`) to extract biological neuromuscular kinematic metrics (Angular Jerk, Trajectory Curvature Entropy) and train a **Spatial-Temporal Trajectory Transformer (ST-Trans)** in PyTorch.

---

## 📁 Repository Structure
```
cs2-trajectory-transformer/
├── requirements.txt         <- Package dependencies
├── README.md                <- Project documentation
├── demo_sample.py           <- Starter script to verify parsing & model pipeline
├── data/
│   ├── raw_demos/           <- Place downloaded .dem files here
│   └── processed_csv/       <- Extracted trajectory DataFrames
└── src/
    ├── features/
    │   └── kinematics.py    <- Angular Jerk & Curvature Entropy extraction
    └── models/
        └── st_transformer.py<- PyTorch Spatial-Temporal Trajectory Transformer
```

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment (Local PC)
Open PowerShell or Command Prompt in this folder:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Sample Pipeline Verification
```bash
python demo_sample.py
```
This script generates synthetic 128-tick trajectory data, extracts kinematic features (Angular Velocity & Jerk), feeds them into the PyTorch ST-Trans model, and outputs predictions for Aimbot Probability & Estimated Skill ELO.

---

## 🔬 Key Mathematical Metrics (`src/features/kinematics.py`)

1. **Angular Velocity ($\omega_t$):**
   $$\omega_t = \frac{\sqrt{(\theta_t - \theta_{t-1})^2 + (\phi_t - \phi_{t-1})^2}}{\Delta t}$$

2. **Angular Jerk ($j_t$):**
   $$j_t = \frac{\Delta^2 \omega_t}{\Delta t^2}$$

3. **Trajectory Curvature Entropy ($S_c$):**
   $$S_c = -\sum_{i \in W} P(\kappa_i) \log_2 P(\kappa_i)$$
