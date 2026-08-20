# 🚀 Post-Completion Guide & Deployment Runbook

**Thesis Research:** Non-Invasive Server-Side Aimbot and Smurf Detection in FPS Esports Using Micro-Kinematic Trajectory Transformers  
**Author:** Dishann Gutierrez  
**Degree:** Bachelor of Science in Computer Science (Major in Data Science)  

---

## 📌 Overview
This document outlines the complete operational roadmap, deployment steps, defense strategies, and publication procedures for when the core machine learning models and experiments are finished.

---

## 🗺️ Table of Contents
1. [End-to-End Operator Runbook (From Raw Demos to Inference)](#1-end-to-end-operator-runbook)
2. [Production Server-Side Deployment & ONNX Export](#2-production-server-side-deployment--onnx-export)
3. [Thesis Oral Defense Preparation & Live Demo Setup](#3-thesis-oral-defense-preparation--live-demo-setup)
4. [Academic Publication & Manuscript Submission](#4-academic-publication--manuscript-submission)
5. [Open-Source Packaging & Community Release](#5-open-source-packaging--community-release)
6. [Markdown-First Protocol for Project Maintenance](#6-markdown-first-protocol-for-project-maintenance)

---

## 1. End-to-End Operator Runbook

Follow these steps to run the entire pipeline from scratch on any machine:

### Step 1: Environment Setup
```bash
# Windows PowerShell
.\setup_env.ps1

# Or Windows Command Prompt
setup_env.bat

# Or Linux / macOS
./setup_env.sh
```

### Step 2: Ingest CS2 `.dem` Replay Files
Place downloaded 128-tick CS2 demo files into the data directories:
* Clean baseline matches: `data/raw_demos/clean/*.dem`
* Banned cheater matches: `data/raw_demos/cheaters/*.dem`

### Step 3: Run Batch Preprocessing & ATW Feature Extraction
```bash
python -c "
from src.data.batch_processor import batch_process_demos
batch_process_demos('data/raw_demos/clean', 'data/processed_parquet/clean', is_cheater_dataset=False)
batch_process_demos('data/raw_demos/cheaters', 'data/processed_parquet/cheaters', is_cheater_dataset=True)
"
```

### Step 4: Train ST-Trans Dual-Head Model
```bash
python train.py --data_dir data/processed_parquet --epochs 50 --batch_size 32 --lr 1e-4
```

### Step 5: Run Full Evaluation & Benchmark Report
```bash
pytest tests/
python evaluate.py --checkpoint models/checkpoints/best_model.pt
```

---

## 2. Production Server-Side Deployment & ONNX Export

When model training reaches convergence (AUROC $> 0.98$, False Positive Rate $< 0.01\%$):

### 2.1 Export to ONNX / TorchScript for Sub-500ms Server Execution
1. Export model weights to ONNX format:
   ```python
   import torch
   from src.models.st_transformer import STTrajectoryTransformer

   model = STTrajectoryTransformer.load_from_checkpoint("models/checkpoints/best_model.pt")
   model.eval()
   dummy_input = torch.randn(1, 512, 8)
   dummy_mask = torch.ones(1, 512, dtype=torch.bool)

   torch.onnx.export(
       model, 
       (dummy_input, dummy_mask), 
       "models/st_transformer_deploy.onnx",
       input_names=["telemetry_features", "attention_mask"],
       output_names=["aimbot_probability", "smurf_embedding", "predicted_elo"],
       dynamic_axes={"telemetry_features": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
   )
   ```
2. **Game Server Plugin / Daemon Integration:**
   * Integrate ONNX Runtime C++ / Python engine into CS2 match server hook (e.g. CounterStrikeSharp / Metamod:Source plugin).
   * Telemetry is processed asynchronously at the end of each round; match report generated in $< 500$ ms with zero client-side overhead.

---

## 3. Thesis Oral Defense Preparation & Live Demo Setup

### 3.1 Defense Deliverables Checklist
- [ ] **Slide Deck (15–20 Slides):** Incorporating high-resolution architecture diagrams, ROC curves, ablation tables, and t-SNE latent skill clusters.
- [ ] **Live Interactive Demo Script:**
  * Run `demo_sample.py` on stage to demonstrate live feature extraction of 128-tick mouse view-angles.
  * Show contrastive latent separation between high-tier Faceit Level 10 pros, low-tier players, and artificial aimbots.
- [ ] **Defense Panel Q&A Preparation:**
  * *Question: "Why doesn't client-side anti-cheat (like Vanguard) solve this?"*  
    *Answer:* Hardware DMA (Direct Memory Access) cards and external microcontroller mice execute cheats on secondary computers without touching the game PC's operating system or memory space. Server-side micro-kinematics is immune to DMA evasion because the motor physics must still be sent over the network to register hits.
  * *Question: "Why 128-tick over 64-tick?"*  
    *Answer:* Capturing the 8–12 Hz physiological tremor band requires sampling above the Nyquist frequency ($> 24$ Hz) with sufficient temporal resolution to distinguish human motor jitter from quantized sub-tick interpolation.

---

## 4. Academic Publication & Manuscript Submission

### 4.1 Target Publication Venues
* **Conferences:**
  * IEEE Conference on Games (CoG)
  * ACM CHI PLAY (Computer-Human Interaction in Play)
  * Foundations of Digital Games (FDG)
* **Journals:**
  * IEEE Transactions on Games (ToG)
  * International Journal of Esports (IJES)

### 4.2 Manuscript Section Structure
1. **Abstract:** Context, Minimum Jerk Model, ST-Trans, key results (AUROC, FPR, Latency).
2. **Introduction:** Economic impact of esports, failure of kernel drivers against DMA cheats, smurfing impact on player retention.
3. **Related Work:** Client-side anti-cheats, match-level statistical models, trajectory transformers in autonomous driving/robotics.
4. **Theoretical Framework:** Minimum Jerk optimization, neuromuscular 8–12 Hz tremors, Euler wrapping mathematics.
5. **Methodology:** `demoparser2` ingestion, Active Tracking Window (ATW) formulation, dual-task ST-Trans architecture, Supervised InfoNCE loss.
6. **Experimental Setup:** 2,000 match dataset, strict zero-leakage splits, baseline models (RF, XGBoost, LSTM).
7. **Results & Discussion:** Classification performance, feature ablation tables, inference latency, t-SNE latent embeddings.
8. **Conclusion & Future Work:** Generalization to other tactical FPS (Valorant, Apex Legends), real-time server plugin deployment.

---

## 5. Open-Source Packaging & Community Release

### 5.1 Repository Artifacts for GitHub
* Pre-trained PyTorch checkpoint (`st_trans_cs2_weights.pt`).
* Zenodo DOI dataset link with anonymized ATW Parquet arrays.
* Permissive Open-Source License (MIT License).
* Interactive Colab Notebook for reproducing paper figures with one click.

---

## 6. Markdown-First Protocol for Project Maintenance

> [!IMPORTANT]
> **MANDATORY RULE FOR DEVELOPERS AND AI AGENTS:**  
> Whenever a new script, feature extraction method, dataset format, CLI command, or analysis tool is added to this codebase:
> 1. **Update `THESIS_TRACKER.md`:** Check off completed items, update percentage progress, and log the changes in the Changelog.
> 2. **Update `README.md` & `POST_COMPLETION_GUIDE.md`:** Add execution commands, parameter descriptions, and expected outputs so that neither the user nor future agents forget the process.
> 3. **Add Unit Tests:** Write corresponding test functions in `tests/` to guarantee regression-free execution.
