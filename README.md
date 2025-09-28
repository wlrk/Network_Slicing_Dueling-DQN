<<<<<<< HEAD
﻿# DQN/Dueling DQN for 5G Network Slicing

This repository provides a reinforcement learning–based simulator for dynamic bandwidth allocation between **eMBB** (enhanced Mobile Broadband) and **URLLC** (Ultra-Reliable Low-Latency Communications) slices.  
It supports both **DQN** and **Dueling DQN** (with a `--dueling` flag) for performance comparison.

---

## 📂 Project Structure
├── build_trace_from_detector.py # Convert detector outputs → traffic trace (CSV)
├── config.py # Central config (bandwidth, traffic, RL params)
├── main.py # Training entrypoint (DQN / Dueling DQN)
├── eval.py # Evaluation using saved model + trace
├── plot_from_console.py # Training visualization from train-metrics.csv
├── plot_episode_from_csv.py # Episode-level visualization from monitor.csv
└── envs/
└── network_env.py # Custom OpenAI Gym environment

---

## ⚙️ Environment

- Python 3.7.x  
- [TensorFlow 1.15]
- [stable-baselines 2.10.1] 
- gym 0.15.4  
- numpy, pandas, matplotlib  


C:\venvs\slicing_py37\Scripts\Activate.ps1
pip install -r requirements.txt

1. Preprocess traffic trace

If you already have detector outputs (hdf5_predictions.csv):

python build_trace_from_detector.py


This generates:

runs/trace_from_detector/trace_ep2000_slot0p5ms.csv

runs/trace_from_detector/trace_with_labels.csv

2. Train model
python main.py --timesteps 200000
python main.py --timesteps 200000 --dueling


Output is saved in:

runs/YYYYMMDD-HHMMSS/output/
    ├── dqn_slicing_model.zip / dueling_dqn_slicing_model.zip
    ├── monitor.csv
    ├── train-metrics.csv
    ├── episode_log.csv
    └── train-console.txt

3. Evaluate
python eval.py


Uses MODEL_PATH and TRACE_FILE specified in config.py.

4. Visualize
# From train-metrics.csv
python plot_from_console.py --logdir runs --outdir plots

# From monitor.csv
python plot_episode_from_csv.py runs/**/output/monitor.csv


Generated plots include:

Episode reward

QoE (eMBB, URLLC)

Spectral efficiency (SE)

Utility



Metrics

Reward: RL objective

QoE: service success ratio per slice

Spectral Efficiency (SE): throughput / bandwidth

Utility: weighted combination (configurable)


 Key Features

Config-first (all parameters in config.py)

Toggle DQN vs Dueling DQN (--dueling)

Trace / Synth traffic modes supported

Automatic logging (CSV, TensorBoard, console)

Plot scripts for reproducible visualization
=======
# Network_Slicing_Dueling-DQN
>>>>>>> 983d7ed40f0c9daee2b0bcec8c079bfec21983c2
