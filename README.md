# Sustainable AI – Energy-Efficient Prompt & Context Engineering

Group 2

Abdullahi Abdirizak Mohamed - 9082466
Albright Maduka Ifechukwude - 9053136
Jose George - 9082825
Kamamo Lesley Wanjiku - 8984971


## 1. Project Overview

**Sustainable AI** is a proof-of-concept system that helps users design and evaluate
large language model (LLM) prompts with an explicit focus on **energy efficiency**.

The app:

- Estimates approximate **energy consumption** (kWh) for a given model configuration
  and prompt.
- Suggests a **shorter, optimized prompt** that aims to preserve intent while
  reducing token count.
- Uses **unsupervised anomaly detection** to flag configurations that look
  unusually high-energy compared to normal usage.
- Exposes everything through an interactive **Streamlit UI** with visual feedback
  and charts.

>  This is an educational/demo project. All energy estimates are **synthetic and
> heuristic**, not measured from real hardware.

---

## 2. High-Level Architecture

The system is organised into several layers:

1. **User Interface (Streamlit, `src/app.py`)**
   - Collects Role / Context / Expectations and model parameters
     (layers, training hours, FLOPs/hour).
   - Displays baseline vs optimised energy, prompts, complexity metrics, anomaly
     status, and visualisations.

2. **Prompt Processing & Optimisation (`src/prompt_optimizer.py`)**
   - Structured prompt format: **Role, Context, Expectations**.
   - Rule-based simplification: removes filler sentences/phrases and trims
     sections to the most important content.
   - Computes simple complexity metrics and an approximate semantic similarity
     between original and optimised prompts.

3. **Energy Estimation Engine (`src/energy_model.py` + heuristic in `app.py`)**
   - Inputs: `num_layers`, `training_hours`, `flops_per_hour`, `prompt_tokens`.
   - Uses a stable **heuristic formula** (and optionally a regression model) to
     produce human-readable energy estimates (roughly 0.1–30 kWh).

4. **Anomaly Detection Module (`src/anomaly_detector.py`)**
   - Trained on a synthetic dataset using **Isolation Forest** +
     `StandardScaler` in an sklearn `Pipeline`.
   - Runtime function `flag_usage_anomaly(record)` returns:
     `is_anomaly`, `score`, and a short explanation.

5. **Visualisation Layer (`src/visualization.py`)**
   - Reusable Matplotlib/Seaborn charts:
     - Before vs After feature comparison
     - Historical energy distribution
     - Token breakdown per section
     - Anomaly score bar

6. **Data & Model Artifacts**
   - Synthetic training CSVs under `data/`.
   - Trained anomaly detector pipeline at
     `model/anomaly_detector/iso_pipeline.pkl`.
   - `energy_logs.csv` storing historical energy values from the UI.

---

## 3. Repository Structure

A simplified view of the project layout:

```text
SUSTAINABLE_AI/
├── data/
│   ├── synthetic_energy_training_data_200.csv
│   └── energy_training_data.csv           # optional / variant
├── model/
│   └── anomaly_detector/
│       └── iso_pipeline.pkl               # Isolation Forest pipeline
├── notebooks/
│   ├── prediction_engine.ipynb            # energy regression / exploration
│   ├── prompt_processing.ipynb            # prompt metrics & optimisation
│   ├── anomaly_detection.ipynb            # anomaly model training
│   └── SustainableAI_FinalProjectProtocol.ipynb
├── src/
│   ├── app.py                             # Streamlit app (main entry point)
│   ├── energy_model.py                    # energy prediction helpers
│   ├── prompt_optimizer.py                # rule-based prompt simplifier
│   ├── anomaly_detector.py                # Isolation Forest wrapper
│   ├── visualization.py                   # plotting utilities
│   └── models.py                          # PromptData dataclass, etc.
├── documentation/
│   └── (reports, PDFs, screenshots)
├── requirements.txt
└── README.md


Installation & Setup
Create and activate a virtual environment - Python Version 3.11.3
  python -m venv .venv
  source .venv/bin/activate      # on Windows: .venv\Scripts\activate
  pip install -r requirements.txt


Running the App
  cd src
  streamlit run app.py


