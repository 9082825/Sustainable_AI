import re
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# top of app.py (near other imports)
from pandas.errors import ParserError


from sklearn.linear_model import LinearRegression

from visualization import (
    feature_comparison_bar,
    energy_distribution_hist,
    token_breakdown_bar,
    anomaly_score_bar,
)
from models import PromptData

# ----------------------------
# Import your project modules
# ----------------------------
try:
    from energy_model import load_energy_model, predict_energy
except ImportError:
    load_energy_model = None
    predict_energy = None

try:
    from prompt_optimizer import optimize_prompt
except ImportError:
    optimize_prompt = None

try:
    from anomaly_detector import flag_usage_anomaly
except ImportError:
    flag_usage_anomaly = None

LOG_PATH = "reports\energy_logs.csv"

# --- Use only the heuristic model for nicer-scale energies ---
USE_TRAINED_MODEL = False

# --- Global matplotlib style: smaller, consistent fonts ---
plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


# ----------------------------
# Helper functions
# ----------------------------

def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt.split()))


def fallback_predict_energy(
    num_layers: int,
    training_hours: float,
    flops_per_hour: float,
    prompt_tokens: int,
) -> float:
    """
    Heuristic with stronger dependence on prompt length and a sane scale.
    Typical range: ~0.5–20 kWh.
    """
    # Normalise around a "typical" config (24 layers, 4h, 1e20 FLOPs)
    base_layers = num_layers / 24.0
    base_hours = training_hours / 4.0
    base_flops = flops_per_hour / 1e20

    # Complexity from token count (≈ [0, 1.5] for 1–200 tokens)
    complexity = np.log1p(prompt_tokens) / np.log(200)

    # Training contribution
    training_energy = 5.0 * base_layers * base_hours * base_flops

    # Per-prompt overhead dominated by tokens
    prompt_energy = 5.0 * complexity

    energy = training_energy + prompt_energy
    return float(np.clip(energy, 0.1, 30.0))


def safe_predict_energy(
    num_layers: int,
    training_hours: float,
    flops_per_hour: float,
    prompt_tokens: int,
    gpu_power_watts: float = 250.0,
    batch_size: int = 32,
    model=None,
) -> float:
    # For the demo, always use heuristic to avoid crazy scales
    if (not USE_TRAINED_MODEL) or (predict_energy is None):
        return fallback_predict_energy(num_layers, training_hours, flops_per_hour, prompt_tokens)

    try:
        return predict_energy(
            num_layers=num_layers,
            training_hours=training_hours,
            flops_per_hour=flops_per_hour,
            prompt_tokens=prompt_tokens,
            gpu_power_watts=gpu_power_watts,
            batch_size=batch_size,
            model=model,
        )
    except Exception as e:
        st.warning(f"Energy model error: {e}. Using heuristic instead.")
        return fallback_predict_energy(num_layers, training_hours, flops_per_hour, prompt_tokens)


def safe_optimize_prompt(prompt: str) -> dict:
    if optimize_prompt is None:
        st.info("Prompt optimizer module not found. Returning original prompt.")
        return {
            "original_prompt": prompt,
            "simplified_prompt": prompt,
            "complexity_before": {},
            "complexity_after": {},
            "semantic_similarity": 1.0,
        }

    try:
        return optimize_prompt(prompt)
    except Exception as e:
        st.warning(f"Prompt optimizer error: {e}. Using original prompt.")
        return {
            "original_prompt": prompt,
            "simplified_prompt": prompt,
            "complexity_before": {},
            "complexity_after": {},
            "semantic_similarity": 1.0,
        }


def safe_flag_anomaly(
    num_layers: int,
    training_hours: float,
    flops_per_hour: float,
    energy_kwh: float,
    prompt_tokens: int,
) -> dict:
    if flag_usage_anomaly is None:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "explanation": "Anomaly detector not available.",
        }

    record = {
        "num_layers": num_layers,
        "training_hours": training_hours,
        "flops_per_hour": flops_per_hour,
        "energy_kwh": energy_kwh,
        "prompt_tokens": prompt_tokens,
    }

    try:
        result = flag_usage_anomaly(record)
        if result is None:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "explanation": "Anomaly detector returned no result.",
            }

        return {
            "is_anomaly": bool(result.get("is_anomaly", False)),
            "score": float(result.get("score", 0.0)),
            "explanation": str(result.get("explanation", "")),
        }
    except Exception as e:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "explanation": f"Error during anomaly detection: {e}",
        }


def log_energy(energy_kwh: float, path: str = LOG_PATH) -> None:
    """
    Append current energy_kwh to a CSV for historical plots.
    If the existing file is corrupted or has inconsistent columns,
    we ignore it and start a fresh log.
    """
    row = pd.DataFrame([{"energy_kwh": energy_kwh}])

    try:
        existing = pd.read_csv(path)
        # Keep only the energy_kwh column if it exists
        if "energy_kwh" in existing.columns:
            existing = existing[["energy_kwh"]]
        df = pd.concat([existing, row], ignore_index=True)
    except (FileNotFoundError, ParserError):
        # No file or bad file → start fresh
        df = row

    df.to_csv(path, index=False)



def mock_energy_curve_plot(num_layers: int, energy_kwh: float) -> plt.Figure:
    """
    Small mocked curve, similar shape to your original GUI, but compact.
    """
    data = pd.DataFrame({
        "num_layers": [2, 4, 6, 8, 10],
        "training_hours": [1, 2, 3, 4, 5],
        "flops_per_hour": [10, 20, 40, 60, 90],
        "energy_kwh": [0.5, 1.3, 2.8, 4.5, 6.9],
    })

    X = data[["num_layers", "training_hours", "flops_per_hour"]]
    y = data["energy_kwh"]
    lr = LinearRegression()
    lr.fit(X, y)
    data["predicted"] = lr.predict(X)

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(data["num_layers"], data["energy_kwh"], "o-", label="Actual")
    ax.plot(data["num_layers"], data["predicted"], "x--", label="Predicted")

    ax.scatter([num_layers], [energy_kwh], s=80, marker="D", color="teal", zorder=5)
    ax.annotate(
        "Prediction",
        xy=(num_layers, energy_kwh),
        xytext=(num_layers + 0.2, energy_kwh + 0.3),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="teal", alpha=0.9),
        fontsize=8,
    )

    ax.set_title("Energy Prediction vs Actual")
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Energy (kWh)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def parse_structured_prompt(full_prompt: str):
    """
    Parse 'Role: ... Context: ... Expectations: ...' back into 3 sections.
    Used when the Improve button is clicked.
    """
    pattern = r"Role:\s*(.*?)\s*Context:\s*(.*?)\s*Expectations:\s*(.*)"
    m = re.search(pattern, full_prompt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None, None
    role, context, expectations = [part.strip() for part in m.groups()]
    return role, context, expectations


# ----------------------------
# Streamlit App
# ----------------------------

def main():
    st.set_page_config(
        page_title="Sustainable AI – Energy-Efficient Prompt Engineering",
        page_icon="⚡",
        layout="wide",
    )

    # Simple, clean styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #101218;
            color: #f5f5f5;
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 2rem;
        }
        .hero-subtitle {
            font-size: 1.0rem;
            margin-top: 1rem;
            color: #d0d0d0;
        }
        .prompt-card {
            background-color: #f6a623;
            padding: 1rem 1.2rem;
            border-radius: 6px;
            color: #111111;
            font-size: 0.95rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([2, 3])

    # ---------- Left hero ----------
    with col_left:
        st.markdown(
            """
            <div class="hero-title">
            Sustainable AI:<br/>
            Energy-Efficient Prompt<br/>
            and Context Engineering
            </div>
            <div class="hero-subtitle">
            Estimate energy usage, optimize prompts, and detect unusually
            wasteful configurations for large language models.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Right: prompt + controls ----------
    with col_right:
        st.markdown("### Enter prompt here:")

        # We keep our own "default" text in session_state, but DO NOT use keys on the widgets.
        default_role = (
            "You are an AI assistant helping ML engineers design "
            "energy-efficient large language model workflows."
        )
        default_context = (
            "Our team is building a customer-support chatbot using a large language model "
            "hosted in the cloud. We have historical chat logs, limited GPU budget, and strict "
            "latency targets. Right now, prompts are long and repetitive."
        )
        default_expectations = (
            "Remove redundancy and unnecessary wording. Briefly explain why the optimized "
            "version should reduce token usage and energy while keeping responses accurate."
        )

        if "role_default" not in st.session_state:
            st.session_state["role_default"] = default_role
        if "context_default" not in st.session_state:
            st.session_state["context_default"] = default_context
        if "expectations_default" not in st.session_state:
            st.session_state["expectations_default"] = default_expectations

        with st.form("sustainable_form"):
            st.markdown('<div class="prompt-card">', unsafe_allow_html=True)

            role = st.text_area(
                "Role",
                value=st.session_state["role_default"],
                height=70,
            )
            context = st.text_area(
                "Context",
                value=st.session_state["context_default"],
                height=70,
            )
            expectations = st.text_area(
                "Expectations",
                value=st.session_state["expectations_default"],
                height=70,
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### Set parameters:")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                num_layers = st.number_input(
                    "# Layers", min_value=1, max_value=2000, value=7, step=1
                )
            with pcol2:
                training_hours = st.number_input(
                    "Training time (hrs)",
                    min_value=0.1,
                    max_value=2000.0,
                    value=2.0,
                    step=0.5,
                )
            with pcol3:
                flops_exp = st.number_input(
                    "FLOPs/hr (10^x)",
                    min_value=5.0,
                    max_value=25.0,
                    value=20.0,
                    step=1.0,
                )
            flops_per_hour = float(10 ** flops_exp)

            b1, b2 = st.columns(2)
            with b1:
                submit_clicked = st.form_submit_button("Submit")
            with b2:
                improve_clicked = st.form_submit_button("Improve")

    submitted = submit_clicked or improve_clicked

    if submitted:
        full_prompt = f"Role:\n{role}\n\nContext:\n{context}\n\nExpectations:\n{expectations}"
        prompt_tokens = estimate_prompt_tokens(full_prompt)

        # (We still create this, even if we only use the heuristic)
        model = None
        if USE_TRAINED_MODEL and (load_energy_model is not None):
            try:
                model = load_energy_model()
            except Exception as e:
                st.warning(f"Could not load energy model: {e}. Using heuristic.")

        baseline_energy = safe_predict_energy(
            num_layers=num_layers,
            training_hours=training_hours,
            flops_per_hour=flops_per_hour,
            prompt_tokens=prompt_tokens,
            model=model,
        )
        log_energy(baseline_energy)

        opt_result = safe_optimize_prompt(full_prompt)
        simplified_prompt = opt_result["simplified_prompt"]
        complexity_before = opt_result.get("complexity_before", {})
        complexity_after = opt_result.get("complexity_after", {})
        semantic_similarity = opt_result.get("semantic_similarity", 1.0)
        simplified_tokens = estimate_prompt_tokens(simplified_prompt)

        alt_energy = safe_predict_energy(
            num_layers=num_layers,
            training_hours=training_hours,
            flops_per_hour=flops_per_hour,
            prompt_tokens=simplified_tokens,
            model=model,
        )
        delta_energy = alt_energy - baseline_energy

        anomaly_info = safe_flag_anomaly(
            num_layers=num_layers,
            training_hours=training_hours,
            flops_per_hour=flops_per_hour,
            energy_kwh=baseline_energy,
            prompt_tokens=prompt_tokens,
        )

        # If the user clicked "Improve", push the optimized text back into defaults
        if improve_clicked:
            r_new, c_new, e_new = parse_structured_prompt(simplified_prompt)
            if r_new:
                st.session_state["role_default"] = r_new
            if c_new:
                st.session_state["context_default"] = c_new
            if e_new:
                st.session_state["expectations_default"] = e_new

        # ---------- Results ----------
        st.markdown("---")
        st.subheader("Energy prediction and prompt optimization")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Baseline energy (kWh)", f"{baseline_energy:,.3f}")
            st.text_area("Original prompt", full_prompt, height=200)
            st.caption(f"Estimated tokens: {prompt_tokens}")
        with colB:
            st.metric(
                "Alternative energy (kWh)",
                f"{alt_energy:,.3f}",
                delta=f"{delta_energy:,.3f} kWh",
                delta_color="inverse" if delta_energy < 0 else "normal",
            )
            st.text_area("Optimized prompt (suggested)", simplified_prompt, height=200)
            st.caption(
                f"Estimated tokens: {simplified_tokens} • "
                f"Semantic similarity: {semantic_similarity:.2f}"
            )

        # --- First row of charts: energy curve + complexity ---
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown("##### Energy Prediction vs Actual")
            fig_curve = mock_energy_curve_plot(int(num_layers), baseline_energy)
            st.pyplot(fig_curve, use_container_width=False)
        with row1_col2:
            if complexity_before and complexity_after:
                st.markdown("##### Prompt complexity – before vs after")
                fig_feat = feature_comparison_bar(complexity_before, complexity_after)
                st.pyplot(fig_feat, use_container_width=False)

        # --- Second row: history + token breakdown ---
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.markdown("##### Energy usage vs previous runs")
            fig_hist = energy_distribution_hist(baseline_energy, log_path=LOG_PATH)
            if fig_hist is not None:
                st.pyplot(fig_hist, use_container_width=False)
            else:
                st.caption("Not enough historical data yet.")
        with row2_col2:
            st.markdown("##### Token breakdown by section")
            prompt_data = PromptData(role=role, context=context, expectations=expectations)
            fig_tokens = token_breakdown_bar(prompt_data)
            st.pyplot(fig_tokens, use_container_width=False)

        # --- Anomaly section ---
        st.markdown("---")
        st.subheader("🚨 Anomaly Detection")
        if anomaly_info["is_anomaly"]:
            st.error(
                f"This configuration is flagged as **high-energy / anomalous**.\n\n"
                f"Score: {anomaly_info['score']:.3f}\n\n"
                f"Reason: {anomaly_info['explanation']}"
            )
        else:
            st.success("This configuration is within normal energy usage range.")
            st.caption(f"Reason: {anomaly_info['explanation']}")

        fig_anom = anomaly_score_bar(anomaly_info["score"], anomaly_info["is_anomaly"])
        st.pyplot(fig_anom, use_container_width=False)


if __name__ == "__main__":
    main()
