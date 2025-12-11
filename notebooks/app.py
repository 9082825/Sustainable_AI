import streamlit as st
import numpy as np
import joblib
from typing import Optional

# ---------------------------------------------------------------------
# Model loading / prediction helpers
# ---------------------------------------------------------------------

@st.cache_resource
def load_energy_model(path: str = "model/energy_predictor/energy_predictor.pkl"):
    """
    Try to load a trained energy prediction model.
    If not available, returns None and a simple heuristic will be used.
    """
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        # We keep this as a warning so the app still runs.
        st.warning(
            "⚠️ Could not load trained energy model from "
            f"`{path}`. Falling back to a simple heuristic."
        )
        return None


def estimate_prompt_tokens(prompt: str) -> int:
    """
    Very rough token estimate: number of whitespace-separated words.
    In a real system, you'd use the tokenizer of the target LLM.
    """
    return max(1, len(prompt.split()))


def predict_energy(
    num_layers: int,
    training_hours: float,
    flops_per_hour: float,
    prompt: str,
    model=None,
    default_gpu_power: float = 250.0,
    default_batch_size: int = 32,
) -> float:
    """
    Predict energy consumption (kWh) for a given configuration.
    - If a trained model is available, use it.
    - Otherwise, use a simple heuristic based on layers, hours, FLOPs, and prompt length.
    """
    tokens = estimate_prompt_tokens(prompt)

    if model is not None:
        # Adjust this feature order to match your trained model
        X = np.array(
            [[
                num_layers,
                training_hours,
                flops_per_hour,
                tokens,            # prompt_tokens
                default_gpu_power, # gpu_power_watts
                default_batch_size # batch_size
            ]],
            dtype=float,
        )
        return float(model.predict(X)[0])

    # ----- Heuristic fallback (no trained model) -----
    # Base term scaled into a reasonable range
    base = num_layers * training_hours * flops_per_hour * 1e-19
    complexity = np.log1p(tokens) / 10.0
    energy = base * (0.5 + complexity)

    # Clip to a sensible range
    return float(np.clip(energy, 0.1, 5000.0))


# ---------------------------------------------------------------------
# Prompt simplification helper
# (You can later replace this with your T5 / GPT-2 / OpenAI-based optimizer)
# ---------------------------------------------------------------------

def simplify_prompt(prompt: str) -> str:
    """
    Placeholder simplifier:
    - If the prompt is longer than 40 tokens, keep the first 40 and add '...'.
    - Otherwise, return it as is.

    Replace this later with a call to your:
    - T5 model, or
    - sentence-embedding-based optimizer, or
    - OpenAI API.
    """
    tokens = prompt.split()
    if len(tokens) <= 40:
        return prompt.strip()
    return " ".join(tokens[:40]) + " ..."


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Sustainable AI – Sustainable AI Prompt UI",
        page_icon="⚡",
        layout="centered",
    )

    st.title("⚡ Sustainable AI")
    st.caption("Part of the Sustainable AI project – estimates energy and suggests lower-energy prompts.")

    st.markdown(
        """
        ### How it works

        1. Enter your **prompt** and basic **model configuration**.  
        2. The app estimates **energy consumption (kWh)** for the given setup.  
        3. It also proposes a **simpler alternative prompt** and shows its **estimated energy**.

        > Note: This is a proof-of-concept. Estimates are based on a synthetic or approximated model, not on real hardware measurements.
        """
    )

    # ---------------------- Input form ----------------------
    with st.form("energy_form"):
        prompt = st.text_area(
            "📝 Prompt text",
            height=160,
            placeholder="Describe what you want the model to do...",
        )

        col1, col2 = st.columns(2)

        with col1:
            num_layers = st.number_input(
                "Number of model layers",
                min_value=1,
                max_value=1000,
                value=24,
                step=1,
            )

            training_hours = st.number_input(
                "Training time (hours)",
                min_value=0.1,
                max_value=2000.0,
                value=8.0,
                step=0.5,
            )

        with col2:
            flops_per_hour = st.number_input(
                "Estimated FLOPs per hour",
                min_value=1e5,
                max_value=1e21,
                value=1e18,
                step=1e16,
                format="%.2e",
                help="Use scientific notation (e.g., 1e18).",
            )

        submitted = st.form_submit_button("🚀 Estimate energy")

    # ---------------------- Compute & display results ----------------------
    if submitted:
        if not prompt.strip():
            st.error("Please enter a prompt before estimating energy.")
            return

        # Load model (or heuristic)
        model = load_energy_model()

        # Baseline energy for original prompt
        baseline_energy = predict_energy(
            num_layers=num_layers,
            training_hours=training_hours,
            flops_per_hour=flops_per_hour,
            prompt=prompt,
            model=model,
        )

        # Alternative (simplified) prompt and its energy
        alt_prompt = simplify_prompt(prompt)
        alt_energy = predict_energy(
            num_layers=num_layers,
            training_hours=training_hours,
            flops_per_hour=flops_per_hour,
            prompt=alt_prompt,
            model=model,
        )

        delta_energy = alt_energy - baseline_energy

        st.markdown("---")
        st.subheader("🔍 Results")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                label="Baseline energy (kWh)",
                value=f"{baseline_energy:,.3f}",
            )
            st.text_area(
                "Original prompt",
                value=prompt,
                height=180,
            )

        with colB:
            st.metric(
                label="Alternative energy (kWh)",
                value=f"{alt_energy:,.3f}",
                delta=f"{delta_energy:,.3f} kWh",
                delta_color="inverse" if delta_energy < 0 else "normal",
            )
            st.text_area(
                "Alternative prompt (suggested)",
                value=alt_prompt,
                height=180,
            )

        st.caption(
            "Lower energy is better. Negative delta means the alternative prompt is estimated to use less energy."
        )


if __name__ == "__main__":
    main()
