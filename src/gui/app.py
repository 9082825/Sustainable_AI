# src/gui/app.py

import streamlit as st
from backend.energy_model import predict_energy, classify_energy_level
from backend.prompt_optimizer import suggest_greener_prompt

# ---------- Page config ----------
st.set_page_config(
    page_title="Sustainable AI – Energy-Aware Prompt Assistant",
    page_icon="🌱",
    layout="wide",
)

# ---------- Header ----------
st.title("🌱 Sustainable AI: Energy-Aware Prompt Assistant")
st.markdown(
    """
This tool helps you estimate the **energy impact** of an LLM prompt  
and suggests a more **energy-efficient alternative**.
"""
)

# ---------- Layout: sidebar for model params ----------
with st.sidebar:
    st.header("⚙️ Model Parameters")

    num_layers = st.number_input(
        "Number of model layers",
        min_value=1,
        max_value=300,
        value=24,
        step=1,
        help="Approximate number of transformer layers or depth of the model.",
    )

    training_time = st.number_input(
        "Training time (hours)",
        min_value=0.0,
        value=72.0,
        step=1.0,
        help="Total number of hours the model was trained.",
    )

    flops_per_hour = st.number_input(
        "FLOPs per hour",
        min_value=0.0,
        value=1e15,
        step=1e13,
        format="%.2e",
        help="Estimated floating point operations per hour during training/inference.",
    )

    st.caption("These inputs are used to estimate the energy footprint.")

# ---------- Main layout: two columns ----------
col_left, col_right = st.columns([2, 1])

# ---------- Left: prompt input & actions ----------
with col_left:
    st.subheader("🧠 LLM Prompt")

    prompt = st.text_area(
        "Enter your prompt:",
        height=180,
        placeholder="E.g., 'Write a detailed report on the impact of climate change on global agriculture...'",
    )

    estimate_button = st.button("🔍 Estimate Energy")
    optimize_button = st.button("🌿 Suggest Greener Prompt")

# ---------- Right: results ----------
with col_right:
    st.subheader("📊 Energy & Insights")

    if "last_energy" not in st.session_state:
        st.session_state.last_energy = None
    if "optimized_prompt" not in st.session_state:
        st.session_state.optimized_prompt = None

    # When user clicks "Estimate Energy"
    if estimate_button:
        if not prompt.strip():
            st.warning("Please enter a prompt before estimating energy.")
        else:
            energy_kwh = predict_energy(
                prompt=prompt,
                num_layers=num_layers,
                training_time=training_time,
                flops_per_hour=flops_per_hour,
            )
            st.session_state.last_energy = energy_kwh

            energy_level = classify_energy_level(energy_kwh)

            st.metric(
                label="Estimated Energy Consumption",
                value=f"{energy_kwh:.3f} kWh",
                delta=None,
            )

            if energy_level == "Low":
                st.success("Energy level: **Low** ✅ This prompt looks efficient.")
            elif energy_level == "Medium":
                st.info("Energy level: **Medium** ⚠️ There might be room for optimization.")
            else:
                st.error("Energy level: **High** 🔴 Consider optimizing your prompt.")

    # When user clicks "Suggest Greener Prompt"
    if optimize_button:
        if not prompt.strip():
            st.warning("Please enter a prompt before optimizing.")
        else:
            optimized = suggest_greener_prompt(prompt)
            st.session_state.optimized_prompt = optimized

    # Display last results if available
    if st.session_state.last_energy is not None:
        st.markdown("---")
        st.markdown("### Last Energy Estimate")
        st.write(f"**{st.session_state.last_energy:.3f} kWh**")

    if st.session_state.optimized_prompt:
        st.markdown("---")
        st.markdown("### 🌿 Suggested Greener Prompt")
        st.write(st.session_state.optimized_prompt)

# ---------- Explanation / documentation section ----------
st.markdown("---")
with st.expander("ℹ️ How this tool works (for your report/presentation)"):
    st.markdown(
        """
- The **frontend (this app)** collects:
  - The user's LLM **prompt**
  - Approximate **model parameters** (layers, training time, FLOPs/hour)

- The **backend models** then:
  1. Use a predictive function to estimate **energy consumption** (in kWh).
  2. Classify energy usage into **Low / Medium / High**.
  3. Optionally generate a **more concise, lower-energy prompt** with similar intent.

You can replace the dummy functions in `backend/energy_model.py` and
`backend/prompt_optimizer.py` with your trained models and real formulas.
"""
    )
