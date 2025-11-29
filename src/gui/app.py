import streamlit as st
from backend.energy_model import predict_energy, classify_energy_level
from backend.prompt_optimizer import suggest_greener_prompt

# ---------- Page config ----------
st.set_page_config(
    page_title="Sustainable AI – Energy-Aware Prompt Assistant",
    page_icon="🌱",
    layout="wide",
)
