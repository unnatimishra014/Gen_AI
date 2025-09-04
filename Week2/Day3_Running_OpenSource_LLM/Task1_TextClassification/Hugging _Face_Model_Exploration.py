#!/usr/bin/env python3
import streamlit as st
from transformers import pipeline
import torch
import pandas as pd
import altair as alt

# ===== Model Config =====
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TASK = "sentiment-analysis"

# ===== Device Selection =====
device = 0 if torch.cuda.is_available() else -1

# ===== Load Model =====
@st.cache_resource
def load_model():
    return pipeline(task=TASK, model=MODEL_ID, device=device)

classifier = load_model()

# ===== UI =====
st.set_page_config(
    page_title="✨ Smart Sentiment Analyzer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Smart Sentiment Analyzer")
st.subheader("💡 Analyze text sentiment instantly with AI-powered insights!")

# Example sentences
samples = [
    "I love this product! It works perfectly. 😍",
    "This is the worst experience ever. 😡",
    "It's okay, not too bad but not great either. 🤔"
]

st.write("### 🔹 Quick Examples")
cols = st.columns(len(samples))
for i, sentence in enumerate(samples):
    if cols[i].button(f"✏️ {sentence[:25]}..."):
        st.session_state.user_input = sentence

# Text input
user_input = st.text_area("✍️ Enter your text here...", key="user_input", height=120)

# ===== Prediction =====
if user_input.strip():
    # Get main prediction
    result = classifier(user_input)[0]
    label, score = result["label"], result["score"]

    # Color-coded label with emoji
    label_map = {
        "Negative": ("❌ Negative", "red"),
        "Neutral": ("⚖️ Neutral", "orange"),
        "Positive": ("✅ Positive", "green")
    }
    label_text, color = label_map.get(label, (label, "blue"))

    st.markdown(f"### Sentiment: <span style='color:{color}'>{label_text}</span>", unsafe_allow_html=True)
    st.metric("Confidence", f"{score:.2%}", delta_color="normal")

    # Show probability distribution
    res_all = classifier(user_input, top_k=None)
    if isinstance(res_all, dict):
        res_all = [res_all]

    df = pd.DataFrame(res_all)

    st.write("### 📊 Probability Distribution")
    # Pie chart using Altair
    chart = alt.Chart(df).mark_arc().encode(
        theta=alt.Theta(field="score", type="quantitative"),
        color=alt.Color(field="label", type="nominal", scale=alt.Scale(scheme="set1")),
        tooltip=["label", alt.Tooltip("score", format=".2%")]
    )
    st.altair_chart(chart, use_container_width=True)

    # Expandable raw output
    with st.expander("🔍 See Raw Model Output"):
        st.json(res_all)
else:
    st.warning("⚠️ Please enter some text to analyze.")
