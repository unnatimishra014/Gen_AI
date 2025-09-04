
# streamlit_local_llm_interactive.py
import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# ----- Suppress warnings -----
logging.set_verbosity_error()

# ----- Page config -----
st.set_page_config(
    page_title="🧠 Local LLM Interface",
    page_icon="🤖",
    layout="wide"
)

# ----- Sidebar -----
st.sidebar.title("Settings ⚙️")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device}")

MODEL_OPTIONS = {
    "GPT-2 (CPU friendly)": "gpt2",
    # Add more models if available locally
}
selected_model = st.sidebar.selectbox("Select Model", options=list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[selected_model]

# ----- Load model -----
@st.cache_resource
def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None

with st.spinner("Loading model... ⏳"):
    tokenizer, model = load_model(MODEL_PATH)

if tokenizer is None or model is None:
    st.stop()

st.success("✅ Model loaded successfully!")
st.title("🧠 Local LLM Interface")

# ----- Layout: two columns -----
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Enter your prompt")
    prompt = st.text_area("", height=150, placeholder="Type something like 'Write a short poem about AI...'")
    generate_btn = st.button("Generate 🤖")

with col2:
    st.subheader("💡 Example Prompts")
    st.write("• Write a short poem about a robot learning to love")
    st.write("• Explain AI as if I were 5 years old")
    st.write("• Start a mystery story on a rainy night")
    st.write("• Give me a funny programmer joke")
    st.write("• Summarize supervised vs unsupervised learning")

# ----- Generate output -----
if generate_btn:
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt first!")
    else:
        try:
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed = time.time() - start_time

            # ----- Display output -----
            st.markdown("### 📝 Output:")
            st.info(text_output)
            st.success(f"⏱ Response time: {elapsed:.2f} seconds")
        except Exception as e:
            st.error(f"❌ Error generating output: {e}")
