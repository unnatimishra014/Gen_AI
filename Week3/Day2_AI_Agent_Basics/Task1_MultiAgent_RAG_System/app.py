
import streamlit as st
from pathlib import Path
from agents import Document, build_system
from gtts import gTTS
import base64
import os

# ---- Page Config ----
st.set_page_config(
    page_title="HR Buddy 🤖",
    page_icon="🤖",
    layout="wide"
)

# ---- Sidebar ----
with st.sidebar:
    st.title("🧭 Navigator")
    st.caption("Your personal HR chatbot for salary 💼 and insurance 🛡️ questions.")

    st.markdown("### ✨ Quick Questions")
    if st.button("💰 How do I calculate annual salary?"):
        st.session_state.prefill = "How do I calculate annual salary?"
    if st.button("📄 What is included in my CTC?"):
        st.session_state.prefill = "What is included in my CTC?"
    if st.button("🛡️ How can I claim insurance?"):
        st.session_state.prefill = "How can I claim insurance?"
    if st.button("📑 What documents are needed to claim insurance?"):
        st.session_state.prefill = "What documents are needed to claim insurance?"

    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Load docs & coordinator ----
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"

def safe_read(path: Path, default: str) -> str:
    """Read file safely, return default text if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default

salary_docs = [Document("salary", safe_read(data_dir / "salary.txt", "Salary info unavailable."))]
insurance_docs = [Document("insurance", safe_read(data_dir / "insurance.txt", "Insurance info unavailable."))]

coordinator = build_system(salary_docs, insurance_docs)

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ---- Header ----
st.markdown("<h1 style='text-align: center;'>🤖 HR Buddy</h1>", unsafe_allow_html=True)
st.caption("Ask me anything about your **Salary 💼** or **Insurance 🛡️** and I’ll give you a clear answer.")

# ---- Display chat history ----
for role, content, agent in st.session_state.messages:
    avatar = "🧑" if role == "user" else ("💼" if agent == "Salary Agent" else "🛡️")
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# ---- Chat input ----
prompt = st.chat_input(
    st.session_state.prefill or "Type your HR question here…"
)
st.session_state.prefill = ""

if prompt:
    # Add user message
    st.session_state.messages.append(("user", prompt, ""))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Route query
    result = coordinator.route(prompt)
    answer = result.answer

    # Show assistant reply
    with st.chat_message("assistant", avatar=("💼" if result.agent_name == "Salary Agent" else "🛡️")):
        st.markdown(f"**{result.agent_name} (HR Buddy 🤖):**\n\n{answer}")

        # Voice output
        try:
            tts = gTTS(answer)
            tts.save("reply.mp3")
            audio_file = open("reply.mp3", "rb").read()
            audio_b64 = base64.b64encode(audio_file).decode()
            st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
        except Exception:
            st.warning("Voice assistant unavailable (gTTS error).")

    # Save assistant reply
    st.session_state.messages.append(("assistant", f"{answer}", result.agent_name))
