# dual_role_cot_app.py
"""
Streamlit App: Individual Inputs for Role-Based + CoT Model
Generates meaningful, justified outputs & creates a comparison PDF
"""

import io, difflib
from datetime import datetime
from typing import Dict, Any, List
import streamlit as st
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ------------ Helpers ------------
def create_generator(model_name: str, task: str = "text2text-generation", device: int = -1):
    return pipeline(task, model=model_name, tokenizer=model_name, device=device)

def generate(gen, prompt: str, max_len: int, temp: float, top_p: float) -> str:
    out = gen(prompt, max_length=max_len, do_sample=(temp > 0.0),
              temperature=temp, top_p=top_p)
    if isinstance(out, list):
        return out[0].get("generated_text", str(out))
    return str(out)

def diff_text(a: str, b: str) -> str:
    return "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                          fromfile="Role", tofile="CoT", lineterm=""))

def reflection(role_out: str, cot_out: str) -> str:
    r = difflib.SequenceMatcher(None, role_out, cot_out).ratio()
    if r > 0.8: sim = "Outputs are very similar."
    elif r > 0.4: sim = "Outputs differ moderately."
    else: sim = "Outputs diverge strongly."
    return f"Role-based explains as a teacher, CoT gives reasoning steps. {sim}"

def build_pdf(entries: List[Dict[str, Any]], title="Role_vs_CoT_Report") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = [Paragraph(title, styles["Title"]),
             Paragraph(f"Generated: {datetime.utcnow().isoformat()} UTC", styles["Normal"]),
             Spacer(1, 12)]
    data = [["#", "Role Prompt", "Role Output", "CoT Prompt", "CoT Output", "Diff", "Reflection"]]
    for i,e in enumerate(entries,1):
        def tr(s,n=300): return (s[:n]+"...") if len(s)>n else s
        data.append([str(i), tr(e["role_prompt"],200), tr(e["role"],300),
                     tr(e["cot_prompt"],200), tr(e["cot"],300), tr(e["diff"],200), tr(e["reflection"],200)])
    tbl = Table(data, colWidths=[20,100,140,100,140,100,100])
    tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                             ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f0f0f0")),
                             ("FONTSIZE",(0,0),(-1,-1),8)]))
    elems.append(tbl); doc.build(elems); buf.seek(0); return buf.read()


# ------------ Streamlit App ------------
st.set_page_config("Role vs CoT (Individual Inputs)", layout="wide")
st.title("🎓 Role-based vs 🧠 Chain-of-Thought — AI Comparison")

st.sidebar.header("Model Settings")
role_model = st.sidebar.text_input("Role-based Model", "google/flan-t5-large")
cot_model  = st.sidebar.text_input("CoT Model", "google/flan-t5-base")
max_len = st.sidebar.slider("Max Length", 100, 1024, 400, step=50)
temp    = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)
top_p   = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)

# Two separate inputs
role_prompt = st.text_area("✍️ Enter Role-based Prompt (as Teacher, Expert etc.)",
    "You are a high school biology teacher. Explain photosynthesis to students in simple words.")
cot_prompt = st.text_area("🧩 Enter CoT Prompt (Step-by-step reasoning)",
    "Explain the process of Photosynthesis step by step, reasoning clearly each step.")

if st.button("Run Both Models"):
    with st.spinner("Loading models and generating..."):
        role_gen = create_generator(role_model)
        cot_gen  = create_generator(cot_model)

        role_out = generate(role_gen, role_prompt, max_len, temp, top_p)
        cot_out  = generate(cot_gen,  cot_prompt, max_len, temp, top_p)

        d = diff_text(role_out, cot_out)
        refl = reflection(role_out, cot_out)
        entry = {"role_prompt": role_prompt, "cot_prompt": cot_prompt,
                 "role": role_out, "cot": cot_out, "diff": d, "reflection": refl}

    cols = st.columns(2)
    with cols[0]: st.subheader("📘 Role-based Output"); st.write(role_out)
    with cols[1]: st.subheader("🧠 CoT Output"); st.write(cot_out)

    st.subheader("🔍 Diff")
    st.code(d or "(No major diff)")
    st.subheader("💡 Reflection")
    st.write(refl)

    pdf_bytes = build_pdf([entry])
    st.download_button("📥 Download PDF Report", pdf_bytes, "role_vs_cot.pdf", "application/pdf")
