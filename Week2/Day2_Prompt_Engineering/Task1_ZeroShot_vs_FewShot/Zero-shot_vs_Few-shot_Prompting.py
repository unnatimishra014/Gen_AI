import streamlit as st
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

# -----------------------------
# Load Hugging Face models (cached for speed)
# -----------------------------
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

@st.cache_resource
def load_few_shot():
    return pipeline("text2text-generation", model="google/flan-t5-base")

zero_shot_classifier = load_zero_shot()
few_shot_model = load_few_shot()

# -----------------------------
# Helper functions
# -----------------------------
def zero_shot_sentiment(sentence: str):
    labels = ["positive", "negative"]
    result = zero_shot_classifier(sentence, candidate_labels=labels)
    return result["labels"][0], result["scores"][0]

def few_shot_sentiment(sentence: str):
    prompt = f"""
    Decide whether the sentiment of the following sentence is Positive or Negative.
    Give short justification.

    Example 1:
    Sentence: I love this movie, it was amazing!
    Answer: Positive - expresses enjoyment.

    Example 2:
    Sentence: This food tastes terrible and I hate it.
    Answer: Negative - expresses dislike.

    Sentence: {sentence}
    Answer:
    """
    output = few_shot_model(prompt, max_length=60, do_sample=False)[0]["generated_text"]
    return output.strip()

# -----------------------------
# Generate PDF Report
# -----------------------------
def generate_pdf(zs_input, zs_result, zs_conf, fs_input, fs_result):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Sentiment Classification Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Input section
    elements.append(Paragraph("<b>User Inputs</b>", styles['Heading2']))
    elements.append(Paragraph(f"Zero-shot Input: {zs_input}", styles['Normal']))
    elements.append(Paragraph(f"Few-shot Input: {fs_input}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Results table
    data = [["Approach", "Prediction", "Confidence / Reasoning"]]
    data.append(["Zero-shot", zs_result, f"Confidence: {zs_conf:.2f}"])
    data.append(["Few-shot", fs_result.split("-",1)[0], fs_result])

    table = Table(data, colWidths=[100, 150, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),1,colors.black),
    ]))

    elements.append(Paragraph("<b>Results Comparison</b>", styles['Heading2']))
    elements.append(table)

    doc.build(elements)
    return temp_file.name

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Zero-shot vs Few-shot Sentiment", layout="centered")
st.title("📊 Sentiment Classification: Zero-shot vs Few-shot")

col1, col2 = st.columns(2)
with col1:
    zs_input = st.text_area("✍️ Enter sentence for Zero-shot:", height=120)
with col2:
    fs_input = st.text_area("✍️ Enter sentence for Few-shot:", height=120)

if st.button("Run Analysis"):
    if zs_input.strip() or fs_input.strip():
        with st.spinner("Analyzing..."):
            # Zero-shot
            zs_label, zs_conf = ("", 0.0)
            if zs_input.strip():
                zs_label, zs_conf = zero_shot_sentiment(zs_input)

            # Few-shot
            fs_output = ""
            if fs_input.strip():
                fs_output = few_shot_sentiment(fs_input)

        # Display results
        if zs_input:
            st.subheader("🔹 Zero-shot Result")
            st.write(f"**Prediction:** {zs_label}")
            st.write(f"**Confidence:** {zs_conf:.2f}")

        if fs_input:
            st.subheader("🔹 Few-shot Result")
            st.write(fs_output)

        # Generate and download PDF
        pdf_path = generate_pdf(zs_input, zs_label, zs_conf, fs_input, fs_output)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Report (PDF)",
                data=pdf_file,
                file_name="sentiment_report.pdf",
                mime="application/pdf"
            )
        os.remove(pdf_path)
    else:
        st.warning("Please enter at least one sentence.")
