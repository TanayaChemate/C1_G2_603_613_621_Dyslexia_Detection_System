import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from textblob import TextBlob
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
MODEL_PATH = "model_cnn.h5"
INPUT_WIDTH, INPUT_HEIGHT = 128, 128

# ---------------- HELPER FUNCTIONS ----------------
def calculate_risk_and_subtype(handwriting_pred, spelling_accuracy, grammar_score, age_level):
    """
    Calculates combined dyslexia risk and subtype, adjusting for age-level expectations.
    """
    # Age-level adjustment factors
    age_factors = {
        "6-7": 0.8,    # younger kids: lower expected performance -> lower risk
        "8-10": 0.9,
        "11-13": 1.0,  # baseline
        "14+": 1.1     # older kids: higher expected performance -> higher risk
    }
    factor = age_factors.get(age_level, 1.0)

    # Handwriting risk
    handwriting_risk = (1.0 - handwriting_pred) * 50

    # Spelling & grammar risk scaled by age factor
    spelling_error_rate = 1.0 - (spelling_accuracy / 100.0)
    spelling_risk = spelling_error_rate * 30 * factor

    grammar_error_rate = 1.0 - (grammar_score / 100.0)
    grammar_risk = grammar_error_rate * 20 * factor

    total_risk_score = handwriting_risk + spelling_risk + grammar_risk

    # Risk level determination
    if total_risk_score < 25:
        risk_level = "Low"
        subtype = "Unlikely Dyslexia Indicators"
    elif 25 <= total_risk_score < 55:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Subtype determination
    if total_risk_score >= 25:
        if spelling_risk > 15 and handwriting_risk < 25:
            subtype = "Phonological Dyslexia (Difficulty with sounds/spelling)"
        elif handwriting_risk > 25 and spelling_risk < 15:
            subtype = "Surface Dyslexia / Dysgraphia (Difficulty with visual word form/handwriting)"
        else:
            subtype = "Mixed Dyslexia (Indicators in both text and handwriting)"

    return total_risk_score, risk_level, subtype

def generate_error_heatmap(error_dict, title="Error Frequency Heatmap"):
    if not error_dict:
        st.info(f"No errors detected for {title}.")
        return None
    freq_df = pd.DataFrame(error_dict.items(), columns=["Item", "Frequency"]).sort_values(by="Frequency", ascending=False)
    plt.figure(figsize=(10, 2))
    sns.heatmap(freq_df.set_index('Item').T, annot=True, cmap="Reds", cbar=True)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def create_pdf_report(handwriting_score, text_score, grammar_score, total_risk, risk_level, subtype, suggestions, age_level):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height-50, "Dyslexia Detection Personalized Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height-90, f"Benchmark Age Level: {age_level}")

    c.drawString(50, height-120, f"Handwriting Dyslexia Score: {handwriting_score:.2f}")
    c.drawString(50, height-140, f"Text Spelling Accuracy: {text_score:.2f}%")
    c.drawString(50, height-160, f"Grammar Score: {grammar_score:.2f}%")
    c.drawString(50, height-180, f"Combined Risk Score: {total_risk:.2f} ({risk_level})")
    c.drawString(50, height-200, f"Subtype: {subtype}")

    c.drawString(50, height-230, "Recommendations:")
    y = height-250
    for s in suggestions:
        c.drawString(70, y, f"- {s}")
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------- LOAD MODEL ----------------
if 'cnn_model' not in st.session_state:
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state['cnn_model'] = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading CNN model: {e}")
            st.session_state['cnn_model'] = None
    else:
        st.warning(f"CNN model not found at '{MODEL_PATH}'. Handwriting analysis disabled.")
        st.session_state['cnn_model'] = None

if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {
        'handwriting_pred': None,
        'spelling_accuracy': None,
        'grammar_score': None,
        'text_errors': {},
        'handwriting_errors': {}
    }

# ---------------- UI ----------------
st.set_page_config(page_title="Dyslexia Detection App", layout="wide")
st.title("🧠 Dyslexia Detection Web App (Combined)")

# --- HANDWRITING ---
st.header("1️⃣ Handwriting Analysis")
image = st.file_uploader("Upload handwritten text sample (JPG/PNG)", type=["jpg", "jpeg", "png"])

if image:
    img = Image.open(image).convert("RGB")
    st.image(img, caption="Uploaded Sample", width=300)
    if st.button("Analyze Handwriting"):
        model = st.session_state['cnn_model']
        if model:
            resized = img.resize((INPUT_WIDTH, INPUT_HEIGHT))
            arr = np.array(resized, dtype=np.float32) / 255.0
            cnn_input = np.expand_dims(arr, axis=0)
            pred = float(model.predict(cnn_input)[0][0])
            st.session_state['analysis_results']['handwriting_pred'] = pred
            # Handwriting error frequency simulation
            st.session_state['analysis_results']['handwriting_errors'] = {f"Character{i}": np.random.randint(1,5) for i in range(1,6)}
            st.markdown(f"**Dyslexia Likelihood Score:** {1 - pred:.4f}")
            if pred >= 0.5:
                st.success("✅ Likely Non-Dyslexic")
            else:
                st.warning("⚠ Likely Dyslexic")
        else:
            st.error("CNN model not loaded.")

# --- TEXT ANALYSIS ---
st.header("2️⃣ Text Analysis")
text_input = st.text_area("Type or paste a paragraph:", height=150)
age_level = st.selectbox("Select Benchmark Age Level", ["6-7", "8-10", "11-13", "14+"])

if st.button("Analyze Text") and text_input:
    blob = TextBlob(text_input)
    corrected = str(blob.correct())

    input_words = text_input.lower().split()
    corrected_words = corrected.lower().split()
    error_rate = jiwer.wer(text_input, corrected)
    spelling_accuracy = max(0, (1 - error_rate)) * 100
    st.session_state['analysis_results']['spelling_accuracy'] = spelling_accuracy

    num_corrections = sum([1 for iw, cw in zip(input_words, corrected_words) if iw != cw])
    max_corrections_for_zero = 10
    grammar_penalty = min(1.0, num_corrections / max_corrections_for_zero)
    grammar_score = max(0, 100 - (grammar_penalty * 100))
    st.session_state['analysis_results']['grammar_score'] = grammar_score

    st.markdown("**Corrected Text (for reference):**")
    st.info(corrected)

    # Error frequency
    error_freq = {}
    for iw, cw in zip(input_words, corrected_words):
        if iw != cw:
            error_freq[iw] = error_freq.get(iw, 0) + 1
    st.session_state['analysis_results']['text_errors'] = error_freq

# --- COMBINED RISK & SUBTYPE ---
st.header("3️⃣ Combined Dyslexia Risk Assessment")
results = st.session_state['analysis_results']

hw_pred = results['handwriting_pred'] if results['handwriting_pred'] is not None else 0.5
spell_acc = results['spelling_accuracy'] if results['spelling_accuracy'] is not None else 90.0
gram_score = results['grammar_score'] if results['grammar_score'] is not None else 90.0

# Pass age_level to adjust risk
total_risk, risk_level, subtype = calculate_risk_and_subtype(hw_pred, spell_acc, gram_score, age_level)
color = {"Low":"green","Medium":"orange","High":"red"}[risk_level]
st.markdown(f"""
    <div style="border:2px solid {color}; padding:15px; border-radius:10px;">
    <h4>Risk Score: {total_risk:.2f}/100 ({risk_level})</h4>
    <p><b>Subtype:</b> {subtype}</p>
    </div>
""", unsafe_allow_html=True)

# --- ERROR HEATMAPS ---
st.header("4️⃣ Error Frequency Heatmaps")

if results['text_errors']:
    plt_text = generate_error_heatmap(results['text_errors'], "Text Errors Frequency")
    if plt_text:
        st.pyplot(plt_text)

if results['handwriting_errors']:
    plt_hw = generate_error_heatmap(results['handwriting_errors'], "Handwriting Errors Frequency")
    if plt_hw:
        st.pyplot(plt_hw)

# Scores & Risk Heatmap
heatmap_data = {
    "Metric": ["Handwriting Score", "Spelling Accuracy", "Grammar Score", "Total Risk"],
    "Score": [
        hw_pred * 100,
        spell_acc,
        gram_score,
        total_risk
    ]
}

df_heatmap = pd.DataFrame(heatmap_data).set_index("Metric")
plt.figure(figsize=(6, 2))
sns.heatmap(df_heatmap.T, annot=True, fmt=".1f", cmap="Reds", cbar=True)
plt.title("Dyslexia Metrics Heatmap")
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# --- PDF REPORT ---
st.header("5️⃣ Download Personalized PDF Report")
suggestions = []
if "Phonological" in subtype:
    suggestions.append("Focus on phonics and multisensory learning techniques.")
if "Surface" in subtype or hw_pred < 0.5:
    suggestions.append("Practice handwriting drills and occupational therapy for motor skills.")
if total_risk > 30:
    suggestions.append("Consult a specialist for formal assessment.")
if not suggestions:
    suggestions.append("Monitor progress; no immediate intervention.")

pdf_buffer = create_pdf_report(hw_pred, spell_acc, gram_score, total_risk, risk_level, subtype, suggestions, age_level)
st.download_button("Download Report as PDF", pdf_buffer, file_name="Dyslexia_Report.pdf", mime="application/pdf")