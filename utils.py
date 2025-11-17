import numpy as np
import jiwer
from textblob import TextBlob
from fpdf import FPDF
from io import BytesIO

# ------------------ Existing Functions ------------------
def calculate_risk_and_subtype(handwriting_pred, spelling_accuracy, grammar_score):
    handwriting_risk = (1.0 - handwriting_pred) * 50
    spelling_error_rate = 1.0 - (spelling_accuracy / 100.0)
    spelling_risk = spelling_error_rate * 30
    grammar_error_rate = 1.0 - (grammar_score / 100.0)
    grammar_risk = grammar_error_rate * 20

    total_risk_score = handwriting_risk + spelling_risk + grammar_risk

    if total_risk_score < 25:
        risk_level = "Low"
        subtype = "Unlikely Dyslexia Indicators"
    elif 25 <= total_risk_score < 55:
        risk_level = "Medium"
    else:
        risk_level = "High"

    if total_risk_score >= 25:
        if spelling_risk > 15 and handwriting_risk < 25:
            subtype = "Phonological Dyslexia (Difficulty with sounds/spelling)"
        elif handwriting_risk > 25 and spelling_risk < 15:
            subtype = "Surface Dyslexia / Dysgraphia (Difficulty with visual word form/handwriting)"
        else:
            subtype = "Mixed Dyslexia (Indicators in both text and handwriting)"

    return total_risk_score, risk_level, subtype


def analyze_text(text_input):
    blob = TextBlob(text_input)
    corrected = str(blob.correct())

    error_rate = jiwer.wer(text_input, corrected)
    spelling_accuracy = max(0, (1 - error_rate)) * 100

    input_words = text_input.lower().split()
    corrected_words = corrected.lower().split()
    num_corrections = sum([1 for iw, cw in zip(input_words, corrected_words) if iw != cw])

    max_corrections_for_zero = 10
    grammar_penalty = min(1.0, num_corrections / max_corrections_for_zero)
    grammar_score = max(0, 100 - (grammar_penalty * 100))

    # Create error list for heatmap / frequency graph
    error_list = []
    for iw, cw in zip(input_words, corrected_words):
        if iw != cw:
            error_list.append((iw, cw))

    return corrected, spelling_accuracy, grammar_score, error_list


# ------------------ NEW FUNCTIONS ------------------

def generate_pdf_report(handwriting_pred, spelling_accuracy, grammar_score, total_risk, risk_level, subtype, age_benchmark, error_list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Personalized Dyslexia Assessment Report", ln=True, align='C')

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Handwriting Likelihood Score: {handwriting_pred:.2f}", ln=True)
    pdf.cell(0, 8, f"Spelling Accuracy: {spelling_accuracy:.2f}%", ln=True)
    pdf.cell(0, 8, f"Grammar Score: {grammar_score:.2f}%", ln=True)
    pdf.cell(0, 8, f"Total Risk Score: {total_risk:.2f} ({risk_level})", ln=True)
    pdf.cell(0, 8, f"Estimated Dyslexia Subtype: {subtype}", ln=True)
    pdf.cell(0, 8, f"Benchmark (Age-Level Expected Performance): {age_benchmark}", ln=True)

    pdf.ln(10)
    if error_list:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Detected Errors:", ln=True)
        pdf.set_font("Arial", '', 12)
        for orig, corr in error_list:
            pdf.cell(0, 8, f"{orig} → {corr}", ln=True)
    else:
        pdf.cell(0, 8, "No spelling/grammar errors detected.", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 8, "This is a preliminary, automated assessment. For accurate diagnosis, consult a specialist.")

    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


def benchmark_score(age=None):
    """
    Returns a benchmark description for the user based on age.
    Example: can be customized for grade/age level
    """
    if age is None:
        return "10-12 yrs (default benchmark)"
    if age < 7:
        return "7-9 yrs"
    elif age < 10:
        return "10-12 yrs"
    elif age < 13:
        return "13-15 yrs"
    else:
        return "16+ yrs"
