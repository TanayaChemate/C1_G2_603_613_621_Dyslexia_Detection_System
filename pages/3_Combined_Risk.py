import streamlit as st
from utils import calculate_risk_and_subtype

st.title("📊 Combined Dyslexia Risk Assessment")

results = st.session_state['analysis_results']

handwriting_pred = results.get('handwriting_pred', None)
spelling_accuracy = results.get('spelling_accuracy', None)
grammar_score = results.get('grammar_score', None)

if handwriting_pred is not None or spelling_accuracy is not None:
    hw = handwriting_pred if handwriting_pred is not None else 0.5
    spell = spelling_accuracy if spelling_accuracy is not None else 90.0
    gram = grammar_score if grammar_score is not None else 90.0

    total_risk, risk_level, subtype = calculate_risk_and_subtype(hw, spell, gram)

    if risk_level == "High":
        st.error(f"🔴 High Risk: {total_risk:.2f}/100")
        color = "red"
    elif risk_level == "Medium":
        st.warning(f"🟠 Medium Risk: {total_risk:.2f}/100")
        color = "orange"
    else:
        st.success(f"🟢 Low Risk: {total_risk:.2f}/100")
        color = "green"

    st.markdown(f"""
    <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; margin-top: 15px;">
        <h4>Dyslexia Subtype Classification:</h4>
        <p style="font-size: 1.1em; font-weight: bold;">{subtype}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Analyze handwriting or text first to see combined risk.")
