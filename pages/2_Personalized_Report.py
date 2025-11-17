import streamlit as st
import pandas as pd
from utils import generate_pdf_report, benchmark_score


st.title("📝 Personalized Report")

results = st.session_state['analysis_results']

if not any(results.values()):
    st.info("Please complete handwriting and text analysis first.")
else:
    st.subheader("Summary")
    summary = {
        "Handwriting Likelihood Score": results.get('handwriting_pred'),
        "Spelling Accuracy": results.get('spelling_accuracy'),
        "Grammar Score": results.get('grammar_score'),
        "Number of Text Errors": len(results.get('error_list', []))
    }

    for k, v in summary.items():
        if v is not None:
            st.write(f"**{k}:** {v:.2f}" if isinstance(v, float) else f"**{k}:** {v}")

    st.subheader("Error Frequency Table")
    error_list = results.get('error_list', [])
    if error_list:
        df = pd.DataFrame(error_list, columns=['Original', 'Corrected'])
        st.dataframe(df)
    else:
        st.write("No spelling/grammar errors detected.")
