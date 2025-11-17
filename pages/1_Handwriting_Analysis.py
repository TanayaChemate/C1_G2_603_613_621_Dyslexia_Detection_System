import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "model_cnn.h5"
INPUT_WIDTH, INPUT_HEIGHT = 128, 128

st.title("🖊️ Handwriting Analysis")

# Initialize session state for results
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {
        'handwriting_pred': None,
        'spelling_accuracy': None,
        'grammar_score': None
    }

# Load CNN model once
if 'cnn_model' not in st.session_state:
    try:
        st.session_state['cnn_model'] = tf.keras.models.load_model(MODEL_PATH)
        st.success("CNN model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        st.session_state['cnn_model'] = None

# File uploader
image = st.file_uploader("Upload handwritten sample:", type=["jpg","jpeg","png"])
if image:
    img = Image.open(image).convert("RGB")
    st.image(img, caption="Uploaded Sample", width=300)

    # Analyze button
    if st.button("Analyze Handwriting"):
        model = st.session_state['cnn_model']
        if model:
            resized = img.resize((INPUT_WIDTH, INPUT_HEIGHT))
            arr = np.array(resized, dtype=np.float32)/255.0
            cnn_input = np.expand_dims(arr, axis=0)

            # Prediction
            try:
                pred = float(model.predict(cnn_input)[0][0])
                st.session_state['analysis_results']['handwriting_pred'] = pred
                st.markdown(f"**Dyslexia Likelihood Score:** {1 - pred:.4f}")
                if pred >= 0.5:
                    st.success("✅ Likely Non-Dyslexic")
                else:
                    st.warning("⚠ Likely Dyslexic")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("CNN model not loaded.")
