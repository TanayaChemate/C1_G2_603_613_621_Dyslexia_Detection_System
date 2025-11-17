# Dyslexia Detection Web App

A Python-based web application to detect dyslexia traits using handwriting samples and text input analysis, 
with future support for audio-based analysis.

The frontend is built with Streamlit, and machine learning models are used for handwriting analysis.

---

## Features

1. Handwriting Analysis
   - Upload a handwriting image.
   - Predict dyslexic or non-dyslexic handwriting.
   - Shows model confidence score.

2. Text Input Analysis
   - Type or paste text.
   - Check spelling and grammar corrections.
   - Simulate phonics/pronunciation accuracy.
   - Calculate dyslexia-related traits:
       - Spelling errors
       - Grammatical errors
       - Phonetic accuracy (%)
       - Percentage of corrections

3. Future Audio Analysis
   - Upload audio recordings to extract speech features.
   - Analyze pronunciation, syllable, and phoneme accuracy.

---

## How It Works

### Handwriting Analysis
1. Resize uploaded image to CNN input size (128x128).
2. Predict dyslexia likelihood with pre-trained CNN.
3. Display prediction score with warnings.

### Text Input Analysis
1. User enters text.
2. Check spelling/grammar with TextBlob.
3. Simulate phonics/accuracy using Word Error Rate (WER).
4. Compute dyslexia-related traits (spelling, grammar, phonics).
5. Display outputs to user.

### Audio Analysis (Future)
1. Upload audio.
2. Convert speech to text.
3. Calculate WER, syllable, phoneme error rate.
4. Display additional dyslexia traits.

---

## Running the Web App

1. Clone repo:
   git clone <repository_url>
   cd Dyslexia_Detection-main

2. Create virtual environment:
   python -m venv venv

3. Activate environment:
   Windows: .\venv\Scripts\Activate
   Mac/Linux: source venv/bin/activate

4. Install packages:
   pip install -r requirements.txt

5. Run app:
   streamlit run app.py

---

## Folder Structure

### data
- dyslexic/: Handwriting samples of dyslexic children.
- non_dyslexic/: Handwriting samples of non-dyslexic children.
- data.csv: Extracted features:
    - Spelling accuracy
    - Grammatical accuracy
    - % of corrections
    - Phonetic accuracy
    - Presence of dyslexia
- school_symptoms.txt: Documented dyslexia symptoms

### images
- Visualizations of extracted features:
    - percentage_of_corrections.jpg
    - percentage_of_phonetic_accuraccy.jpg
    - spelling_accuracy.jpg

### model_training
- Scripts for training and evaluating ML models.

### app.py
- Main Streamlit app.
- Upload handwriting or text to analyze dyslexia traits.

---

## Notes
- Data may contain biases; not medically verified.
- Audio features to be added in future.
- Handwriting model may have reduced accuracy for unseen samples.

---

## License
Educational purposes only. Use responsibly.

## Team Members
UCE2023603 - Ahilya Kerle || 
UCE2023613 - Tanaya Cheamte || 
UCE2023621 - Ishwari Gojare

