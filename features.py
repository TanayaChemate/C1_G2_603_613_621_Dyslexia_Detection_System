# features.py
from PIL import Image
import pytesseract
from textblob import TextBlob
from abydos.phonetic import Soundex

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Levenshtein function
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# OCR
def image_to_text(path):
    text = pytesseract.image_to_string(Image.open(path))
    return text.strip()

# Spelling accuracy
def spelling_accuracy(extracted_text):
    spell_corrected = TextBlob(extracted_text).correct()
    return ((len(extracted_text) - levenshtein(extracted_text, str(spell_corrected))) / (len(extracted_text) + 1)) * 100

# Phonetic accuracy
def percentage_of_phonetic_accuracy(extracted_text: str):
    soundex = Soundex()
    spell_corrected = TextBlob(extracted_text).correct()
    extracted_text_list = extracted_text.split(" ")
    extracted_soundex_string = " ".join([soundex.encode(w) for w in extracted_text_list])
    spell_corrected_soundex_string = " ".join([soundex.encode(w) for w in str(spell_corrected).split()])
    soundex_score = (len(extracted_soundex_string) - levenshtein(extracted_soundex_string, spell_corrected_soundex_string)) / (len(extracted_soundex_string) + 1)
    return soundex_score * 100

# Additional features
def additional_features(extracted_text):
    words = extracted_text.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / (word_count + 1)
    unique_word_ratio = len(set(words)) / (word_count + 1)
    return [word_count, avg_word_length, unique_word_ratio]

# Feature array
def get_feature_array(path: str):
    extracted_text = image_to_text(path)
    base_features = [
        spelling_accuracy(extracted_text),
        percentage_of_phonetic_accuracy(extracted_text)
    ]
    extra_features = additional_features(extracted_text)
    return base_features + extra_features
