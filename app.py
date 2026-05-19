import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Language Detector",
    page_icon="🌍",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}

.title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
    margin-top: 20px;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 35px;
}

.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 18px !important;
    border: 2px solid #38bdf8 !important;
    font-size: 18px !important;
    padding: 15px !important;
}

.stButton button {
    width: 100%;
    background: linear-gradient(to right, #0ea5e9, #2563eb);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
    padding: 14px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #2563eb, #0ea5e9);
}

.result-box {
    background: #1e293b;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
    border: 2px solid #38bdf8;
    box-shadow: 0px 0px 20px rgba(56,189,248,0.4);
}

.result-text {
    font-size: 32px;
    color: #38bdf8;
    font-weight: bold;
}

.confidence-text {
    font-size: 22px;
    color: white;
    margin-top: 12px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 60px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model("language_model.keras")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

encoder = pickle.load(open("encoder.pkl", "rb"))

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🌍 AI Language Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Powered Multilingual Text Classifier</div>',
    unsafe_allow_html=True
)

# =========================================================
# INPUT AREA
# =========================================================

text = st.text_area(
    "✍️ Enter Your Text",
    height=200,
    placeholder="Type text here... Example: Bonjour mon ami"
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

max_len = 40

def predict_language(sentence):

    sequence = tokenizer.texts_to_sequences([sentence])

    padded = pad_sequences(sequence, maxlen=max_len)

    prediction = model.predict(padded)

    predicted_index = np.argmax(prediction)

    language = encoder.inverse_transform([predicted_index])[0]

    confidence = np.max(prediction) * 100

    return language, confidence

# =========================================================
# BUTTON
# =========================================================

if st.button("🚀 Detect Language"):

    if text.strip() == "":

        st.warning("⚠️ Please enter some text.")

    else:

        language, confidence = predict_language(text)

        # FLAGS
        flags = {
            "English": "🇺🇸",
            "French": "🇫🇷",
            "Spanish": "🇪🇸",
            "Hindi": "🇮🇳",
            "Italian": "🇮🇹",
            "Portuguese": "🇵🇹",
            "German": "🇩🇪",
            "Dutch": "🇳🇱",
            "Russian": "🇷🇺",
            "Turkish": "🇹🇷",
            "Arabic": "🇸🇦",
            "Swedish": "🇸🇪"
        }

        flag = flags.get(language, "🌐")

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-text">
                    {flag} {language}
                </div>

                <div class="confidence-text">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 About")

st.sidebar.info(
    """
This project uses:

✅ Deep Learning  
✅ LSTM Neural Networks  
✅ NLP Tokenization  
✅ TensorFlow  
✅ Streamlit Deployment  

### Features
- Real-time prediction
- Multilingual detection
- Attractive UI
- Fast inference
"""
)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Made with ❤️ using Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)
