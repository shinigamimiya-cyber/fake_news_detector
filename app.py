import streamlit as st
import pickle
import re

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Load model@st.cache_resource
def load_model():
    import pickle
    
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, vectorizer = load_model()
# 🔍 DEBUG (ADD THIS RIGHT HERE)
st.write("Vectorizer type:", type(vectorizer))
st.write("Is fitted:", hasattr(vectorizer, "vocabulary_"))
# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered news authenticity checker</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.write("This app detects whether a news article is Fake or Real using Machine Learning.")

# Input
user_input = st.text_area("Enter News Text:", height=200)

# Button
if st.button("Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform(str[cleaned])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        st.markdown("---")

        if prediction == 0:
            st.error("❌ Fake News Detected")
        else:
            st.success("✅ Real News Detected")

        st.subheader("Confidence Score")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: {round(confidence * 100, 2)}%")
