import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# ==========================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(page_title="Multilingual Sentiment AI", page_icon="🌍", layout="centered")

st.markdown("""
    <style>
    .stTextArea textarea {
        border-radius: 12px;
        border: 1.5px solid #ced4da;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 25px;
        width: 100%;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .translated-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #007BFF;
        font-style: italic;
        color: #495057;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 80px;
        color: #6c757d;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD AI MODEL (WITH CACHING FOR SPEED)
# ==========================================
@st.cache_resource
def load_model():
    MODEL_NAME = "dhruvpal02/ultimate-sentiment-ai" 
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

# ==========================================
# 3. APP HEADER & USER INPUT
# ==========================================
st.title("🌍 Multilingual AI Sentiment Analyzer")
st.markdown("Enter text in **Hindi, Marathi, Spanish, or ANY language!** My AI will auto-translate and analyze its true emotion.")

with st.spinner("Waking up the AI Brain... 🤖"):
    analyzer = load_model()

user_text = st.text_area("✍️ Type your text here (Any Language):", height=120, placeholder="Example: यह फोन बहुत अच्छा है लेकिन इसकी बैटरी जल्दी खत्म हो जाती है...")

# ==========================================
# 4. TRANSLATION & PREDICTION LOGIC
# ==========================================
if st.button("🔮 Analyze Sentiment"):
    if user_text.strip() == "":
        st.error("⚠️ Please enter some text first!")
    else:
        with st.spinner("🌍 Auto-Translating and Analyzing deep semantics..."):
            
            # 1. Translate to English
            try:
                english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
            except Exception as e:
                english_text = user_text # Fallback if translation fails
                st.warning("Translation failed, analyzing original text...")

            # 2. AI Prediction
            result = analyzer(english_text)[0]
            raw_label = result['label']
            score = result['score'] * 100  
            
            label_map = {
                "LABEL_0": "Negative", 
                "LABEL_1": "Neutral", 
                "LABEL_2": "Positive", 
                "LABEL_3": "Mixed"
            }
            final_label = label_map.get(raw_label, raw_label)
            
        st.markdown("---")
        
        # Show what the AI actually read
        if english_text.lower() != user_text.lower():
            st.markdown(f'<div class="translated-box"><b>Translated to English:</b> "{english_text}"</div>', unsafe_allow_html=True)
            
        st.markdown("### 📊 AI Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            if final_label == "Positive":
                st.success("### 🥳 Positive")
            elif final_label == "Negative":
                st.error("### 😡 Negative")
            elif final_label == "Neutral":
                st.info("### 😐 Neutral / Factual")
            elif final_label == "Mixed":
                st.warning("### 🤔 Mixed Sentiment")
            else:
                st.write(f"### {final_label}")
                
        with col2:
            st.metric(label="AI Confidence Score", value=f"{score:.2f}%")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown('<div class="footer">Developed with ❤️ by Dhruv</div>', unsafe_allow_html=True)
