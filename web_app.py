import streamlit as st
from transformers import pipeline

# ==========================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(page_title="Ultimate Sentiment AI", page_icon="🧠", layout="centered")

# Custom CSS for modern look
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
    # ⚠️ YAHAN APNA HUGGING FACE USERNAME DAALEIN (e.g., "dhruvpal02/ultimate-sentiment-ai")
    MODEL_NAME = "aapka_username/yahan_model_ka_naam_daalein" 
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

# ==========================================
# 3. APP HEADER & USER INPUT
# ==========================================
st.title("🧠 Ultimate AI Sentiment Analyzer")
st.markdown("Enter any review, comment, or paragraph, and my **Custom 4-Class Deep Learning Model** will analyze its true emotion!")

with st.spinner("Waking up the AI Brain... 🤖"):
    analyzer = load_model()

user_text = st.text_area("✍️ Type your text here:", height=150, placeholder="Example: The camera quality is amazing, but the battery drains too fast...")

# ==========================================
# 4. PREDICTION LOGIC & RESULT DISPLAY
# ==========================================
if st.button("🔮 Analyze Sentiment"):
    if user_text.strip() == "":
        st.error("⚠️ Please enter some text first!")
    else:
        with st.spinner("🧠 Analyzing deep semantics..."):
            result = analyzer(user_text)[0]
            label = result['label']
            score = result['score'] * 100  # Convert to percentage
            
        st.markdown("---")
        st.markdown("### 📊 AI Prediction Result")
        
        # Displaying result in a clean 2-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            if label == "Positive":
                st.success("### 🥳 Positive")
            elif label == "Negative":
                st.error("### 😡 Negative")
            elif label == "Neutral":
                st.info("### 😐 Neutral / Factual")
            elif label == "Mixed":
                st.warning("### 🤔 Mixed Sentiment")
                
        with col2:
            st.metric(label="AI Confidence Score", value=f"{score:.2f}%")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown('<div class="footer">Developed with ❤️ by Dhruv</div>', unsafe_allow_html=True)
