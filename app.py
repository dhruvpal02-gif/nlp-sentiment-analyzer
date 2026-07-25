import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Ultimate 4-Class Sentiment AI", page_icon="🧠", layout="centered")

# 2. Model Loading (Pointing to the 4-class folder)
@st.cache_resource
def load_model():
    model_path = "Aapka-HF-Username/ultimate-sentiment-ai"
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

st.title("🧠 Ultimate 4-Class Sentiment Analyzer")
st.markdown("Test our upgraded AI that perfectly understands **Negative, Factual (Neutral), Positive, Mixed, and Sarcastic** sentences!")

# 3. Load the AI Brain
with st.spinner("Waking up the 4-Class AI Brain... Please wait..."):
    sentiment_ai = load_model()

# 4. User Input Box
user_input = st.text_area("Type your review or sentence here:", height=150, placeholder="Example: Great job! Another product that doesn't even turn on.")

# 5. Analyze Button
if st.button("Analyze Sentiment 🚀"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing..."):
            # Predict Sentiment using AI
            result = sentiment_ai(user_input)[0]
            label = result['label']
            score = result['score']
            
            text_lower = user_input.lower()
            
            # --- THE HYBRID ENSEMBLE LOGIC (Overriding AI mistakes) ---
            
            # Catching Sarcasm & Missed Fatal Flaws with an expanded list
            fatal_phrases = [
                'ignored', 
                'does not turn on', 
                "doesn't turn on", 
                "doesn't even turn on", 
                "won't turn on",
                'worst', 
                'pathetic', 
                'waste of money', 
                'never buy',
                'crashes',
                'broken',
                'defective'
            ]
            mixed_phrases = ['but', 'however', 'although', 'though']
            positive_words = ['great', 'wonderful', 'amazing', 'brilliant', 'good', 'beautiful', 'awesome', 'excellent']
            
            is_fatal = any(phrase in text_lower for phrase in fatal_phrases)
            has_mixed_words = any(word in text_lower.split() for word in mixed_phrases)
            has_positive_words = any(word in text_lower.split() for word in positive_words)
            
            # Logic 1: Sarcasm override (Highly positive words BUT contains fatal flaws)
            if is_fatal and has_positive_words:
                label = "LABEL_0_SARCASM" 
                
            # Logic 2: Mixed override (If AI missed a clear contrast, like IMDb issue)
            elif has_mixed_words and has_positive_words and label in ["LABEL_0", "LABEL_2"]:
                label = "LABEL_3"
                
            # Logic 3: Hard Negative override (If it's just a fatal flaw without sarcasm)
            elif is_fatal:
                label = "LABEL_0"
            
            # --- FINAL UI RENDERING ---
            if label == "LABEL_0":
                sentiment = "Negative 😠"
                color = "red"
            elif label == "LABEL_0_SARCASM":
                sentiment = "Negative (Sarcasm Detected) 😒"
                color = "darkred"
            elif label == "LABEL_1":
                sentiment = "True Neutral / Factual 😐"
                color = "gray"
            elif label == "LABEL_2":
                sentiment = "Positive 🤩"
                color = "green"
            elif label == "LABEL_3":
                sentiment = "Mixed / Contrastive 🤔"
                color = "orange"
            
            # Display Results
            st.markdown(f"### Result: <span style='color:{color};'>{sentiment}</span>", unsafe_allow_html=True)
            
            # Smart notification if the Hybrid Logic intervened
            if label == "LABEL_0_SARCASM" or (has_mixed_words and result['label'] != "LABEL_3") or (is_fatal and result['label'] != "LABEL_0"):
                st.info(f"AI Base Score: {score * 100:.2f}% | Adjusted by Hybrid Heuristic Pipeline ⚙️")
            else:
                st.info(f"AI Confidence Score: {score * 100:.2f}%")
            
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hugging Face Transformers")