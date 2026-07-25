import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# 1. Page Configuration
st.set_page_config(page_title="Ultimate 4-Class Sentiment AI", page_icon="🧠", layout="centered")

# 2. Model Loading (Pointing to your Hugging Face model)
@st.cache_resource
def load_model():
    model_path = "dhruvpal02/ultimate-sentiment-ai"
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

st.title("🌍 Multilingual 4-Class Sentiment Analyzer")
st.markdown("Test our upgraded AI that understands **Hindi, Marathi, Gujarati, English** & 100+ languages!")

# 3. Load the AI Brain
with st.spinner("Waking up the Multilingual AI Brain... Please wait..."):
    sentiment_ai = load_model()

# 4. User Input Box
user_input = st.text_area("Type your review here (in any language):", height=150, placeholder="Example: Product bohot achha hai, par customer service bekar hai.")

# 5. Analyze Button
if st.button("Analyze Sentiment 🚀"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Translating & Analyzing..."):
            
            # --- NEW: Translation Pipeline ---
            # Automatically detect language and translate to English
            translated_text = GoogleTranslator(source='auto', target='en').translate(user_input)
            
            # Show the translation so users know it worked
            if user_input.strip().lower() != translated_text.lower():
                st.info(f"🌐 Translated to English: {translated_text}")
            
            # --- AI Prediction (Using the translated text) ---
            result = sentiment_ai(translated_text)[0]
            label = result['label']
            score = result['score']
            
            text_lower = translated_text.lower()
            
            # --- THE HYBRID ENSEMBLE LOGIC ---
            fatal_phrases = [
                'ignored', 'does not turn on', "doesn't turn on", "doesn't even turn on", 
                "won't turn on", 'worst', 'pathetic', 'waste of money', 'never buy',
                'crashes', 'broken', 'defective', 'useless'
            ]
            mixed_phrases = ['but', 'however', 'although', 'though']
            positive_words = ['great', 'wonderful', 'amazing', 'brilliant', 'good', 'beautiful', 'awesome', 'excellent']
            
            is_fatal = any(phrase in text_lower for phrase in fatal_phrases)
            has_mixed_words = any(word in text_lower.split() for word in mixed_phrases)
            has_positive_words = any(word in text_lower.split() for word in positive_words)
            
            # Logic overrides
            if is_fatal and has_positive_words:
                label = "LABEL_0_SARCASM" 
            elif has_mixed_words and has_positive_words and label in ["LABEL_0", "LABEL_2"]:
                label = "LABEL_3"
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
            
            if label == "LABEL_0_SARCASM" or (has_mixed_words and result['label'] != "LABEL_3") or (is_fatal and result['label'] != "LABEL_0"):
                st.info(f"AI Base Score: {score * 100:.2f}% | Adjusted by Hybrid Heuristic Pipeline ⚙️")
            else:
                st.info(f"AI Confidence Score: {score * 100:.2f}%")
            
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Hugging Face & Deep Translator")
