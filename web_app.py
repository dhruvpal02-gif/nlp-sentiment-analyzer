import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import urllib.request
import json

# 1. Page Configuration
st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="🧠", layout="wide")

# 2. Load the trained AI Brain
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# 3. Free Translation Function using Google API (No library installation required for Cloud)
def translate_to_english(text):
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        translated_text = "".join([sentence[0] for sentence in data[0] if sentence[0]])
        detected_lang = data[2]
        return translated_text, detected_lang
    except Exception:
        return text, "en" # Fallback to original text if translation fails

# 4. Initialize Prediction History
if 'history' not in st.session_state:
    st.session_state.history = []

# --- UI LAYOUT ---
st.title("🧠 Universal Multilingual Sentiment Analyzer")
st.write("Supports Hindi, Hinglish, Marathi, English, and 100+ languages!")

tab1, tab2 = st.tabs(["🔍 Analyze Review", "📊 Model Performance"])

# --- TAB 1: MAIN ANALYZER ---
with tab1:
    review_text = st.text_area("Paste Review Here (Hindi, Hinglish, English, etc.):", height=150, placeholder="E.g., Ye phone bahut mast hai! OR This product is terrible...")

    if st.button("Analyze Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter some text first!")
        else:
            # Step A: Translate to English dynamically
            with st.spinner("Analyzing language context..."):
                english_text, lang_code = translate_to_english(review_text)

            # Step B: Vectorization & Prediction
            text_vector = vectorizer.transform([english_text])
            prediction = model.predict(text_vector)[0].capitalize()
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities) * 100

            # Extract Top Keywords
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = text_vector.toarray()[0]
            top_indices = tfidf_scores.argsort()[-5:][::-1]
            top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

            # Save to History
            st.session_state.history.append({
                "review": review_text,
                "sentiment": prediction,
                "confidence": confidence
            })

            st.write("---")
            
            # Show translation info if it wasn't originally English
            if lang_code != "en":
                st.info(f"🌐 **Detected Language:** {lang_code.upper()} | **Internal Translation:** \"{english_text}\"")

            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == "Positive":
                    st.success(f"### **Verdict: Positive** 🎉\n**Confidence Score:** {confidence:.2f}%")
                elif prediction == "Negative":
                    st.error(f"### **Verdict: Negative** ⚠️\n**Confidence Score:** {confidence:.2f}%")
                else:
                    st.info(f"### **Verdict: Neutral** 😐\n**Confidence Score:** {confidence:.2f}%")
                
                st.write("**🔑 Top Keywords driving this prediction (English Context):**")
                if top_keywords:
                    for word in top_keywords:
                        st.write(f"- {word}")
                else:
                    st.write("- Generic Context")

            with col2:
                try:
                    wc = WordCloud(width=400, height=200, background_color='black', colormap='Blues').generate(english_text)
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except ValueError:
                    st.write("Not enough unique words for a Word Cloud.")

# --- TAB 2: MODEL EVALUATION ---
with tab2:
    st.header("📊 Model Evaluation & Metrics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Algorithm:** Logistic Regression")
        st.info("**Architecture:** Multi-Domain API Pipeline")
        st.info("**Data Domains:** Movies (IMDB) + Products (Amazon) + Restaurants (Yelp)")
    with col_b:
        st.success("**Core Accuracy:** ~88.2%")
        st.success("**Feature:** Cross-Language Translation Wrapper Enabled")

# --- SIDEBAR: HISTORY ---
st.sidebar.title("🕒 Prediction History")
for item in reversed(st.session_state.history):
    color = "🟢" if item['sentiment'] == "Positive" else "🔴" if item['sentiment'] == "Negative" else "⚪"
    st.sidebar.write(f"{color} **{item['sentiment']}** ({item['confidence']:.1f}%)")
    st.sidebar.caption(item['review'][:40] + "..." if len(item['review']) > 40 else item['review'])
    st.sidebar.write("---")
