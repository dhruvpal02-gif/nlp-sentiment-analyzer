import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Configuration (Browser tab settings)
st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="🧠", layout="centered")

# 2. Load the trained AI Brain (Using cache taaki app baar-baar reload na ho)
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# 3. Web UI Elements
st.title("🧠 Natural Language Processing")
st.subheader("Customer Review Sentiment Analyzer")
st.write("Trained on 10,000+ real-world customer reviews. Enter any text below to instantly check its sentiment context.")

# Input Box (Web based Text Area)
review_text = st.text_area("Paste Customer Review Here:", height=150, placeholder="Type or paste a product review (e.g., 'The quality is excellent but shipping was delayed...')")

# Analyze Button
if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter some text first!")
    else:
        # Vectorization & Prediction
        text_vector = vectorizer.transform([review_text])
        prediction = model.predict(text_vector)[0].capitalize()
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities) * 100

        # 4. Display Results with beautiful UI boxes
        st.write("---")
        if prediction == "Positive":
            st.success(f"### **Verdict: Positive** 🎉\n**Confidence Score:** {confidence:.2f}%")
        elif prediction == "Negative":
            st.error(f"### **Verdict: Negative** ⚠️\n**Confidence Score:** {confidence:.2f}%")
        else:
            st.info(f"### **Verdict: Neutral** 😐\n**Confidence Score:** {confidence:.2f}%")

        # 5. Generate and Display Word Cloud
        st.write("---")
        st.subheader("📊 Words driving the prediction:")
        
        try:
            wc = WordCloud(width=800, height=400, background_color='black', colormap='Blues').generate(review_text)
            
            # Matplotlib ka use karke web page par image render karna
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off") # Borders/Axis hide karne ke liye
            
            st.pyplot(fig) # Streamlit dedicated image component
        except ValueError:
            st.write("Not enough unique words to generate a word cloud.")