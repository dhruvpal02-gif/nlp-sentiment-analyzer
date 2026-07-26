import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime
import random

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS 
# ==========================================
st.set_page_config(page_title="InsightAI Dashboard", page_icon="🟣", layout="wide")

st.markdown("""
    <style>
    /* Input box styling */
    .stTextInput input {
        border: 1px solid #ff4b4b !important;
        background-color: #1e1e24;
        color: white;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Texts and Headers */
    .dashboard-title { font-size: 2.5rem; font-weight: 700; margin-bottom: 0;}
    .dashboard-subtitle { color: #a0a0a5; font-size: 1.1rem; margin-bottom: 2rem;}
    .section-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 5px;}
    .section-subtitle { color: #a0a0a5; font-size: 0.9rem; margin-bottom: 15px;}
    
    /* Date Badge */
    .date-badge {
        background-color: #1e1e24; border: 1px solid #333;
        padding: 8px 15px; border-radius: 6px; float: right;
        color: #a0a0a5; font-size: 0.9rem; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD AI MODEL
# ==========================================
@st.cache_resource
def load_model():
    MODEL_NAME = "dhruvpal02/ultimate-sentiment-ai" 
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, top_k=None)

with st.spinner("Initializing Dashboard Engines..."):
    analyzer = load_model()

# ==========================================
# 3. HEADER SECTION
# ==========================================
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown('<p class="dashboard-title">Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Analyze text data to extract insights and understand sentiment patterns.</p>', unsafe_allow_html=True)
with col_header2:
    today_date = datetime.datetime.now().strftime("%b %d, %Y")
    st.markdown(f'<div class="date-badge">🗓️ Last 7 Days: {today_date}</div>', unsafe_allow_html=True)

# 💡 FIX: Added a default Hindi sentence so dashboard is NEVER empty on load
default_text = "यह प्रोडक्ट दिखने में बहुत अच्छा है, लेकिन इसकी बैटरी बहुत जल्दी खत्म हो जाती है।"
user_text = st.text_input("", value=default_text, placeholder="Paste customer review here to analyze instantly...")

# ==========================================
# 4. DASHBOARD PROCESSING & UI 
# ==========================================
if user_text:
    # --- TRANSLATION ---
    try:
        english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
    except:
        english_text = user_text
        
    # 💡 FIX: Beautiful Translation Box (Shows only if text was translated)
    if english_text.lower().strip() != user_text.lower().strip():
        st.markdown(f"""
        <div style="background-color: #2b2b36; border-left: 4px solid #007BFF; padding: 12px 20px; border-radius: 8px; margin-top: -15px; margin-bottom: 25px;">
            <span style="color: #a0a0a5; font-size: 13px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">🌍 Translated to English for Analysis:</span><br>
            <span style="color: #ffffff; font-size: 16px; font-weight: 500;">"{english_text}"</span>
        </div>
        """, unsafe_allow_html=True)

    # --- AI PREDICTION ---
    results = analyzer(english_text)[0]
    
    scores = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive", "LABEL_3": "Mixed"}
    
    for res in results:
        mapped_name = label_mapping.get(res['label'], res['label'])
        scores[mapped_name] = round(res['score'] * 100)
    
    top_sentiment = max(scores, key=scores.get)
    top_score = scores[top_sentiment]
    
    gauge_color = "#28a745" if top_sentiment == "Positive" else "#dc3545" if top_sentiment == "Negative" else "#fd7e14" if top_sentiment == "Neutral" else "#6f42c1"

    # --- ROW 1: METRICS & GAUGE CHART ---
    row1_col1, row1_col2 = st.columns([1.5, 1])
    
    with row1_col1:
        st.markdown('<p class="section-title">Overall Sentiment</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Overview of sentiment distribution across 4 classes</p>', unsafe_allow_html=True)
        
        # 💡 FIX: Inline CSS guarantees text visibility regardless of Dark Mode
        c1, c2, c3, c4 = st.columns(4)
        
        def draw_card(title, emoji, percentage, color):
            return f"""
            <div style="background-color: #ffffff; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <h4 style="margin: 0; font-size: 16px; font-weight: 900; color: {color};">{emoji} {title}</h4>
                <h1 style="margin: 10px 0 0 0; font-size: 32px; color: #000000; font-weight: 900;">{percentage}%</h1>
            </div>
            """
            
        with c1: st.markdown(draw_card("Positive", "🤩", scores["Positive"], "#28a745"), unsafe_allow_html=True)
        with c2: st.markdown(draw_card("Negative", "😡", scores["Negative"], "#dc3545"), unsafe_allow_html=True)
        with c3: st.markdown(draw_card("Neutral", "😐", scores["Neutral"], "#fd7e14"), unsafe_allow_html=True)
        with c4: st.markdown(draw_card("Mixed", "🤔", scores["Mixed"], "#6f42c1"), unsafe_allow_html=True)
            
    with row1_col2:
        st.markdown('<p class="section-title">Confidence Score</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-subtitle">Overall analysis confidence ({top_sentiment})</p>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = top_score,
            number = {'suffix': "%", 'font': {'size': 50, 'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'visible': False},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "#1e1e24",
                'borderwidth': 0,
            }
        ))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=200)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- ROW 2: LINE CHART & WORD CLOUD ---
    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns([1.5, 1])
    
    with row2_col1:
        st.markdown('<p class="section-title">Sentiment Over Time</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Simulated trend analysis based on current input</p>', unsafe_allow_html=True)
        
        x_vals = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Current']
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=x_vals, y=[random.randint(20,80), random.randint(20,80), random.randint(20,80), random.randint(20,80), scores["Positive"]], name="Pos", line=dict(color="#28a745")))
        fig_line.add_trace(go.Scatter(x=x_vals, y=[random.randint(10,50), random.randint(10,50), random.randint(10,50), random.randint(10,50), scores["Negative"]], name="Neg", line=dict(color="#dc3545")))
        fig_line.add_trace(go.Scatter(x=x_vals, y=[random.randint(0,20), random.randint(0,20), random.randint(0,20), random.randint(0,20), scores["Neutral"]], name="Neu", line=dict(color="#fd7e14")))
        
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(color="white")),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, color="white"),
            yaxis=dict(showgrid=True, gridcolor="#333", color="white")
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with row2_col2:
        st.markdown('<p class="section-title">Word Cloud</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Most frequent words in analyzed text</p>', unsafe_allow_html=True)
        
        wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='Set2').generate(english_text)
        fig_wc, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        fig_wc.patch.set_facecolor('white')
        st.pyplot(fig_wc)
