import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime
import random

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS (Wide Layout)
# ==========================================
st.set_page_config(page_title="InsightAI Dashboard", page_icon="🟣", layout="wide")

st.markdown("""
    <style>
    /* Input box styling to match the red glow/border */
    .stTextInput input {
        border: 1px solid #ff4b4b !important;
        background-color: #1e1e24;
        color: white;
        border-radius: 8px;
        padding: 15px;
    }
    /* White Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-card h4 { margin: 0; font-size: 16px; font-weight: bold; }
    .metric-card h1 { margin: 10px 0 0 0; font-size: 32px; color: #111; font-weight: 900;}
    
    /* specific text colors */
    .text-pos { color: #28a745; }
    .text-neg { color: #dc3545; }
    .text-neu { color: #fd7e14; }
    .text-mix { color: #6f42c1; }
    
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
    # top_k=None forces the model to return percentages for ALL 4 classes
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

# Input Box
user_text = st.text_input("", placeholder="Paste customer review here to analyze instantly...")

# ==========================================
# 4. DASHBOARD PROCESSING & UI (Only runs if text is entered)
# ==========================================
if user_text:
    # --- TRANSLATION & PREDICTION ---
    try:
        english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
    except:
        english_text = user_text
        
    # Get all 4 scores
    results = analyzer(english_text)[0]
    
    # Map raw labels (LABEL_0, etc.) to readable format and extract percentages
    scores = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive", "LABEL_3": "Mixed"}
    
    for res in results:
        mapped_name = label_mapping.get(res['label'], res['label'])
        scores[mapped_name] = round(res['score'] * 100)
    
    # Find the winning sentiment for the Gauge Chart
    top_sentiment = max(scores, key=scores.get)
    top_score = scores[top_sentiment]
    
    gauge_color = "#28a745" if top_sentiment == "Positive" else "#dc3545" if top_sentiment == "Negative" else "#fd7e14" if top_sentiment == "Neutral" else "#6f42c1"

    # --- ROW 1: METRICS & GAUGE CHART ---
    st.markdown("<br>", unsafe_allow_html=True)
    row1_col1, row1_col2 = st.columns([1.5, 1])
    
    with row1_col1:
        st.markdown('<p class="section-title">Overall Sentiment</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Overview of sentiment distribution across 4 classes</p>', unsafe_allow_html=True)
        
        # 4 Metric Cards in a row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''<div class="metric-card"><h4 class="text-pos">🤩 Positive</h4><h1>{scores["Positive"]}%</h1></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="metric-card"><h4 class="text-neg">😡 Negative</h4><h1>{scores["Negative"]}%</h1></div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="metric-card"><h4 class="text-neu">😐 Neutral</h4><h1>{scores["Neutral"]}%</h1></div>''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''<div class="metric-card"><h4 class="text-mix">🤔 Mixed</h4><h1>{scores["Mixed"]}%</h1></div>''', unsafe_allow_html=True)
            
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
        
        # Generating a simulated trend graph that leads up to the current scores
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
        
        # Generate Word Cloud
        wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='Set2').generate(english_text)
        fig_wc, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        fig_wc.patch.set_facecolor('white') # Keep background of image white like screenshot
        st.pyplot(fig_wc)
