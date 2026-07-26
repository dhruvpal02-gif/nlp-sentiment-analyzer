import streamlit as st
import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS 
# ==========================================
st.set_page_config(page_title="InsightAI Dashboard", page_icon="🟣", layout="wide")

st.markdown("""
    <style>
    .stTextInput input {
        border: 1px solid #ff4b4b !important;
        background-color: #1e1e24;
        color: white;
        border-radius: 8px;
        padding: 15px;
    }
    .dashboard-title { font-size: 2.5rem; font-weight: 700; margin-bottom: 0;}
    .dashboard-subtitle { color: #a0a0a5; font-size: 1.1rem; margin-bottom: 2rem;}
    .section-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 5px;}
    .section-subtitle { color: #a0a0a5; font-size: 0.9rem; margin-bottom: 15px;}
    .date-badge {
        background-color: #1e1e24; border: 1px solid #333;
        padding: 8px 15px; border-radius: 6px; float: right;
        color: #a0a0a5; font-size: 0.9rem; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE (Live Memory Setup)
# ==========================================
# Ye app ki memory hai. Jab tak tab open hai, history save rahegi.
if 'history' not in st.session_state:
    st.session_state.history = []
if 'review_count' not in st.session_state:
    st.session_state.review_count = 0

# ==========================================
# 3. LOAD AI MODEL
# ==========================================
@st.cache_resource
def load_model():
    MODEL_NAME = "dhruvpal02/ultimate-sentiment-ai" 
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, top_k=None)

with st.spinner("Initializing Dashboard Engines..."):
    analyzer = load_model()

# ==========================================
# 4. HEADER SECTION
# ==========================================
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown('<p class="dashboard-title">Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Live Tracking: Analyze reviews and build real-time trend history.</p>', unsafe_allow_html=True)
with col_header2:
    today_date = datetime.datetime.now().strftime("%b %d, %Y")
    st.markdown(f'<div class="date-badge">🗓️ Session Date: {today_date}</div>', unsafe_allow_html=True)

user_text = st.text_input("", placeholder="Paste customer review here and hit Enter...")

# ==========================================
# 5. DATA PROCESSING & SAVING TO HISTORY
# ==========================================
scores = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
top_sentiment = "Awaiting Input"
top_score = 0
gauge_color = "#555555"
wc_text = "Awaiting Data" 
english_text = ""

if user_text.strip():
    # Translate
    try:
        english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
    except:
        english_text = user_text
        
    wc_text = english_text 
    
    # Show Translation Box if needed
    if english_text.lower().strip() != user_text.lower().strip():
        st.markdown(f"""
        <div style="background-color: #2b2b36; border-left: 4px solid #007BFF; padding: 12px 20px; border-radius: 8px; margin-top: -15px; margin-bottom: 25px;">
            <span style="color: #a0a0a5; font-size: 13px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;">🌍 Translated to English for Analysis:</span><br>
            <span style="color: #ffffff; font-size: 16px; font-weight: 500;">"{english_text}"</span>
        </div>
        """, unsafe_allow_html=True)

    # Run AI Prediction
    results = analyzer(english_text)[0]
    label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive", "LABEL_3": "Mixed"}
    
    for res in results:
        mapped_name = label_mapping.get(res['label'], res['label'])
        scores[mapped_name] = round(res['score'] * 100)
    
    top_sentiment = max(scores, key=scores.get)
    top_score = scores[top_sentiment]
    gauge_color = "#28a745" if top_sentiment == "Positive" else "#dc3545" if top_sentiment == "Negative" else "#fd7e14" if top_sentiment == "Neutral" else "#6f42c1"

    # Save to Live History (Only if it's a new review)
    if not st.session_state.history or st.session_state.history[-1]['Original Text'] != user_text:
        st.session_state.review_count += 1
        st.session_state.history.append({
            "ID": f"R-{st.session_state.review_count}",
            "Original Text": user_text,
            "Result": top_sentiment,
            "Positive (%)": scores["Positive"],
            "Negative (%)": scores["Negative"],
            "Neutral (%)": scores["Neutral"],
            "Mixed (%)": scores["Mixed"]
        })

# ==========================================
# 6. DASHBOARD UI (Cards & Gauge)
# ==========================================
row1_col1, row1_col2 = st.columns([1.5, 1])

with row1_col1:
    st.markdown('<p class="section-title">Latest Review Sentiment</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Real-time analysis of the current input</p>', unsafe_allow_html=True)
    
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

# ==========================================
# 7. LIVE TREND CHART & WORD CLOUD
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
row2_col1, row2_col2 = st.columns([1.5, 1])

with row2_col1:
    st.markdown('<p class="section-title">Live Sentiment Trend</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Real historical data of reviews checked in this session</p>', unsafe_allow_html=True)
    
    fig_line = go.Figure()
    
    if len(st.session_state.history) > 0:
        x_vals = [item["ID"] for item in st.session_state.history]
        y_pos = [item["Positive (%)"] for item in st.session_state.history]
        y_neg = [item["Negative (%)"] for item in st.session_state.history]
        y_neu = [item["Neutral (%)"] for item in st.session_state.history]

        fig_line.add_trace(go.Scatter(x=x_vals, y=y_pos, mode='lines+markers', name="Pos", line=dict(color="#28a745", width=3), marker=dict(size=8)))
        fig_line.add_trace(go.Scatter(x=x_vals, y=y_neg, mode='lines+markers', name="Neg", line=dict(color="#dc3545", width=3), marker=dict(size=8)))
        fig_line.add_trace(go.Scatter(x=x_vals, y=y_neu, mode='lines+markers', name="Neu", line=dict(color="#fd7e14", width=3), marker=dict(size=8)))
    else:
        # Empty placeholder if no history
        fig_line.add_trace(go.Scatter(x=["No Data"], y=[0], mode='lines', line=dict(color="#555")))

    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01, font=dict(color="white")),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="#333", color="white", range=[0, 105])
    )
    st.plotly_chart(fig_line, use_container_width=True)

with row2_col2:
    st.markdown('<p class="section-title">Word Cloud</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Words from your latest input</p>', unsafe_allow_html=True)
    
    wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='Set2').generate(wc_text)
    fig_wc, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    fig_wc.patch.set_facecolor('white')
    st.pyplot(fig_wc)

# ==========================================
# 8. LIVE SESSION HISTORY TABLE
# ==========================================
st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
st.markdown('<p class="section-title">📝 Session Log (Live History)</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">List of all reviews analyzed since you opened the page.</p>', unsafe_allow_html=True)

if len(st.session_state.history) > 0:
    # Convert list of dictionaries to Pandas DataFrame for a beautiful table
    df_history = pd.DataFrame(st.session_state.history)
    # Reverse it so newest review is at the top
    df_history = df_history.iloc[::-1].reset_index(drop=True)
    
    # Display dataframe in Streamlit
    st.dataframe(df_history, use_container_width=True, hide_index=True)
else:
    st.info("No reviews analyzed yet. Start typing above to build your history!")
