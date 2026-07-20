import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import urllib.request
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="InsightAI Dashboard", page_icon="🟣", layout="wide")

# --- 2. CUSTOM CSS (Adapted for Dark/Light Mode) ---
st.markdown("""
    <style>
    .card {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .metric-value {font-size: 28px; font-weight: bold; margin-top: 5px; color: inherit;}
    .pos-text {color: #4CAF50;}
    .neg-text {color: #E53935;}
    .neu-text {color: #FF9800;}
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()

# --- 4. TRANSLATION API ---
def translate_to_english(text):
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        translated_text = "".join([sentence[0] for sentence in data[0] if sentence[0]])
        return translated_text
    except:
        return text

# --- 5. INITIALIZE HISTORY (EMPTY/CLEAN STATE) ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MAIN DASHBOARD HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.header("Sentiment Analysis Dashboard")
    st.caption("Analyze text data to extract insights and understand sentiment patterns.")
with col_head2:
    st.button(f"📅 Last 7 Days: {datetime.now().strftime('%b %d, %Y')}")

# INPUT SECTION
review_text = st.text_input("", placeholder="Paste customer review here to analyze instantly...")

# DEFAULT VALUES (Set to 0 when empty)
pos_score, neg_score, neu_score, conf_score = 0, 0, 0, 0
pred_sentiment = "Neutral"
analyzed_text = ""

# PROCESS INPUT
if review_text and model:
    with st.spinner("Analyzing data..."):
        eng_text = translate_to_english(review_text)
        vec = vectorizer.transform([eng_text])
        pred = model.predict(vec)[0].capitalize()
        probs = model.predict_proba(vec)[0]
        
        pos_score = round(probs[1] * 100, 1)
        neg_score = round(probs[0] * 100, 1)
        neu_score = round(max(0, 100 - (pos_score + neg_score)), 1)
        conf_score = round(max(pos_score, neg_score), 1)
        pred_sentiment = pred
        analyzed_text = eng_text

        st.session_state.history.append({
            "Text": review_text[:60] + "...",
            "Sentiment": pred_sentiment,
            "Confidence": conf_score,
            "Pos": pos_score, "Neg": neg_score, "Neu": neu_score,
            "Date": datetime.now().strftime("%b %d, %Y %I:%M %p")
        })
elif len(st.session_state.history) > 0:
    # If text is cleared but history exists, show the last analysis
    latest_data = st.session_state.history[-1]
    pos_score = latest_data["Pos"]
    neg_score = latest_data["Neg"]
    neu_score = latest_data["Neu"]
    conf_score = latest_data["Confidence"]
    pred_sentiment = latest_data["Sentiment"]
    analyzed_text = st.session_state.history[-1].get("Original_Text", "")

# --- ROW 1: CARDS & GAUGE ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("**Overall Sentiment**<br><span style='color:gray; font-size:12px;'>Overview of sentiment distribution</span>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display: flex; gap: 15px; margin-top: 10px;">
        <div class="card" style="flex: 1; text-align: center;">
            <div class="pos-text" style="font-weight: bold;">😊 Positive</div>
            <div class="metric-value">{pos_score}%</div>
        </div>
        <div class="card" style="flex: 1; text-align: center;">
            <div class="neg-text" style="font-weight: bold;">😡 Negative</div>
            <div class="metric-value">{neg_score}%</div>
        </div>
        <div class="card" style="flex: 1; text-align: center;">
            <div class="neu-text" style="font-weight: bold;">😐 Neutral</div>
            <div class="metric-value">{neu_score}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("**Confidence Score**<br><span style='color:gray; font-size:12px;'>Overall analysis confidence</span>", unsafe_allow_html=True)
    gauge_color = "#4CAF50" if pred_sentiment == "Positive" else "#E53935" if pred_sentiment == "Negative" else "#FF9800"
    
    # If no data, show gray empty gauge
    if conf_score == 0:
        gauge_color = "rgba(128,128,128,0.2)"
        
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = conf_score,
        number = {'suffix': "%", 'font': {'size': 35}},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': gauge_color},
            'bgcolor': "rgba(128,128,128,0.1)",
            'borderwidth': 0,
        }
    ))
    fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", font={'color': 'gray'})
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- ROW 2: LINE CHART & WORD CLOUD ---
col3, col4 = st.columns([1.5, 1])

with col3:
    st.markdown("**Sentiment Over Time**<br><span style='color:gray; font-size:12px;'>Sentiment trend analysis</span>", unsafe_allow_html=True)
    if len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)
        fig_line = px.line(df_hist, x="Date", y=["Pos", "Neg", "Neu"], markers=True, 
                           color_discrete_map={"Pos": "#4CAF50", "Neg": "#E53935", "Neu": "#FF9800"})
        fig_line.update_layout(
            height=280, margin=dict(l=0, r=0, t=20, b=0), 
            legend_title_text='', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("📉 No data available. Analyze a review to see the trend line.")

with col4:
    st.markdown("**Word Cloud**<br><span style='color:gray; font-size:12px;'>Most frequent words in analyzed text</span>", unsafe_allow_html=True)
    if analyzed_text:
        try:
            wc = WordCloud(width=400, height=250, background_color='black', colormap='coolwarm', max_words=50).generate(analyzed_text)
            fig_wc, ax = plt.subplots(figsize=(4, 2.5))
            fig_wc.patch.set_facecolor('none') # Transparent background for dark mode
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        except:
            st.warning("Not enough words to generate a cloud.")
    else:
        st.info("☁️ Awaiting input to generate Word Cloud.")

# --- ROW 3: RECENT ANALYSES TABLE ---
st.markdown("**Recent Analyses**<br><span style='color:gray; font-size:12px;'>Latest sentiment analysis results</span>", unsafe_allow_html=True)
if len(st.session_state.history) > 0:
    df_display = pd.DataFrame(st.session_state.history)[::-1]
    st.dataframe(df_display[["Text", "Sentiment", "Confidence", "Date"]], use_container_width=True, hide_index=True)
else:
    st.write("No recent analyses found.")
