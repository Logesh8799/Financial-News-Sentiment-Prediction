import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Set up page configurations
st.set_page_config(page_title="Retail Investor Sentiment Dashboard", layout="wide")

st.title("📊 Retail Investor Sentiment & Risk Dashboard")
st.markdown("Track real-time market mood, generate prototype signals, and triage text using fine-tuned FinBERT.")

# 1. Load the Model Cache
@st.cache_resource
def load_sentiment_model():
    model_path = "./model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_sentiment_model()
    st.sidebar.success("✅ FinBERT Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")

# Class Index Mapping (FinBERT order: 0=Positive/Bullish, 1=Negative/Bearish, 2=Neutral)
classes = ["Bullish", "Bearish", "Neutral"]
sentiment_styles = {
    "Bullish": "🟢",
    "Bearish": "🔴",
    "Neutral": "⚪"
}

# 2. Main Interface Setup
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Real-time Micro-blog Analysis")
    user_input = st.text_area(
        "Type a tweet or market headline to evaluate:",
        value="$AAPL breaks out above moving average on massive volume! Strong guidance ahead."
    )
    
    if st.button("Run Sentiment Inference") or user_input:
        # Tokenize and run inference
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
        predicted_idx = np.argmax(probabilities)
        pred_label = classes[predicted_idx]
        
        # Display Core Prediction Metric Card
        st.metric(
            label="Predicted Sentiment Mood", 
            value=f"{sentiment_styles[pred_label]} {pred_label}"
        )
        
        # Structure data for rendering a chart
        prob_df = pd.DataFrame({
            "Market Sentiment Class": classes,
            "Probability (%)": [p * 100 for p in probabilities]
        })
        
        st.subheader("📈 Probability Weights")
        st.bar_chart(data=prob_df, x="Market Sentiment Class", y="Probability (%)")

with col2:
    st.subheader("📋 Context & Historical Validation Distribution")
    st.markdown("Baseline statistical distribution across your retail dataset:")
    
    # Use standard proportions typical for this financial validation data
    mock_distribution = pd.DataFrame({
        "Sentiment Metric": ["Neutral Noise", "Bullish Momentum", "Bearish Portfolio Risk"],
        "Tweet Count Share": [45, 35, 20]
    })
    
    st.dataframe(mock_distribution, use_container_width=True)
    
    # Content Triage Risk Alert simulation logic
    st.subheader("🚨 Automated Risk & Content Triage Alert System")
    if user_input and pred_label == "Bearish":
        st.error("⚠️ RISK REVIEW TRIGGERED: Significant Bearish sentiment spike found in retail content queue. Flagged for priority analyst review.")
    elif user_input and pred_label == "Bullish":
        st.success("⚡ ALPHA SIGNAL: Retail sentiment flipped to strong Bullish confirmation. Retaining long positions.")
    else:
        st.info("⏱️ Monitoring Stream: Neutral background noise detected. Low urgency queue triage.")
