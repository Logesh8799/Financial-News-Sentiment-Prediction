# 📊 Financial News Sentiment Prediction & Risk Monitoring Dashboard

An end-to-end deep learning framework and interactive web dashboard designed to track, analyze, and triage retail investor sentiment using financial micro-blogs. Built around the structure of the Hugging Face `zeroshot/twitter-financial-news-sentiment` dataset, this project scales from baseline recurrent architectures (RNN, LSTM, GRU) up to a fine-tuned, state-of-the-art **FinBERT** transformer model.

---

## 💡 Core Capabilities

*   **Market Mood Tracking:** Aggregates daily bullish vs. bearish text volumes to map shifting retail momentum across major indices (\$SPY, \$QQQ) and hyper-growth tickers.
*   **Portfolio Risk Monitoring:** Automatically monitors and flags sudden, statistically significant spikes in bearish sentiment to trigger downstream risk reviews or hedging discussions.
*   **News-Driven Trading Signals (Prototype):** Establishes a backtesting alpha layer using a basic quantitative rule: entering long positions when retail sentiment flips from bearish/neutral to bullish.
*   **Content Triage for Analysts:** Systematically filters out background neutral noise (which typically accounts for 40-50% of financial tweets) to isolate high-conviction, polarized inputs for human analyst queues.

---

## 📈 Dataset Overview & Target Layout

The pipeline handles financial micro-blogs classified into three distinct target classes:

| Label ID | Sentiment Class | Market Implication | Approximate Distribution |
| :--- | :--- | :--- | :--- |
| **0** | Bearish | Downside risk, short interest expansion, panic indicators | ~15% - 20% |
| **1** | Bullish | Upside momentum, breakout volume accumulation | ~30% - 35% |
| **2** | Neutral | Standard corporate earnings releases, informational links | ~45% - 50% |

---

## 🏗️ Project Architecture & Pipeline

1. **Text Preprocessing:** Vectorized cleaning routines strip out web tracking links, HTML escape entities, and cluttered user handles while explicitly preserving asset cash tags (`$TSLA`), percentages (`+5.4%`), and directional markers.
2. **Vocabulary Building & Encoding:** Builds an independent vocabulary exclusively on the training split to eliminate data leakage. Unknown out-of-vocabulary terms map cleanly to an `<UNK>` token, and sequence variables pad to index `0` (`<PAD>`).
3. **Baseline Recurrent Suite:** Custom PyTorch implementations of standard Reconstructive Recurrent layers:
    *   `Embedding` + `Simple RNN`
    *   `Embedding` + `LSTM` (Long Short-Term Memory)
    *   `Embedding` + `GRU` (Gated Recurrent Unit)
4. **Transformer Optimization:** Fine-tunes a pre-trained **FinBERT** (`ProsusAI/finbert`) sequence classification network utilizing Hugging Face's automated optimization ecosystem.
5. **Streamlit Deployment:** An interactive web portal displaying inference controls, classification probability metrics, and real-time content triage warning modules.

---

## ⚙️ Installation & Local Setup (Windows)

This repository requires **Python 3.12+**. Follow these precise setup rules to build the environment locally on Windows and prevent runtime dependency faults (`c10.dll` errors).

### 1. Environment Isolation
Open your command prompt (`cmd`) in the project root directory and initialize a clean virtual environment:
```cmd
python -m venv venv
call venv\Scripts\activate
```

### 2. Install PyTorch with Clean Backend Links
To prevent DLL initialization crashes on local Windows architectures, select the installation command matching your hardware setup:

*   **For Standard CPU Execution:**
    ```cmd
    pip install torch --index-url https://pytorch.org
    ```
*   **For Nvidia GPU (CUDA Acceleration) Execution:**
    ```cmd
    pip install torch --index-url https://pytorch.org
    ```

### 3. Install Secondary Pipeline Requirements
Install standard dataset management libraries, transformer components, visualization packages, and the dashboard engine:
```cmd
pip install datasets transformers evaluate scikit-learn pandas numpy matplotlib seaborn streamlit
```

---

## 🚀 Usage Guide

### Running Notebook Pipelines (Google Colab / Local Jupyter)
If utilizing Google Colab for resource-intensive transformer fine-tuning, ensure your Colab instance is routed to hardware acceleration via **Runtime > Change runtime type > T4 GPU**.

> **Note on Patching Environment Bugs:** During training, if an `ImportError` regarding `VideoReader` from `torchvision.io` surfaces due to internal package collision frameworks, place this patch execution block directly before your training command:
> ```python
> import sys
> if "torchvision" in sys.modules:
>     del sys.modules["torchvision"]
> ```

### Launching the Streamlit Dashboard
Ensure your fine-tuned model checkpoint folder (`saved_finbert_dashboard_model/`) is located in your root directory. Boot the web dashboard by running:
```cmd
streamlit run app.py
```

---

## 📊 Evaluation & Benchmarking Strategy

Models are evaluated across the validation split utilizing three primary criteria: **Overall Classification Accuracy**, **Macro-Averaged F1-Score** (to penalize class imbalance neglect), and a **Confusion Matrix Visual Diagonal** to assess the exact rate of high-priority sentiment items being mistakenly flagged as neutral background noise.

*   **Baseline Recurrent Models:** Suffer from sequence limits, but GRU/LSTM layers offer functional performance foundations.
*   **FinBERT Transformer:** Consistently yields the highest metric yields due to its pre-trained exposure to massive economic data layers and underlying financial vernacular context (e.g., tracking phrases like *"guidance cuts"* or *"short squeeze"*).

---

## 📂 Repository File Layout

```text
├── .gitignore
├── README.md
├── app.py                          # Streamlit application entry point
├── requirements.txt                # System dependency configuration lists
├── financial_sentiment_pipeline.ipynb # Full exploratory analysis, RNN training & BERT scripts
└── saved_finbert_dashboard_model/  # Saved tokenizer and transformer weights configuration directory
```
