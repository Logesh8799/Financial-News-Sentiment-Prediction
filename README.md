📊 Finance News Sentiment Prediction & Risk Monitoring Dashboard
    An end-to-end deep learning framework and interactive web dashboard designed to track, analyze, and triage retail investor sentiment using financial micro-blogs. Built around the structure of the Hugging Face zeroshot/twitter-financial-news-sentiment dataset, this project scales from baseline recurrent architectures (RNN, LSTM, GRU) up to a fine-tuned, state-of-the-art FinBERT transformer model.

💡 Core Capabilities
Market Mood Tracking:
Aggregates daily bullish vs. bearish text volumes to map shifting retail momentum across indices ($SPY, $QQQ) and hyper-growth tickers.
Portfolio Risk Monitoring:
Automatically monitors and flags sudden, statistically significant spikes in bearish sentiment to trigger downstream risk reviews or hedging discussions.
News-Driven Trading Signals (Prototype):
Establishes a backtesting alpha layer using a basic quantitative rule: entering long positions when retail sentiment flips from bearish/neutral to bullish.
Content Triage for Analysts: 
Systematically filters out background neutral noise (which typically accounts for 40-50% of financial tweets) to isolate high-conviction, polarized inputs for human analyst queues.
