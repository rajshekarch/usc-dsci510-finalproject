
# 📊 USC DSCI 510 Final Project

**Integrating Financial Metrics and Reddit Sentiment for Stock Market Prediction**

## 📌 Overview

This project analyzes the influence of **traditional financial indicators** (EPS, P/E Ratio, Market Cap) versus **Reddit sentiment** on short-term stock price movements.
By combining structured stock/financial data with crowd-driven sentiment, the project investigates whether online discussions provide predictive value for market dynamics.


## 🚀 Features

* **Multimodal Data Integration**: Historical stock prices + company financials + Reddit sentiment.
* **Data Cleaning & Preprocessing**: Deduplication, sentiment scoring (VADER), scaling, and normalization.
* **Exploratory Analysis**: Correlation heatmaps, time-series visualization, sentiment distribution.
* **Machine Learning & Deep Learning Models**:

  * LSTM & GRU → captured temporal dependencies.
  * CNN → short-term trend recognition.
  * XGBoost → benchmark on tabular data.
* **Sentiment Classification**: Naive Bayes, Logistic Regression, and SVM (best accuracy \~82%).
* **Web Application**: Flask + GCP deployment with live sentiment scraping.


## 📂 Dataset Sources

1. **Historical Stock Prices** → Twelve Data API
2. **Company Financials** → Yahoo Finance API
3. **Reddit Sentiment Data** → Web scraping + VADER sentiment analysis

Final dataset → **1,159 rows × 23 columns** (integrated across all three sources).


## 📈 Key Findings

* Reddit **Net Sentiment** showed weak but higher correlation with short-term price change (0.0754) compared to P/E, EPS, and Market Cap.
* **GRU model** achieved the best performance for stock prediction with RMSE = 5.53, R² = 0.8987.
* Traditional financial indicators (especially **P/E Ratio**) remain strong predictors in small datasets.
* Sentiment did **not** reliably predict volatility.

## 🌐 Deployment

The project includes a **Flask web app** deployed on Google Cloud Platform.

* Local run → `http://127.0.0.1:5000/`
* Cloud deployment → [Live Demo](https://usc-dsci510-finalproject.wl.r.appspot.com/)


## 🛠️ Tech Stack

* **Languages/Frameworks**: Python 3.12, Flask
* **ML/DL Libraries**: TensorFlow, scikit-learn, XGBoost
* **NLP Tools**: VADER Sentiment, NLTK (lemmatization, stopword removal)
* **Visualization**: Matplotlib, Seaborn
* **Cloud Hosting**: Google Cloud Platform (App Engine)

## 📌 Future Work

* Add more sentiment sources (Twitter, StockTwits, financial news).
* Real-time streaming with Kafka/WebSockets.
* Advanced volatility models (GARCH, LSTM-based).
* Portfolio analysis with explainable AI (SHAP, LIME).
* Mobile app for stock sentiment and predictions.

## 👨‍💻 Author

**Raja Shaker Chinthakindi**
Master’s in Applied Data Science @ University of Southern California (USC)
