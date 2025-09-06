#importing libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#***************** FLASK *****************************
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/index')
def index():
   return render_template('index.html')

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


def get_historical(symbol):
    apikey = '9f2a3f356de7477482431de27a7ab59c'
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3650)
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&apikey={apikey}"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if 'values' in data:
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_csv(f"{symbol}.csv", index=False)
        return df
    else:
        return None

def fetch_reddit_data(stock_ticker, limit=1000):
    posts_data = []
    after = None
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    subreddit = 'StockMarket'
    five_years_ago = datetime.now() - timedelta(days=5*365)

    while len(posts_data) < limit:
        params = {
            'limit': 100,
            'after': after,
            'q': stock_ticker
        }

        try:
            response = requests.get(f"https://www.reddit.com/r/{subreddit}/search.json", headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                for post in data['data']['children']:
                    try:
                        title = post['data']['title']
                        post_url = "https://www.reddit.com" + post['data']['permalink']
                        score = post['data']['score']
                        comments = post['data']['num_comments']
                        date = post['data']['created_utc']
                        post_date = pd.to_datetime(date, unit='s')

                        if post_date < five_years_ago or post_date.weekday() >= 5:
                            continue

                        posts_data.append(title)

                        if len(posts_data) >= limit:
                            break
                    except:
                        continue

                after = data['data'].get('after', None)
                if not after:
                    break
                time.sleep(10)
            else:
                break
        except:
            time.sleep(10)
            continue

    return posts_data

def analyze_sentiment(titles):
    analyzer = SentimentIntensityAnalyzer()
    total_sentiment = 0
    for title in titles:
        sentiment_score = analyzer.polarity_scores(title)['compound']
        total_sentiment += sentiment_score
    avg_sentiment = total_sentiment / len(titles) if titles else 0
    return round(avg_sentiment, 4)

def GRU_ALGO(df):
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    training_set = df.iloc[:, 4:5].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(30, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-30:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_forecast = np.append(X_train[-1, 1:], y_train[-1])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

    model = Sequential([
        GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    real_stock_price = dataset_test.iloc[:, 4:5].values
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values.reshape(-1, 1)
    testing_set = sc.transform(testing_set)

    X_test = []
    for i in range(30, len(testing_set)):
        X_test.append(testing_set[i-30:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    error_gru = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    forecasted_stock_price = model.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
    gru_pred = forecasted_stock_price[0, 0]

    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price.flatten(), label='Actual Price')
    plt.plot(predicted_stock_price.flatten(), label='Predicted Price')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig('static/GRU.png')
    plt.close(fig)


    return gru_pred, error_gru

@app.route('/predict', methods=['POST'])
def predict():
    nm = request.form['nm']
    quote = nm.upper()
    try:
        get_historical(quote)
    except:
        return render_template('index.html', not_found=True)

    df = pd.read_csv(f"{quote}.csv")
    df = df.dropna()
    today_stock = df.iloc[-1:]
    gru_pred, error_gru = GRU_ALGO(df)
    reddit_titles = fetch_reddit_data(quote, limit=200)
    sentiment_score = analyze_sentiment(reddit_titles)
    if sentiment_score >= 0.05:
        recommendation = "BUY"
    elif sentiment_score <= -0.05:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return render_template("results.html",
        quote=quote,
        gru_pred=round(gru_pred, 2),
        open_s=today_stock['Open'].to_string(index=False),
        close_s=today_stock['Close'].to_string(index=False),
        high_s=today_stock['High'].to_string(index=False),
        low_s=today_stock['Low'].to_string(index=False),
        vol=today_stock['Volume'].to_string(index=False),
        error_gru=round(error_gru, 2),
        sentiment_score=sentiment_score,
        recommendation=recommendation
    )

if __name__ == '__main__':
    app.run(port=5000)
