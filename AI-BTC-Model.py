import ccxt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pytz
import time
import os
from urllib3.exceptions import ReadTimeoutError
from newsapi import NewsApiClient
import nltk
from nltk.tokenize import word_tokenize
import ssl

# Specify the NLTK data path where you've manually downloaded the resources
nltk.data.path.append("/Path/to/NltkData")

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context

try:
    # Downloading NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print("Error occurred while downloading NLTK resources:", e)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Ensure NLTK data path is set correctly
nltk.data.path.append("Path/to/NltkData")

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

threshold = 99.9 #Threshold for retraining

#Clear screen - can be discarded by preference
os.system('cls' if os.name == 'nt' else 'clear')


def fetch_historical_data(exchange, symbol, timeframe):
    historical_data = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
    return df

def collect_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'  # day timeframe
    return fetch_historical_data(exchange, symbol, timeframe)

def fetch_news_articles():
    # Init
    newsapi = NewsApiClient(api_key='UseYourOwnAPIKeyFromNewsAPI')

    # Define the keywords and date range
    keywords = 'bitcoin cryptocurrency inflation economy'
    start_date = datetime.now() - timedelta(days=30)  # Fetch articles from the last month
    end_date = datetime.now()

    # Fetch news articles
    all_articles = newsapi.get_everything(q=keywords,
                                          language='en',
                                          from_param=start_date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          sort_by='relevancy',
                                          page=1)

    return all_articles['articles']

def preprocess_data(df, news_articles):
    # Preprocess historical price data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    X = []
    y = []
    sequence_length = 100  # Number of time steps to look back
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length][3])
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Preprocess news articles
    preprocessed_articles = preprocess_news_articles(news_articles, X, len(X), sequence_length) 

    # Combine features from historical price data and news articles
    preprocessed_articles_reshaped = np.reshape(preprocessed_articles, (preprocessed_articles.shape[0], preprocessed_articles.shape[1], 1))
    combined_features = np.concatenate((X, preprocessed_articles_reshaped), axis=2)  # Concatenate along the features axis

    # Adjust input shape
    input_shape = (combined_features.shape[1], combined_features.shape[2])

    return combined_features, y, scaler, input_shape  # Return the scaler object and input_shape


def preprocess_news_articles(articles, X, num_samples, sequence_length):
    # Initialize NLTK resources
    stop_words = set(stopwords.words('english'))

    # Preprocess each news article
    preprocessed_articles = []
    for article in articles:
        # Combine title and content (if available)
        text = article['title'] + ' ' + (article.get('content', '') or '')

        # Tokenize and remove stopwords using NLTK
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Join tokens back into a single string
        preprocessed_text = ' '.join(filtered_tokens)
        preprocessed_articles.append(preprocessed_text)

    # Convert text data into numerical features
    vectorizer = TfidfVectorizer(max_features=sequence_length)  # Adjust max_features to match the number of time steps in X
    tfidf_features = vectorizer.fit_transform(preprocessed_articles).toarray()

    # Reshape to have the same number of samples as X
    tfidf_features_resized = np.resize(tfidf_features, (num_samples, tfidf_features.shape[1]))

    return tfidf_features_resized

    #LSTM architecture

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=512, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=8, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=4, return_sequences=False))  
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

    #Model training

def train_model(model, X_train, y_train, epochs=5, batch_size=4, validation_split=0.05):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    return history

def predict_future_price(model, X_test, scaler):
    future_time = datetime.now(pytz.timezone('Europe/Oslo')) + timedelta(days=1)  # Predicting 1 day into the future
    predicted_price = model.predict(X_test[-1].reshape(1, 100, X_test.shape[2]))[0][0]  #reshape array
    return future_time, predicted_price

def inverse_transform(scaled_value, scaler):
    # Reshape the scaled_value to have shape (1, 5)
    scaled_value_reshaped = np.array([[scaled_value] * scaler.n_features_in_])
    # Perform inverse transformation
    return scaler.inverse_transform(scaled_value_reshaped)[0][0]

def evaluate_model(actual_price_actual, predicted_price_actual):
    percentage_difference = ((actual_price_actual - predicted_price_actual) / actual_price_actual) * 100
    return percentage_difference

def calculate_accuracy(actual_price_actual, predicted_price_actual):
    accuracy = 100 - (abs(actual_price_actual - predicted_price_actual) / actual_price_actual) * 100
    return accuracy

while True:
    try:
        # Collect data
        df = collect_data()

        # Fetch news articles
        news_articles = fetch_news_articles()

        # Preprocess data
        X, y, scaler, input_shape = preprocess_data(df, news_articles)  # Capture the scaler object returned by preprocess_data

        # Build LSTM model
        model = build_lstm_model(input_shape)

        # Train the model
        history = train_model(model, X, y)  # Use all data for training for simplicity

        # Predict future price
        future_time, predicted_price_scaled = predict_future_price(model, X, scaler)

        # Inverse transform the scaled prices
        predicted_price_actual = inverse_transform(predicted_price_scaled, scaler)

        # Print the articles found
        print("News Articles:")
        for article in news_articles:
            title = article['title']
            published_at = article['publishedAt']
            print(f"Title: {title}")
            print(f"Published Date: {published_at}")
            print("-------------------------------------")

        # Print predicted price
        print("Predicted BTC Price at", future_time.strftime("%Y-%m-%d %H:%M:%S (Norwegian Time)"), ":", predicted_price_actual,"USD")

        #Fetch the latest data from Binance
        latest_df = collect_data()
        #Print current price for comparison
        actual_price_actual_1 = latest_df.iloc[-1]['close']
        print("Current price:", actual_price_actual_1,"USD")

        price_difference = round(predicted_price_actual - actual_price_actual_1, 2)

        # Determine color based on price difference
        if price_difference > 0:
            price_difference_color = "\033[92m"  # Green color for positive difference
        elif price_difference < 0:
            price_difference_color = "\033[91m"  # Red color for negative difference
        else:
            price_difference_color = ""  # No color for zero difference

        # Print price difference with appropriate color
        print("Price Difference:", f"{price_difference_color}{price_difference} USD\033[0m")
        
        # Provide recommendation based on price difference
        if price_difference > 0:
            recommendation = "\033[92mBUY\033[0m"  # Green color for BUY
            arrow = "▲"
        else:
            recommendation = "\033[91mSELL\033[0m"  # Red color for SELL
            arrow = "▼"

        print("Recommendation:", f"{arrow} {recommendation}")

        # Wait for 1 day
        time.sleep(24*60*60)  

        # Fetch the latest data from Binance for comparison
        latest_df = collect_data()

        # Get the actual price from the current timestamp
        actual_time = datetime.now(pytz.timezone('Europe/Oslo'))
        actual_price_actual = latest_df.iloc[-1]['close']

        # Calculate model accuracy
        accuracy = calculate_accuracy(actual_price_actual, predicted_price_actual)

        # Print actual BTC price and model accuracy
        print("Actual BTC Price at", actual_time.strftime("%Y-%m-%d %H:%M:%S (Norwegian Time)"), ":", actual_price_actual,"USD")
        print("Model Accuracy:", round(accuracy, 2), "%")
        
        # Check if model accuracy is below threshold for retraining
        if accuracy < threshold:
            print("Model accuracy below threshold. Retraining model with new data...")
            # Retrain the model with new data
            X_new, y_new, _, _ = preprocess_data(latest_df, news_articles)  # Preprocess new data
            X_combined = np.concatenate((X, X_new), axis=0)  # Combine old and new data
            y_combined = np.concatenate((y, y_new), axis=0)
            history = train_model(model, X_combined, y_combined)  # Retrain the model


    except ReadTimeoutError as e:
        print("Timeout error occurred:", e)
        print("Retrying after 60 seconds...")
        time.sleep(60)  # Wait for 1 minute before retrying
    # Continue the loop to fetch new data and repeat the process
