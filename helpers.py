import logging
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import redirect, render_template, request, session
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code

def login_required(f):
    """Decorate routes to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def lookup(symbol):
    """Look up stock quote using Yahoo Finance."""
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        if "regularMarketPrice" not in info or info["regularMarketPrice"] is None:
            logging.warning(f"No market data found for {symbol}")
            return None

        price = info["regularMarketPrice"]
        name = info.get("longName", symbol)

        return {
            "symbol": symbol,
            "name": name,
            "price": float(price)
        }
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

def predict_price(symbol):
    """Predict next day's stock price using Yahoo Finance historical data."""
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period="120d")  # last 120 days
        if df.empty:
            logging.warning(f"No historical data found for {symbol}")
            return None

        df = df[['Close']].copy()
        df["SMA_20"] = df["Close"].rolling(window=20).mean().fillna(method="bfill")
        df.reset_index(inplace=True)

        prices = df["Close"].values
        dates = df['Date'].dt.strftime("%Y-%m-%d").tolist()
        features = df[["Close", "SMA_20"]].values

        if len(prices) < 60:
            logging.warning(f"Insufficient data for {symbol}: {len(prices)} days")
            return None

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)

        look_back = 20
        X, y = [], []
        for i in range(look_back, len(scaled_features)):
            X.append(scaled_features[i-look_back:i])
            y.append(scaled_features[i, 0])
        X = np.array(X)
        y = np.array(y)

        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        model = Sequential()
        model.add(Input(shape=(look_back, 2)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")

        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[early_stopping], verbose=0)

        last_sequence = scaled_features[-look_back:]
        last_sequence = np.reshape(last_sequence, (1, look_back, 2))
        predicted_scaled = model.predict(last_sequence, verbose=0)

        predicted_array = np.zeros((1, 2))
        predicted_array[:, 0] = predicted_scaled[:, 0]
        predicted_price = scaler.inverse_transform(predicted_array)[0, 0]

        chart_prices = prices.tolist() + [predicted_price]

        return {
            "predicted_price": round(predicted_price, 2),
            "dates": dates,
            "chart_prices": chart_prices
        }

    except Exception as e:
        logging.error(f"Error in predict_price for {symbol}: {e}")
        return None

def inr(value):
    """Format value as INR (₹) for Indian stocks."""
    return f"₹{value:,.2f}"
