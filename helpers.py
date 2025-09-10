import yfinance as yf
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)


def fetch_stock_data(symbol):
    """Fetch real-time stock data for NSE symbols using yfinance."""
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol + ".NS")
        data = ticker.history(period="1d")

        if data.empty:
            logging.warning(f"No data found for {symbol}")
            return None

        current_price = round(data["Close"].iloc[-1], 2)
        prev_close = round(data["Close"].iloc[0], 2)
        change = round(current_price - prev_close, 2)
        change_percent = round((change / prev_close) * 100, 2)

        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "percent": change_percent,
        }
    except Exception as e:
        logging.error(f"Error fetching stock data for {symbol}: {e}")
        return None


def predict_price(symbol):
    """Predict next day's stock price using Yahoo Finance historical data and Linear Regression."""
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period="120d")  # last 120 days

        if df.empty:
            logging.warning(f"No historical data found for {symbol}")
            return None

        # Prepare features
        df = df[["Close"]].copy()
        df["SMA_20"] = df["Close"].rolling(window=20).mean().fillna(method="bfill")
        df.reset_index(inplace=True)

        prices = df["Close"].values
        dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        features = df[["Close", "SMA_20"]].values

        if len(prices) < 20:
            logging.warning(f"Insufficient data for {symbol}: {len(prices)} days")
            return None

        # Train Linear Regression model
        X = features[:-1]
        y = df["Close"].values[1:]

        model = LinearRegression()
        model.fit(X, y)

        # Predict next day's closing price
        next_features = features[-1].reshape(1, -1)
        predicted_price = model.predict(next_features)[0]

        chart_prices = prices.tolist() + [predicted_price]

        return {
            "predicted_price": round(float(predicted_price), 2),
            "dates": dates,
            "chart_prices": chart_prices,
        }

    except Exception as e:
        logging.error(f"Error in predict_price for {symbol}: {e}")
        return None
