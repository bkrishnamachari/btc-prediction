import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
# Expecting BTC-Hourly.csv in current directory with a column 'date' and 'close'
# Data is assumed sorted from newest to oldest in the file sample, so we sort ascending

def load_data(path='BTC-Hourly.csv'):
    df = pd.read_csv(path)
    df = df.sort_values('date')  # chronological order
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_sequences(prices, look_back=24):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1))
    X, y = [], []
    # Keep last 4 observations for test/prediction
    for i in range(look_back, len(scaled) - 4):
        X.append(scaled[i - look_back:i])
        y.append(scaled[i])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler, scaled

def build_model(look_back):
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model


def main():
    look_back = 24
    df = load_data()
    prices = df['close'].values

    X, y, scaler, scaled = prepare_sequences(prices, look_back)

    model = build_model(look_back)
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    # Generate 4-hour forecast using last look_back values
    history = scaled[-look_back:].tolist()
    preds = []
    for _ in range(4):
        x_input = np.array(history[-look_back:]).reshape(1, look_back, 1)
        pred = model.predict(x_input, verbose=0)
        history.append(pred[0, 0])
        preds.append(pred[0, 0])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Actual values for next 4 hours
    actual = prices[-4:]

    # Baseline: use previous hour's price
    baseline = prices[-5:-1]

    mae_lstm = mean_absolute_error(actual, preds)
    mae_base = mean_absolute_error(actual, baseline)

    print(f"LSTM MAE for next 4 hours: {mae_lstm:.2f}")
    print(f"Baseline MAE for next 4 hours: {mae_base:.2f}")

    # Plot comparison
    dates = pd.to_datetime(df['date'].values[-4:])
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, label='Actual', marker='o')
    plt.plot(dates, preds, label='LSTM Forecast', marker='o')
    plt.plot(dates, baseline, label='Baseline', marker='o')
    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('BTC Price Prediction - Next 4 Hours')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
