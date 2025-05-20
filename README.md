# BTC Price Prediction

This repository demonstrates a simple approach for forecasting short term Bitcoin prices using an LSTM network. The provided dataset contains several years of hourly BTC/USD candles and a training script that predicts the next few hours of the closing price.

## Contents
- `BTC-Hourly.csv` – Hourly open/high/low/close and volume data.
- `btc-price-predict.py` – Training and evaluation script.

## Requirements
- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow`

Install the packages with:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

## Usage
1. Ensure `BTC-Hourly.csv` is available in the repository directory.
2. Run the forecasting script:

```bash
python btc-price-predict.py
```

The script trains a small LSTM model on recent data and then rolls forward to produce 4‑step forecasts. Mean absolute errors for both the model and a naive baseline are printed, and a plot of actual versus predicted prices is displayed.

Hyper‑parameters such as the look‑back window, forecast horizon and training epochs can be modified at the top of `btc-price-predict.py`.

## Dataset
Each row in `BTC-Hourly.csv` represents one hour of trading activity with columns for the open, high, low, close and volume. The script sorts the data chronologically before training and uses only the `close` column for prediction.

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
