import pandas as pd, numpy as np, datetime as dt, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------- hyper-params ------------------- #
LOOK_BACK         = 24      # hours of history per sample
FORECAST_HORIZON  = 4       # predict +1h…+4h
FORECAST_INTERVAL = 4       # issue forecasts every 4 h
TRAIN_DAYS        = 90
TEST_DAYS         = 30
EPOCHS            = 10
BATCH_SIZE        = 16
VERBOSE           = True    # toggle extra prints
# ---------------------------------------------------- #

def load_data(path='BTC-Hourly.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def make_xy(scaled, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i])
        y.append(scaled[i])
    return np.array(X), np.array(y)

def build_model(look_back=LOOK_BACK):
    m = Sequential([
        LSTM(50, input_shape=(look_back, 1)),
        Dense(1)
    ])
    m.compile('adam', loss='mae')
    return m

def main():
    df         = load_data()
    last_date  = df['date'].iloc[-1]

    train_from = last_date - dt.timedelta(days=TRAIN_DAYS)
    test_from  = last_date - dt.timedelta(days=TEST_DAYS)

    train_df   = df[(df['date'] >= train_from) & (df['date'] <  test_from)].copy()
    full_df    = df[df['date'] >= train_from].copy()        # 30-day slice for history + test

    # ------------------------- scale + train ------------------------- #
    scaler     = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df['close'].values.reshape(-1,1))
    X_train, y_train = make_xy(train_scaled)

    model = build_model()
    print(f"[INFO] Training on {len(X_train)} samples "
          f"({TRAIN_DAYS} days, look_back={LOOK_BACK}) …")
    model.fit(X_train, y_train,
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print("[INFO] Training finished.\n")

    # ----------------- prepare series for forecasting ---------------- #
    full_df['scaled_close'] = scaler.transform(full_df['close'].values.reshape(-1,1))
    scaled_series   = full_df['scaled_close'].tolist()
    idx_offset      = full_df.index[0]

    start_idx_7d    = df[df['date'] >= test_from].index[0]

    dates, preds, baselines, actuals = [], [], [], []

    print("[INFO] Rolling forecasts:")
    # iterate through issue times (one step BEFORE first prediction)
    for issue_idx in range(start_idx_7d-FORECAST_HORIZON,
                           len(df)-FORECAST_HORIZON,
                           FORECAST_INTERVAL):

        issue_time = df.at[issue_idx, 'date']
        if VERBOSE:
            print(f"  • issuing {FORECAST_HORIZON}-step forecast at "
                  f"{issue_time:%Y-%m-%d %H:%M}")

        hist_start = issue_idx - LOOK_BACK + 1
        history = scaled_series[hist_start-idx_offset : issue_idx-idx_offset+1].copy()
        last_obs_price = df.at[issue_idx, 'close']

        # produce 4 recursive steps
        for step in range(1, FORECAST_HORIZON+1):
            x_in = np.array(history[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
            pred_s = model.predict(x_in, verbose=0)[0,0]
            history.append(pred_s)

            target_idx  = issue_idx + step
            target_time = df.at[target_idx, 'date']

            dates.append(target_time)
            preds.append(scaler.inverse_transform([[pred_s]])[0,0])
            baselines.append(last_obs_price)
            actuals.append(df.at[target_idx, 'close'])

            if VERBOSE and step == FORECAST_HORIZON:
                print(f"     ↳ step {step}: "
                      f"pred={preds[-1]:,.0f}  "
                      f"base={last_obs_price:,.0f}  "
                      f"actual={actuals[-1]:,.0f}  "
                      f"→ {target_time:%m-%d %H:%M}")

    # ---------------------- evaluation + plot ------------------------ #
    lstm_mae = mean_absolute_error(actuals, preds)
    base_mae = mean_absolute_error(actuals, baselines)
    print(f"\n[RESULT] LSTM   MAE (multi-step) : {lstm_mae:,.2f}")
    print(f"[RESULT] Baseline MAE            : {base_mae:,.2f}")

    plt.figure(figsize=(13,7))
    ctx_start = last_date - dt.timedelta(days=TRAIN_DAYS)
    plt.plot(df[df['date'] >= ctx_start]['date'],
             df[df['date'] >= ctx_start]['close'],
             label='Actual (context)', alpha=.6)

    plt.scatter(dates, actuals,  s=18, label='Actual (7-day test)', marker='o',  color='black')
    plt.scatter(dates, preds,             label='LSTM forecast',    marker='^',  color='red')
    plt.scatter(dates, baselines,         label='Naïve baseline',   marker='x',  color='green')

    plt.title('BTC hourly close – rolling 4-hour-ahead forecasts')
    plt.xlabel('Date / Time'); plt.ylabel('Price')
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
