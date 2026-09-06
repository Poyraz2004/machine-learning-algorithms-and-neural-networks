import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os


# LOAD DATA
print("Downloading BTC-USD data...")
df = yf.download("BTC-USD", start="2018-01-01", end="2024-12-31", auto_adjust=True)
df = df[["Close"]].dropna()
df.index = pd.to_datetime(df.index)

print(f"Dataset shape: {df.shape}")
print(df.head())


# TRAIN / TEST SPLIT
split_date = "2023-01-01"
train_df = df[df.index < split_date]
test_df  = df[df.index >= split_date]

print(f"\nTrain: {train_df.index[0].date()} → {train_df.index[-1].date()}  ({len(train_df)} days)")
print(f"Test : {test_df.index[0].date()}  → {test_df.index[-1].date()}   ({len(test_df)} days)")

# NORMALISE

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df.values)   # fit only on train!
test_scaled  = scaler.transform(test_df.values)


# CREATE SEQUENCES

LOOK_BACK = 60

def make_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, LOOK_BACK)
X_test,  y_test  = make_sequences(test_scaled,  LOOK_BACK)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  1)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")


# BUILD LSTM MODEL
tf.random.set_seed(42)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOK_BACK, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()


# TRAIN
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

final_loss = history.history["loss"][-1]
print(f"\nFinal training loss: {final_loss:.6f}")


# PREDICT & INVERSE-TRANSFORM
train_pred = scaler.inverse_transform(model.predict(X_train))
test_pred  = scaler.inverse_transform(model.predict(X_test))

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv  = scaler.inverse_transform(y_test.reshape(-1, 1))


train_dates = train_df.index[LOOK_BACK:]
test_dates  = test_df.index[LOOK_BACK:]


# EVALUATION METRICS

from sklearn.metrics import mean_squared_error, mean_absolute_error

train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test_inv,  test_pred))
train_mae  = mean_absolute_error(y_train_inv, train_pred)
test_mae   = mean_absolute_error(y_test_inv,  test_pred)

print(f"\n{'='*40}")
print(f"  Train RMSE : ${train_rmse:,.2f}")
print(f"  Train MAE  : ${train_mae:,.2f}")
print(f"  Test  RMSE : ${test_rmse:,.2f}")
print(f"  Test  MAE  : ${test_mae:,.2f}")
print(f"  Final Loss : {final_loss:.6f}")
print(f"{'='*40}\n")


# PLOTS
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(18, 14))

ax1 = fig.add_subplot(2, 2, (1, 2))
ax1.plot(df.index, df["Close"], color="#444444", linewidth=0.8, label="BTC-USD Close Price")
ax1.axvline(pd.Timestamp(split_date), color="crimson", linewidth=2, linestyle="--", label=f"Train/Test split ({split_date})")
ax1.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                 where=(df.index < pd.Timestamp(split_date)),
                 alpha=0.08, color="steelblue", label="Train region")
ax1.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                 where=(df.index >= pd.Timestamp(split_date)),
                 alpha=0.08, color="tomato", label="Test region")
ax1.set_title("Bitcoin (BTC-USD) Daily Close Price — Full Dataset", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price (USD)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.legend()

ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(train_dates, y_train_inv, color="steelblue", linewidth=1, label="Actual (Train)")
ax2.plot(train_dates, train_pred,  color="orange",    linewidth=1, alpha=0.85, label="Predicted (Train)")
ax2.set_title(f"Training Set Predictions\nRMSE: ${train_rmse:,.0f}  |  MAE: ${train_mae:,.0f}", fontsize=12)
ax2.set_ylabel("Price (USD)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.tick_params(axis="x", rotation=30)
ax2.legend()

ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(test_dates, y_test_inv, color="steelblue", linewidth=1.2, label="Actual (Test)")
ax3.plot(test_dates, test_pred,  color="crimson",   linewidth=1.2, alpha=0.85, label="Predicted (Test)")
ax3.set_title(f"Test Set Predictions\nRMSE: ${test_rmse:,.0f}  |  MAE: ${test_mae:,.0f}", fontsize=12)
ax3.set_ylabel("Price (USD)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.tick_params(axis="x", rotation=30)
ax3.legend()

plt.suptitle("LSTM Model — BTC-USD Price Prediction", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("predictions_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: predictions_plot.png")

fig2, ax = plt.subplots(figsize=(9, 4))
ax.plot(history.history["loss"],     label="Training Loss",   color="steelblue")
ax.plot(history.history["val_loss"], label="Validation Loss", color="crimson", linestyle="--")
ax.set_title(f"Training Loss Curve  (final loss = {final_loss:.6f})", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: loss_curve.png")


# SAVE MODEL
model.save("lstm_btc_model.keras")
print("Model saved: lstm_btc_model.keras")

print("\nDone! Files created:")
print("  • lstm_project.py          (this script)")
print("  • predictions_plot.png     (train + test predictions)")
print("  • loss_curve.png           (training loss)")
print("  • lstm_btc_model.keras     (exported model)")
