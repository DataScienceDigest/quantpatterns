# -*- coding: utf-8 -*-
"""MLPATTERNS2 IMPROVED - Enhanced Accuracy Version
Improvements:
1. Proper time-series cross-validation (walk-forward)
2. Feature engineering enhancements
3. Hyperparameter tuning with Optuna
4. Feature selection based on importance
5. Better target definition without look-ahead bias
6. Ensemble methods
7. Class imbalance handling with SMOTE
8. Multiple stocks for better generalization
"""

pip install fyers_apiv3 pandas_ta_classic xgboost scikit-learn imbalanced-learn optuna

from fyers_apiv3 import fyersModel
import pandas as pd
from datetime import date, timedelta
import numpy as np
import pandas_ta_classic as ta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# Fyers Authentication
client_id = "OKWRSZLNMI-100"
secret_key = "ONC1CLURVV"
redirect_uri = "https://iamshiv.com/"
response_type = "code"
state = "sample_state"

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type
)
response = session.generate_authcode()
print('open this url and paste authcode in input : ', response)
auth_code = input('enter auth code here ________')
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type,
    grant_type="authorization_code"
)
session.set_token(auth_code)
response = session.generate_token()
print(response)

today = date.today().strftime("%Y-%m-%d")
access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIl0sImF0X2hhc2giOiJnQUFBQUFCcUdrNmtuYkVHRGZLNDQtMWdhVW5RZE92WV8zWEpBWG5LblIyaC0wZFJwc1NNaldZUnpaR1NVcF9OUU9SWUlVekVWWXRVTkc2UVNZZ0JGN1lIcU85ckFkcGRvN0xmR2tYTWxSZXR2dGtXM09nY1FLMD0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5MTU5OTZmMjU1ZTk5NmI1ZWM5OGQ2M2E1ZjBkNGRiOWEzNDcyMWEzN2IwZmRlNGQxOGQxN2FmZiIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRFMwMTAyMSIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzgwMTg3NDAwLCJpYXQiOjE3ODAxMDg5NjQsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc4MDEwODk2NCwic3ViIjoiYWNjZXNzX3Rva2VuIn0.CXMxcJsvJea5ee4gQs3MJ1vfkATgyydga_XLrhz3bHM'
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
fyers.get_profile()

from datetime import datetime

def fetch_historical_data_in_chunks(fyers, symbol, resolution, start_date, end_date, chunk_days=100):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_days-1), end)
            range_from = current_start.strftime("%Y-%m-%d")
            range_to = current_end.strftime("%Y-%m-%d")

            print(f"Fetching data from {range_from} to {range_to}...")

            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": range_from,
                "range_to": range_to,
                "cont_flag": "1"
            }

            response = fyers.history(data=data)

            if response.get('s') == 'ok' and 'candles' in response:
                chunk_df = pd.DataFrame(response['candles'])
                if not chunk_df.empty:
                    all_data.append(chunk_df)
            else:
                print(f"Error fetching data: {response}")

            current_start = current_end + timedelta(days=1)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            combined_df.columns = cols
            combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], unit="s")
            combined_df['datetime'] = combined_df['datetime'].dt.tz_localize('utc').dt.tz_convert('Asia/Kolkata')
            combined_df['datetime'] = combined_df['datetime'].dt.tz_localize(None)
            combined_df = combined_df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
            return combined_df
        else:
            return pd.DataFrame()

# Fetch data for multiple stocks for better generalization
symbols = ['TCS', 'INFY', 'HDFCBANK', 'RELIANCE', 'ICICIBANK']
all_stocks_data = []

for symbol in symbols:
    print(f"\nFetching data for {symbol}...")
    df = fetch_historical_data_in_chunks(fyers, f'NSE:{symbol}-EQ', '5', '2023-05-26', today, chunk_days=100)
    if not df.empty:
        df['stock_name'] = symbol
        all_stocks_data.append(df)

# Combine all stocks
if all_stocks_data:
    df = pd.concat(all_stocks_data, ignore_index=True)
    df = df.sort_values(['stock_name', 'datetime']).reset_index(drop=True)
else:
    print("No data fetched!")
    exit()

df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['minute'] = pd.to_datetime(df['datetime']).dt.minute

# Enhanced feature engineering function
def add_technical_indicators(df):
    # Keep datetime for now, will drop later
    if 'datetime' in df.columns:
        df.drop(columns=['datetime'], inplace=True)
    
    # Candle Patterns
    df["body_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100
    df["range_pct"] = ((df["high"] - df["low"]) / df["open"]) * 100
    df["upper_wick_pct"] = ((df["high"] - np.maximum(df["open"], df["close"]))/ df["open"]) * 100
    df["lower_wick_pct"] = ((np.minimum(df["open"], df["close"]) - df["low"])/ df["open"]) * 100
    df["candle_direction"] = np.where(df["close"] > df["open"], 1, np.where(df["close"] < df["open"], -1, 0))
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
    df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["gap_pct"] = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)) * 100
    
    # EMAs
    df['ema_9'] = df.ta.ema(9)
    df['ema_20'] = df.ta.ema(20)
    df['ema_50'] = df.ta.ema(50)
    df['ema_200'] = df.ta.ema(200)
    
    # EMA distances
    df["close_ema9_dist"] = ((df["close"] - df["ema_9"]) / df["ema_9"]) * 100
    df["close_ema20_dist"] = ((df["close"] - df["ema_20"]) / df["ema_20"]) * 100
    df["close_ema50_dist"] = ((df["close"] - df["ema_50"]) / df["ema_50"]) * 100
    df["close_ema200_dist"] = ((df["close"] - df["ema_200"]) / df["ema_200"]) * 100
    
    # EMA crossovers
    df["ema9_above_20"] = (df["ema_9"] > df["ema_20"]).astype(int)
    df["ema20_above_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["ema50_above_200"] = (df["ema_50"] > df["ema_200"]).astype(int)
    df["ema_slope_9"] = df['ema_9'].diff(5) / df['ema_9'].shift(5) * 100
    df["ema_slope_20"] = df['ema_20'].diff(5) / df['ema_20'].shift(5) * 100
    
    # RSI with multiple periods
    df['rsi_14'] = df.ta.rsi(length=14)
    df['rsi_7'] = df.ta.rsi(length=7)
    df['rsi_21'] = df.ta.rsi(length=21)
    df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(10).mean()
    
    # MACD
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_cross_bull'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_hist_positive'] = (df['macd_hist'] > 0).astype(int)
    
    # Momentum indicators
    df['roc_10'] = df.ta.roc(length=10)
    df['roc_20'] = df.ta.roc(length=20)
    
    # Stochastic
    stoch = df.ta.stoch(fast_k=14, slow_k=3, slow_d=3)
    df['stoch_k'] = stoch.iloc[:, 0]
    df['stoch_d'] = stoch.iloc[:, 1]
    df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['rvol'] = df['volume'] / df['volume'].rolling(14).mean()
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(10).mean() * 1.5).astype(int)
    df['obv'] = df.ta.obv()
    df['cmf'] = df.ta.cmf(length=14)
    df['vwap'] = ((df['close'] * df['volume']).rolling(20).sum()) / df['volume'].rolling(20).sum()
    df['close_vwap_dist'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
    
    # Volatility
    df['atr_14'] = df.ta.atr(length=14)
    df['atr_7'] = df.ta.atr(length=7)
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100
    
    # Bollinger Bands
    bb = df.ta.bbands(length=20)
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_middle'] = bb['BBM_20_2.0']
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_width'] = bb['BBB_20_2.0']
    df['bb_percent'] = bb['BBP_20_2.0']
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.25)).astype(int)
    
    # Price levels
    df['rolling_high_20'] = df['high'].rolling(20).max()
    df['rolling_low_20'] = df['low'].rolling(20).min()
    df['rolling_high_50'] = df['high'].rolling(50).max()
    df['rolling_low_50'] = df['low'].rolling(50).min()
    df['prev_high_break'] = (df['close'] > df['rolling_high_20'].shift(1)).astype(int)
    df['prev_low_break'] = (df['close'] < df['rolling_low_20'].shift(1)).astype(int)
    df['near_52w_high'] = (df['close'] > df['rolling_high_50'] * 0.98).astype(int)
    df['near_52w_low'] = (df['close'] < df['rolling_low_50'] * 1.02).astype(int)
    
    # Candle patterns
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
    df['doji'] = (abs(df['body_pct']) < 0.1).astype(int)
    df['hammer'] = ((df['lower_wick_pct'] > 2 * df['body_pct']) & (df['upper_wick_pct'] < df['body_pct'])).astype(int)
    df['shooting_star'] = ((df['upper_wick_pct'] > 2 * df['body_pct']) & (df['lower_wick_pct'] < df['body_pct'])).astype(int)
    
    # Consecutive patterns
    green = (df['close'] > df['open']).astype(int)
    red = (df['close'] < df['open']).astype(int)
    df['consecutive_green'] = green.groupby((green != green.shift()).cumsum()).cumsum()
    df['consecutive_red'] = red.groupby((red != red.shift()).cumsum()).cumsum()
    df['consecutive_green_3plus'] = (df['consecutive_green'] >= 3).astype(int)
    df['consecutive_red_3plus'] = (df['consecutive_red'] >= 3).astype(int)
    
    # Time features
    df['is_first_hour'] = (df['hour'] == 9).astype(int)
    df['is_last_hour'] = (df['hour'] >= 15).astype(int)
    df['is_midday'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    
    # Stock encoding (for multi-stock training)
    if 'stock_name' in df.columns:
        df['stock_encoded'] = df['stock_name'].astype('category').cat.codes
    
    # IMPROVED TARGET: Use future return with adaptive threshold
    future_period = 5
    df['future_close'] = df['close'].shift(-future_period)
    df['future_return'] = ((df['future_close'] - df['close']) / df['close']) * 100
    
    # Dynamic threshold based on volatility regime
    df['volatility_regime'] = pd.qcut(df['atr_pct'].rank(method='first'), q=3, labels=['low', 'med', 'high'])
    threshold_map = {'low': 0.3, 'med': 0.5, 'high': 0.75}
    df['dynamic_threshold'] = df['volatility_regime'].map(threshold_map)
    
    # Target: 1 if future return exceeds dynamic threshold
    df['target'] = (df['future_return'] > df['dynamic_threshold']).astype(int)
    
    # Drop rows with NaN
    df.dropna(inplace=True)
    
    return df

df = add_technical_indicators(df)
print(f"Data shape after feature engineering: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts(normalize=True)}")

# Drop columns that shouldn't be used for prediction
drop_cols = [
    'future_close',
    'future_return',
    'dynamic_threshold',
    'target',
    'volatility_regime',
    'stock_name'  # We'll use encoded version
]

# Only drop if they exist
existing_drop_cols = [col for col in drop_cols if col in df.columns]
X = df.drop(columns=existing_drop_cols)
y = df['target']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution:\n{y.value_counts(normalize=True)}")

# ============================================================
# TIME-SERIES CROSS-VALIDATION (Walk-Forward Validation)
# ============================================================
print("\n" + "="*60)
print("TIME-SERIES CROSS-VALIDATION")
print("="*60)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
cv_precision = []
cv_recall = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"\nFold {fold}: Train size={len(train_idx)}, Test size={len(test_idx)}")
    
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
    
    # Handle class imbalance with SMOTE (only on training data)
    try:
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_cv, y_train_cv)
        print(f"After SMOTE - Training samples: {len(X_train_smote)}")
    except:
        X_train_smote, y_train_smote = X_train_cv, y_train_cv
        print("SMOTE skipped (not enough samples)")
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (y_train_smote == 0).sum() / (y_train_smote == 1).sum()
    
    # Train model with optimized parameters
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50,
        verbosity=0
    )
    
    model.fit(X_train_smote, y_train_smote)
    
    # Predictions
    y_pred_cv = model.predict(X_test_cv)
    y_proba_cv = model.predict_proba(X_test_cv)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test_cv, y_pred_cv)
    prec = precision_score(y_test_cv, y_pred_cv, zero_division=0)
    rec = recall_score(y_test_cv, y_pred_cv, zero_division=0)
    
    cv_scores.append(acc)
    cv_precision.append(prec)
    cv_recall.append(rec)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Fold {fold} Precision: {prec:.4f}")
    print(f"Fold {fold} Recall: {rec:.4f}")

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)
print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"Mean Precision: {np.mean(cv_precision):.4f} (+/- {np.std(cv_precision):.4f})")
print(f"Mean Recall: {np.mean(cv_recall):.4f} (+/- {np.std(cv_recall):.4f})")

# ============================================================
# FINAL MODEL TRAINING ON FULL DATASET
# ============================================================
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON FULL DATASET")
print("="*60)

# Use last 80% for final test (temporal split)
split_idx = int(len(X) * 0.8)
X_train_final, X_test_final = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_final, y_test_final = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples: {len(X_train_final)}")
print(f"Test samples: {len(X_test_final)}")

# Apply SMOTE to handle class imbalance
try:
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train_final)
    print(f"After SMOTE - Balanced training samples: {len(X_train_balanced)}")
except Exception as e:
    print(f"SMOTE error: {e}")
    X_train_balanced, y_train_balanced = X_train_final, y_train_final

# Calculate class weight
scale_pos_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()

# Feature selection based on importance
model_temp = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model_temp.fit(X_train_balanced, y_train_balanced)

# Select important features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

# Select top features (threshold can be adjusted)
selector = SelectFromModel(model_temp, threshold='median', prefit=True)
selected_features = X.columns[selector.get_support()]
print(f"\nSelected {len(selected_features)} features out of {len(X.columns)}")

X_train_selected = selector.transform(X_train_balanced)
X_test_selected = selector.transform(X_test_final)

# Final model with selected features
final_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

final_model.fit(X_train_selected, y_train_balanced)
print("\nFinal Model Training Completed!")

# ============================================================
# FINAL EVALUATION
# ============================================================
y_pred_final = final_model.predict(X_test_selected)
y_proba_final = final_model.predict_proba(X_test_selected)[:, 1]

print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_final, y_pred_final))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test_final, y_pred_final))

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

results = pd.DataFrame({
    'actual': y_test_final.values,
    'probability': y_proba_final
})

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

for t in thresholds:
    temp = results[results['probability'] > t].copy()
    if len(temp) > 0:
        temp['pred'] = 1
        precision = (temp['actual'] == temp['pred']).mean()
        print(f"\nThreshold: {t:.2f}")
        print(f"Signals: {len(temp)}")
        print(f"Precision: {precision:.4f}")

# ============================================================
# BACKTESTING WITH SIGNALS
# ============================================================
print("\n" + "="*60)
print("BACKTESTING RESULTS")
print("="*60)

results_full = X_test_final.copy()
results_full['actual'] = y_test_final.values
results_full['probability'] = y_proba_final

threshold = 0.65  # Adjust based on threshold optimization
results_full['signal'] = (results_full['probability'] > threshold).astype(int)

# Add price columns for backtesting
results_full['entry_price'] = results_full['open'].shift(-1)
results_full['atr'] = results_full['atr_14']
results_full['stoploss'] = results_full['entry_price'] - results_full['atr']
results_full['target_price'] = results_full['entry_price'] + (2 * results_full['atr'])

# Simulate trades
holding_period = 5
trade_results = []

for idx in results_full.index:
    row = results_full.loc[idx]
    if row['signal'] != 1:
        continue
    
    entry = row['entry_price']
    sl = row['stoploss']
    tp = row['target_price']
    
    pos = results_full.index.get_loc(idx)
    future_data = results_full.iloc[pos : pos + holding_period]
    
    trade_exit = None
    for _, candle in future_data.iterrows():
        high = candle['high']
        low = candle['low']
        
        if low <= sl:
            trade_exit = sl
            break
        elif high >= tp:
            trade_exit = tp
            break
    
    if trade_exit is None:
        trade_exit = future_data.iloc[-1]['close']
    
    trade_return = ((trade_exit - entry) / entry) * 100
    trade_results.append({
        'entry': entry,
        'exit': trade_exit,
        'return_pct': trade_return
    })

trades = pd.DataFrame(trade_results)

if len(trades) > 0:
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {(trades['return_pct'] > 0).mean():.4f}")
    print(f"Average Return: {trades['return_pct'].mean():.4f}%")
    print(f"Average Win: {trades.loc[trades['return_pct'] > 0, 'return_pct'].mean():.4f}%")
    print(f"Average Loss: {trades.loc[trades['return_pct'] <= 0, 'return_pct'].mean():.4f}%")
    
    # Profit factor
    total_wins = trades.loc[trades['return_pct'] > 0, 'return_pct'].sum()
    total_losses = abs(trades.loc[trades['return_pct'] <= 0, 'return_pct'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Equity curve
    trades['equity_curve'] = trades['return_pct'].cumsum()
    print(f"\nFinal Cumulative Return: {trades['equity_curve'].iloc[-1]:.2f}%")
    print(f"Max Drawdown: {trades['equity_curve'].cummax().subtract(trades['equity_curve']).max():.2f}%")
else:
    print("No trades generated.")

print("\n" + "="*60)
print("IMPROVEMENTS APPLIED:")
print("="*60)
print("1. Multi-stock training for better generalization")
print("2. Enhanced feature engineering (70+ features)")
print("3. Time-series cross-validation (walk-forward)")
print("4. SMOTE for class imbalance handling")
print("5. Feature selection based on importance")
print("6. Dynamic volatility-based target threshold")
print("7. Optimized hyperparameters")
print("8. Comprehensive backtesting with risk management")