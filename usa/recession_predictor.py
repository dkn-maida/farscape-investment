import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Parameters
N = 12
drop_threshold = 0.16
weeks_per_year = 52
time_steps = N  # Use the same horizon for LSTM sequences

# Load macro data
yield_curve_file = 'YieldCurveSpread_weekly.csv'
pmi_file = 'pmi_weekly.csv'

yield_curve_data = pd.read_csv(yield_curve_file, parse_dates=['time'])
yield_curve_data.set_index('time', inplace=True)

pmi_data = pd.read_csv(pmi_file, parse_dates=['time'])
pmi_data.set_index('time', inplace=True)

# Download S&P 500 data
spy_data = yf.download("^GSPC", period="max", interval="1wk")

# Handle multi-level columns if present
if isinstance(spy_data.columns, pd.MultiIndex):
    if '^GSPC' in spy_data.columns.levels[1]:
        spy_data = spy_data.xs('^GSPC', level=1, axis=1)
    else:
        print("MultiIndex columns:", spy_data.columns)
        raise ValueError("Could not find '^GSPC' level in multi-index columns.")

# Ensure 'Close' column is present
if 'Close' not in spy_data.columns:
    print("Columns after selection:", spy_data.columns)
    raise ValueError("Could not find a 'Close' column.")

# Align all datasets to a common start date
common_start = max(spy_data.index.min(), yield_curve_data.index.min(), pmi_data.index.min())
spy_data = spy_data[spy_data.index >= common_start]
yield_curve_data = yield_curve_data[yield_curve_data.index >= common_start]
pmi_data = pmi_data[pmi_data.index >= common_start]

# Compute intermediate values (not final features)
spy_data['Log Close'] = np.log(spy_data['Close'])
spy_data['Weekly Returns'] = spy_data['Log Close'].diff()

spy_data['3M Volatility'] = (
    spy_data['Weekly Returns'].rolling(window=12, min_periods=1).std() * np.sqrt(weeks_per_year)
)
spy_data['1M Volatility'] = (
    spy_data['Weekly Returns'].rolling(window=4, min_periods=1).std() * np.sqrt(weeks_per_year)
)

# Compute rolling average returns
period_range = range(52, 3, -1)
rolling_avg_returns_list = []
for period in period_range:
    rolling_return = spy_data['Weekly Returns'].rolling(window=period, min_periods=1).mean()
    rolling_avg_returns_list.append(rolling_return)
rolling_avg_returns = pd.concat(rolling_avg_returns_list, axis=1).mean(axis=1)

# Create labels for downturn in the next N weeks
prices = spy_data['Close'].values
labels = []
for i in range(len(prices)):
    if i + N < len(prices):
        current_price = prices[i]
        future_min = prices[i+1:i+N+1].min()
        if future_min <= current_price * (1 - drop_threshold):
            labels.append(1)
        else:
            labels.append(0)
    else:
        labels.append(np.nan)

labels = pd.Series(labels, index=spy_data.index, name='Label')
spy_data['Label'] = labels
spy_data.dropna(subset=['Label'], inplace=True)

# Create final features DataFrame
features = pd.DataFrame(index=spy_data.index)
features['3M Vol'] = spy_data['3M Volatility']
features['1M Vol'] = spy_data['1M Volatility']
features['Avg Rolling Returns'] = rolling_avg_returns.reindex(features.index)

# Reindex macro data
yield_curve_data = yield_curve_data.reindex(features.index, method='ffill')
pmi_data = pmi_data.reindex(features.index, method='ffill')

# Join yield curve and PMI data
features = features.join(yield_curve_data[['close']], how='left')
features = features.join(pmi_data[['close']], how='left', rsuffix='_pmi')

# Rename columns
features.rename(columns={'close': 'Yield_Curve', 'close_pmi': 'PMI'}, inplace=True)

features = features.join(spy_data['Label'])
features.dropna(inplace=True)

X = features.drop('Label', axis=1)
y = features['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check class imbalance
num_positive = sum(y_train == 1)
num_negative = sum(y_train == 0)
print(f"Training set: {num_negative} negative, {num_positive} positive samples.")

# Compute scale_pos_weight for imbalance in XGBoost
scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1

#####################
# Using XGBoost with scale_pos_weight and GridSearch
#####################
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [50,100],
    'max_depth': [3, 100],
    'learning_rate': [0.1, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters found by GridSearchCV:", grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred_xgb = best_model.predict(X_test)
y_proba_xgb = best_model.predict_proba(X_test)[:,1]

print("=== XGBoost Classifier (Tuned) ===")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))

#####################
# Logistic Regression with class_weight='balanced'
#####################
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:,1]

print("\n=== Logistic Regression (Balanced) ===")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

#####################
# Random Forest with class_weight='balanced'
#####################
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:,1]

print("\n=== Random Forest (Balanced) ===")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

# Feature importance (XGBoost)
importances = best_model.feature_importances_
feature_names = X_train.columns
feat_imp_df = pd.DataFrame({'Features': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Features'], feat_imp_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Plot the S&P 500 Close Price and highlight points with Label=1
plt.figure(figsize=(12,6))
plt.plot(spy_data.index, spy_data['Close'], label='S&P 500 Close', color='blue')

downturn_points = spy_data[spy_data['Label'] == 1]
plt.scatter(downturn_points.index, downturn_points['Close'], color='red', label='Downturn Label = 1', zorder=5)

plt.title("S&P 500 with Downturn Labels")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plot the Label values over time separately
plt.figure(figsize=(12,3))
plt.plot(spy_data.index, spy_data['Label'], 'o', label='Label', markersize=3)
plt.title("Label Over Time")
plt.xlabel("Date")
plt.ylabel("Label (0 or 1)")
plt.grid(True)
plt.show()

#####################
# LSTM Implementation
#####################

# Convert data into sequences for LSTM
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i+time_steps)])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

X_all = X.values
y_all = y.values

X_seq, y_seq = create_sequences(X_all, y_all, time_steps=time_steps)

# Split sequences
seq_train_size = int(len(X_seq)*0.7)
X_seq_train, X_seq_test = X_seq[:seq_train_size], X_seq[seq_train_size:]
y_seq_train, y_seq_test = y_seq[:seq_train_size], y_seq[seq_train_size:]

# Compute class weights for LSTM
classes = np.unique(y_seq_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_seq_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(time_steps, X_seq_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train LSTM model
history = model.fit(
    X_seq_train, y_seq_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate LSTM
y_seq_proba = model.predict(X_seq_test)
y_seq_pred = (y_seq_proba >= 0.5).astype(int)

print("\n=== LSTM Model ===")
print(classification_report(y_seq_test, y_seq_pred))
print("ROC-AUC:", roc_auc_score(y_seq_test, y_seq_proba))

# Plot LSTM training history (Optional)
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
