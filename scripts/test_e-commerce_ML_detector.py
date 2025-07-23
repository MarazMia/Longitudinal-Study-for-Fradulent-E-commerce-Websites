import pandas as pd
import json
import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- Config ---
DATA_PATH = "scripts/ecommerce_feature_dataset_with_keywords.csv"
MODEL_PATH = "scripts/xgb_ecommerce_detector_voting.pkl"
FP_PATH = "scripts/ML_false_positives.json"
FN_PATH = "scripts/ML_false_negatives.json"
SCALER_PATH = "scripts/feature_scaler.pkl"

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)
print(df['label'].value_counts())
X = df.drop(columns=['label', 'url'])
y = df['label']
urls = df['url']

# Save feature column order
with open("scripts/feature_column_order.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# --- Train-test split ---
X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
    X, y, urls, test_size=0.2, stratify=y, random_state=42
)

# --- Normalize features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# --- Compute class weights ---
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
weights_dict = {0: class_weights[0], 1: class_weights[1]}

# --- Define base models ---
xgb_model = xgb.XGBClassifier(
    subsample=0.6,
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    eval_metric='logloss',
    random_state=42
)

lgb_model = lgb.LGBMClassifier(
    num_leaves=50,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    verbose=-1,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    class_weight=weights_dict,
    random_state=42
)

log_reg = LogisticRegression(
    max_iter=500,
    class_weight=weights_dict,
    random_state=42
)

# --- Voting Classifier (soft voting) ---
voting_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model),
        ('lr', log_reg)
    ],
    voting='soft',  # use soft voting to average predicted probabilities
    n_jobs=-1
)

# --- Train and evaluate ---
voting_model.fit(X_train_scaled, y_train)
y_pred = voting_model.predict(X_test_scaled)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# --- Save model ---
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(voting_model, f)

print(f"✅ Model saved to {MODEL_PATH}")

# --- Save false positive and false negative URLs ---
false_positives = urls_test[(y_test == 0) & (y_pred == 1)].tolist()
false_negatives = urls_test[(y_test == 1) & (y_pred == 0)].tolist()

with open(FP_PATH, 'w') as f:
    json.dump(false_positives, f, indent=2)
with open(FN_PATH, 'w') as f:
    json.dump(false_negatives, f, indent=2)

print(f"🟥 Saved {len(false_positives)} false positives to {FP_PATH}")
print(f"🟦 Saved {len(false_negatives)} false negatives to {FN_PATH}")
