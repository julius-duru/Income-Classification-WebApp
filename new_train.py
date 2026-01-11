import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("income_evaluation.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df.columns = df.columns.str.strip().str.replace("-", "_")
df = df.replace("?", "Unknown")

# Binary target encoding
df["income"] = df["income"].astype(str).str.strip()

df["income"] = df["income"].replace({
    "<=50K": 0,
    "<=50K.": 0,
    "<50K": 0,
    ">50K": 1,
    ">50K.": 1,
    ">=50K": 1
})

df.drop_duplicates(inplace=True)

# -----------------------------
# Feature Selection (MATCHES APP)
# -----------------------------
features = [
    "age",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hours_per_week",
    "capital_gain",
    "capital_loss",
    "native_country"
]

X = df[features]
y = df["income"]

# -----------------------------
# Column Types
# -----------------------------
num_features = [
    "age",
    "hours_per_week",
    "capital_gain",
    "capital_loss"
]

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"
]

# -----------------------------
# Preprocessor
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Train
# -----------------------------
X_processed = preprocessor.fit_transform(X)
model.fit(X_processed, y)

# -----------------------------
# Save Artifacts
# -----------------------------
joblib.dump(model, "income_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ income_model.pkl and preprocessor.pkl saved successfully")
