import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/train/KDDTrain+.csv")

if "difficulty" in df.columns:
    df = df.drop("difficulty", axis=1)

df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

X = df.drop("label", axis=1)
y = df["label"]

categorical_cols = ["protocol_type", "service", "flag"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("XGBoost Results:")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "model/xgboost.pkl")
print("Model saved to model/xgboost.pkl")
