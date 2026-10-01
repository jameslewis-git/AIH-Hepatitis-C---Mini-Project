"""
Train and save the Hepatitis C staging model.
Dataset: UCI HCV dataset (data/hcvdat0.csv)
Target classes: 0=Blood Donor, 0s=suspect Blood Donor, 1=Hepatitis, 2=Fibrosis, 3=Cirrhosis
Run: python train_model.py
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = "data/hcvdat0.csv"
MODEL_PATH = "model/hcv_model.pkl"

FEATURE_COLS = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL",
                "CHE", "CHOL", "CREA", "GGT", "PROT"]

CLASS_LABELS = {
    0: "Blood Donor",
    1: "Suspect Blood Donor",
    2: "Hepatitis",
    3: "Fibrosis",
    4: "Cirrhosis",
}


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path, index_col=0)
    # Normalize target labels: "0=Blood Donor" -> 0, "0s=suspect Blood Donor" -> 1, etc.
    category_map = {
        "0=Blood Donor": 0,
        "0s=suspect Blood Donor": 1,
        "1=Hepatitis": 2,
        "2=Fibrosis": 3,
        "3=Cirrhosis": 4,
    }
    df["Category"] = df["Category"].map(category_map)
    df["Sex"] = df["Sex"].map({"m": 1, "f": 0})
    return df


def build_pipeline_data(df):
    X = df[FEATURE_COLS].copy()
    y = df["Category"].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=FEATURE_COLS)

    return X_scaled, y, imputer, scaler


def main():
    df = load_and_clean()
    X, y, imputer, scaler = build_pipeline_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_f1 = f1_score(y_test, rf.predict(X_test), average="macro")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=RANDOM_STATE, eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train)
    xgb_f1 = f1_score(y_test, xgb.predict(X_test), average="macro")

    print("Random Forest macro-F1:", round(rf_f1, 4))
    print("XGBoost macro-F1:", round(xgb_f1, 4))

    best_model, best_name = (rf, "RandomForest") if rf_f1 >= xgb_f1 else (xgb, "XGBoost")
    print(f"\nSelected best model: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=list(CLASS_LABELS.values())))

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "class_labels": CLASS_LABELS,
        "accuracy": accuracy_score(y_test, best_model.predict(X_test)),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel bundle saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
