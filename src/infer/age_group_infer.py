import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

RANDOM_STATE = 42

COLUMNS = [
    "gender",
    "city",
    "customer_segment",
    "product_category",
    "original_price",
    "discount_pct",
    "final_price",
    "quantity",
    "payment_method",
    "purchase_date",
    "purchase_hour",
    "is_weekend",
    "is_black_friday",
]
CLASS_COLUMN = "age_group"

NUMERIC_COLUMNS = [
    "original_price",
    "discount_pct",
    "final_price",
    "quantity",
    "purchase_month",
    "purchase_hour",
]
ORDINAL_COLUMNS = {"customer_segment": ["New", "Returning", "Loyal", "VIP"]}
ONEHOT_COLUMNS = ["gender", "city", "is_weekend", "is_black_friday", "product_category", "payment_method"]


MODELS_PATH = os.path.join(os.curdir, "models", CLASS_COLUMN)

MINMAX_MODEL_PATH = os.path.join(MODELS_PATH, "minmax.pkl")
ORDINAL_MODEL_PATH = os.path.join(MODELS_PATH, "ordinal.pkl")
ONEHOT_MODEL_PATH = os.path.join(MODELS_PATH, "onehot.pkl")

MODEL_PATH = os.path.join(MODELS_PATH, "model.pkl")

TEST_SIZE = 0.3


def main() -> None:
    os.makedirs(MODELS_PATH, exist_ok=True)

    X = pd.DataFrame(
        [
            [
                "Male",
                "San Francisco",
                "Loyal",
                "Footwear",
                153.73,
                35,
                99.92,
                1,
                "Credit Card",
                "2025-12-01",
                0,
                0,
                0,
            ]
        ],
        columns=COLUMNS,
    )
    X["purchase_month"] = pd.to_datetime(X["purchase_date"]).dt.month
    X = X.drop(columns=["purchase_date"])

    with open(MINMAX_MODEL_PATH, "rb") as f:
        min_max_scaler = pickle.load(f)

    X_numeric = pd.DataFrame(
        min_max_scaler.transform(X[NUMERIC_COLUMNS]), columns=NUMERIC_COLUMNS
    )

    with open(ORDINAL_MODEL_PATH, "rb") as f:
        ordinal_encoder = pickle.load(f)

    X_ordinal = pd.DataFrame(
        ordinal_encoder.transform(X[list(ORDINAL_COLUMNS.keys())]),
        columns=list(ORDINAL_COLUMNS.keys()),
    )

    with open(ONEHOT_MODEL_PATH, "rb") as f:
        one_hot_encoder = pickle.load(f)

    X_onehot = pd.DataFrame(
        one_hot_encoder.transform(X[ONEHOT_COLUMNS]),
        columns=one_hot_encoder.get_feature_names_out(ONEHOT_COLUMNS),
    )

    X_normalized = X_numeric.join(X_ordinal).join(X_onehot)

    print(X_normalized)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(X_normalized)
    probability = model.predict_proba(X_normalized)

    print(f"Predito: {prediction[0]}")
    print(f"Probabilidade: {np.max(probability[0]) * 100:0.2f}%")


if __name__ == "__main__":
    main()
