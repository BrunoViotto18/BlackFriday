import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, BaggingClassifier


RANDOM_STATE = 42

DROP_COLUMNS = [
    "transaction_id",
    "customer_id",
    "product_id",
    "age_group",
    "purchase_amount",
]
CLASS_COLUMN = "payment_method"

NUMERIC_COLUMNS = [
    "original_price",
    "discount_pct",
    "final_price",
    "quantity",
    "purchase_month",
    "purchase_hour",
]
ORDINAL_COLUMNS = {
    "customer_segment": ["New", "Returning", "Loyal", "VIP"]
}
ONEHOT_COLUMNS = ["gender", "city", "is_weekend", "is_black_friday", "product_category"]

MODELS_PATH = os.path.join(os.curdir, "models", CLASS_COLUMN)

MINMAX_MODEL_PATH = os.path.join(MODELS_PATH, "minmax.pkl")
ORDINAL_MODEL_PATH = os.path.join(MODELS_PATH, "ordinal.pkl")
ONEHOT_MODEL_PATH = os.path.join(MODELS_PATH, "onehot.pkl")

MODEL_PATH = os.path.join(MODELS_PATH, "model.pkl")

TEST_SIZE = 0.3


def main() -> None:
    os.makedirs(MODELS_PATH, exist_ok=True)
    dataset_path = kagglehub.dataset_download("noopurbhatt/retail-black-friday-sales-dataset")

    csv_file = os.listdir(dataset_path)[0]
    csv_path = os.path.join(dataset_path, csv_file)

    df = pd.read_csv(csv_path).drop(columns=DROP_COLUMNS)

    X = df.drop(columns=[CLASS_COLUMN])
    X["purchase_month"] = pd.to_datetime(df["purchase_date"]).dt.month
    X = X.drop(columns=["purchase_date"])

    Y = df[CLASS_COLUMN].squeeze()

    min_max_scaler = MinMaxScaler()
    X_numeric = pd.DataFrame(
        min_max_scaler.fit_transform(X[NUMERIC_COLUMNS]),
        columns=NUMERIC_COLUMNS
    )

    with open(MINMAX_MODEL_PATH, "wb") as f:
        pickle.dump(min_max_scaler, f)

    ordinal_encoder = OrdinalEncoder(categories=list(ORDINAL_COLUMNS.values()))
    X_ordinal = pd.DataFrame(
        ordinal_encoder.fit_transform(X[list(ORDINAL_COLUMNS.keys())]),
        columns=list(ORDINAL_COLUMNS.keys()),
    )

    with open(ORDINAL_MODEL_PATH, "wb") as f:
        pickle.dump(ordinal_encoder, f)

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    X_onehot = pd.DataFrame(
        one_hot_encoder.fit_transform(X[ONEHOT_COLUMNS]),
        columns=one_hot_encoder.get_feature_names_out(ONEHOT_COLUMNS),
    )

    with open(ONEHOT_MODEL_PATH, "wb") as f:
        pickle.dump(one_hot_encoder, f)

    X_normalized = X_numeric.join(X_ordinal).join(X_onehot)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_normalized, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    balancer = SMOTE(random_state=RANDOM_STATE)
    X_balanced, Y_balanced = balancer.fit_resample(X_train, Y_train)

    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
    criterion = ["gini", "entropy"]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
    max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
    max_features = ["sqrt", "log2"]

    forest_parameter_map = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "max_depth": max_depth,
        "max_features": max_features,
    }

    models = [
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            forest_parameter_map,
            MODEL_PATH,
        )
    ]

    from pprint import pprint

    for model_name, model, parameter_map, model_path in models:

        print(f"Treinando modelo: {model_name}")

        hyper_parameters = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_map,
            n_iter=10,
            cv=3,
            n_jobs=1,
            verbose=1,
        )

        hyper_parameters.fit(X_balanced, Y_balanced)

        print("Melhores parametros:")
        pprint(hyper_parameters.best_params_)

        model = hyper_parameters.best_estimator_

        model = model.fit(X_balanced, Y_balanced)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        Y_predict = model.predict(X_test)

        labels = np.sort(Y_test.unique())
        cm = confusion_matrix(Y_test, Y_predict, labels=labels)

        accuracy = accuracy_score(Y_test, Y_predict)
        print(f"Acurácia global: {accuracy * 100:.2f}%")

        total = cm.sum()

        print("Métricas por classe:")
        for i, label in enumerate(labels):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = total - TP - FN - FP

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

            print(f"Classe {label}:")
            print(f"  Sensibilidade: {sensitivity * 100:.2f}%")
            print(f"  Especificidade: {specificity * 100:.2f}%")

        f1_micro = f1_score(Y_test, Y_predict, average="micro")
        print(f"F1-score micro: {f1_micro * 100:.2f}%")

        f1_macro = f1_score(Y_test, Y_predict, average="macro")
        print(f"F1-score macro: {f1_macro * 100:.2f}%")

        f1_weighted = f1_score(Y_test, Y_predict, average="weighted")
        print(f"F1-score weighted: {f1_weighted * 100:.2f}%")

        ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)
        plt.show()


if __name__ == "__main__":
    main()
