
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
DATA_FILE = "customer_churn.csv"
MODEL_FILE = "churn_model.pkl"


# 1. Load dataset or generate sample data
def load_data(n_samples=2000):
    if os.path.exists(DATA_FILE):
        print(f"Loading dataset: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print("CSV not found. Generating synthetic dataset.")

    rng = np.random.default_rng(RANDOM_STATE)

    gender = rng.choice(["Male", "Female"], n_samples)
    senior = rng.choice([0, 1], n_samples, p=[0.85, 0.15])
    tenure = rng.integers(0, 73, n_samples)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples, p=[0.55, 0.25, 0.20]
    )

    internet = rng.choice(
        ["DSL", "Fiber optic", "No"],
        n_samples, p=[0.35, 0.45, 0.20]
    )

    payment = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer", "Credit card"],
        n_samples
    )

    support = rng.choice(
        ["Yes", "No"], n_samples, p=[0.4, 0.6]
    )

    monthly = rng.normal(65, 25, n_samples).clip(18, 120).round(2)
    total = (
        monthly * tenure + rng.normal(0, 50, n_samples)
    ).clip(0).round(2)

    churn_prob = (
        0.45 * (contract == "Month-to-month")
        + 0.20 * (internet == "Fiber optic")
        + 0.15 * (support == "No")
        + 0.15 * (payment == "Electronic check")
        + 0.20 * (tenure < 12)
        + 0.10 * (monthly > 80)
        - 0.20 * (tenure > 48)
    )

    churn_prob = (
        (churn_prob - churn_prob.min())
        / (churn_prob.max() - churn_prob.min())
    )

    churn = rng.binomial(
        1, np.clip(churn_prob, 0.03, 0.9)
    )

    return pd.DataFrame({
        "Gender": gender,
        "SeniorCitizen": senior,
        "Tenure": tenure,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "TechSupport": support,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No")
    })


# 2. Explore dataset
def explore_data(df):
    print("\n--- Dataset Overview ---")
    print("Shape:", df.shape)
    print(df.head())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Churn Distribution ---")
    print(df["Churn"].value_counts())

    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig("churn_distribution.png")
    plt.show()
    plt.close()


# 3. Prepare features and target
def preprocess_data(df):
    df = df.copy()

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' column.")

    # Remove customer ID columns
    id_columns = [
        col for col in ["CustomerID", "customerID", "ID"]
        if col in df.columns
    ]
    df.drop(columns=id_columns, inplace=True)

    # Convert target into 0 and 1
    target = df["Churn"]

    if target.isna().any():
        raise ValueError("The Churn column contains missing values.")

    if target.dtype == object or isinstance(target.dtype, pd.StringDtype):
        normalized = target.astype(str).str.strip().str.lower()

        if set(normalized.unique()) <= {"yes", "no"}:
            y = normalized.map({"no": 0, "yes": 1})
        else:
            raise ValueError(
                "Churn must contain Yes/No or numeric 0/1 values."
            )
    else:
        y = pd.to_numeric(target, errors="raise")

        if y.isna().any() or not set(y.unique()) <= {0, 1}:
            raise ValueError("Churn must contain only 0 and 1.")

    X = df.drop(columns=["Churn"])

    if X.shape[1] == 0:
        raise ValueError("No input features found.")

    return X, y.astype(int)


# 4. Build preprocessing and machine learning pipeline
def build_pipeline(model, numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


# 5. Train models
def train_models(X_train, y_train, numeric_cols, categorical_cols):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    trained_models = {}

    for name, model in models.items():
        pipeline = build_pipeline(
            model, numeric_cols, categorical_cols
        )
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        print(f"{name} training completed.")

    return trained_models


# 6. Evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    plt.figure(figsize=(7, 5))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        auc = (
            roc_auc_score(y_test, y_prob)
            if len(np.unique(y_test)) == 2 else float("nan")
        )

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        }

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}")
        print(f"ROC-AUC  : {auc:.3f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0
        ))

        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()
    plt.close()

    comparison = pd.DataFrame(results).T.round(3)
    comparison.to_csv("model_comparison.csv")

    print("\n--- Model Comparison ---")
    print(comparison)

    comparison.plot(kind="bar", figsize=(9, 5))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
    plt.close()

    return results


# 7. Plot Random Forest feature importance
def plot_feature_importance(model):
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return

    names = preprocessor.get_feature_names_out()
    importance = pd.Series(
        classifier.feature_importances_, index=names
    ).sort_values(ascending=False).head(15)

    importance = importance.sort_values()

    importance.plot(kind="barh", figsize=(8, 6))
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    plt.close()


# 8. Predict a new customer
def predict_new_customer(model, customer_dict, feature_order):
    customer = pd.DataFrame(
        [customer_dict], columns=feature_order
    )

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1] * 100

    label = "Yes" if prediction == 1 else "No"

    return label, round(probability, 2)


# 9. Main program
def main():
    df = load_data()
    explore_data(df)

    X, y = preprocess_data(df)

    # Split before fitting preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    numeric_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    models = train_models(
        X_train, y_train, numeric_cols, categorical_cols
    )

    results = evaluate_models(models, X_test, y_test)

    best_name = max(
        results,
        key=lambda name: (
            results[name]["roc_auc"]
            if not np.isnan(results[name]["roc_auc"])
            else -1
        )
    )

    best_model = models[best_name]

    print(f"\nBest Model: {best_name}")

    plot_feature_importance(models["Random Forest"])

    # Save the complete pipeline
    joblib.dump(best_model, MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}")

    # Sample customer prediction
    sample_customer = {
        "Gender": "Female",
        "SeniorCitizen": 0,
        "Tenure": 5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

    # Ensure sample matches the dataset's feature columns
    sample_customer = {
        col: sample_customer[col] if col in sample_customer
        else X_train[col].mode().iloc[0]
        for col in X.columns
    }

    label, probability = predict_new_customer(
        best_model, sample_customer, list(X.columns)
    )

    print("\n--- Sample Customer Prediction ---")
    print(f"Churn Prediction: {label}")
    print(f"Predicted Churn Probability: {probability}%")


if __name__ == "__main__":
    main()
