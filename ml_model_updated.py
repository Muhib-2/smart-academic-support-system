import os
import joblib
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# ============================================================
# ✅ LECTURER REQUIREMENT (PART A - ML)
# - Classification ONLY
# - Use EXISTING dataset (UCI/Kaggle)
# - Implement TWO models only: KNN + MLP
# ============================================================

KNN_MODEL_PATH = os.path.join("data", "knn_model.joblib")
MLP_MODEL_PATH = os.path.join("data", "mlp_model.joblib")
META_PATH = os.path.join("data", "ml_meta.joblib")

# Default dataset path (you MUST replace this file with a real dataset you downloaded)
DEFAULT_DATASET_PATH = os.path.join("data", "student_performance.csv")

# Target column (we try to auto-detect if not found)
DEFAULT_TARGET_COL = "grade_category"


@dataclass
class MLMetadata:
    features: List[str]
    target_col: str
    class_names: List[str]


def _derive_grade_category_if_needed(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    If your real dataset has final score columns (common in UCI Student Performance),
    we derive a classification label WITHOUT inventing new data:
      - If 'G3' exists (0..20), create grade_category:
           A: >= 16, B: 14-15, C: 10-13, D: < 10
    """
    if DEFAULT_TARGET_COL in df.columns:
        return df, DEFAULT_TARGET_COL

    if "G3" in df.columns:
        def cat(g):
            try:
                g = float(g)
            except Exception:
                return "Unknown"
            if g >= 16:
                return "A"
            if g >= 14:
                return "B"
            if g >= 10:
                return "C"
            return "D"
        df = df.copy()
        df[DEFAULT_TARGET_COL] = df["G3"].apply(cat)
        return df, DEFAULT_TARGET_COL

    # Pass/Fail derivation if only numeric score exists
    if "final_score" in df.columns:
        df = df.copy()
        df["pass_fail"] = df["final_score"].apply(lambda x: "Pass" if float(x) >= 50 else "Fail")
        return df, "pass_fail"

    # Otherwise: user must set a target column in their dataset
    return df, ""


def _build_preprocess_pipeline(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    # Identify numeric/categorical columns automatically
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocess, (numeric_cols + categorical_cols)


def train_and_save_models(
    dataset_csv: str = DEFAULT_DATASET_PATH,
    target_col: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """
    Trains and saves BOTH required models (KNN + MLP) and returns evaluation results.
    """
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(
            f"❌ Dataset not found: {dataset_csv}\n"
            f"✅ Put your real dataset CSV at: {DEFAULT_DATASET_PATH} (or pass dataset_csv=...)"
        )

    df = pd.read_csv(dataset_csv)

    # Auto-create/auto-detect target for common student datasets
    df, derived_target = _derive_grade_category_if_needed(df)

    chosen_target = target_col or derived_target or DEFAULT_TARGET_COL
    if chosen_target not in df.columns:
        raise ValueError(
            f"❌ Target column '{chosen_target}' not found in dataset.\n"
            f"✅ Fix: set target_col=... to a real label column in your CSV.\n"
            f"Dataset columns: {list(df.columns)}"
        )

    # Drop rows with missing target
    df = df.dropna(subset=[chosen_target])

    # Features = everything except target
    X = df.drop(columns=[chosen_target])
    y = df[chosen_target].astype(str)

    preprocess, feature_list = _build_preprocess_pipeline(X)

    # Split (stratified when possible)
    stratify = y if y.nunique() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # ✅ Model 1: KNN
    knn_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ])

    # ✅ Model 2: MLP (replace hyperparams with lecturer's code settings if provided)
    mlp_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=400,
            random_state=random_state,
        )),
    ])

    results = {}

    for name, model, path in [
        ("KNN", knn_pipe, KNN_MODEL_PATH),
        ("MLP", mlp_pipe, MLP_MODEL_PATH),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Multi-class friendly metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
        report = classification_report(y_test, preds, zero_division=0)

        results[name] = {
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "confusion_matrix": cm.tolist(),
            "labels": sorted(y.unique()),
            "classification_report": report,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    # Save metadata (so app knows features/labels)
    meta = MLMetadata(
        features=feature_list,
        target_col=chosen_target,
        class_names=sorted(y.unique()),
    )
    joblib.dump(meta, META_PATH)

    return results


def load_model() -> Tuple[object, object, MLMetadata]:
    if not (os.path.exists(KNN_MODEL_PATH) and os.path.exists(MLP_MODEL_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            "❌ Models not found.\n"
            "✅ Run training first: python main.py (or train_and_save_models())"
        )
    knn = joblib.load(KNN_MODEL_PATH)
    mlp = joblib.load(MLP_MODEL_PATH)
    meta = joblib.load(META_PATH)
    return knn, mlp, meta


def predict_class(model, input_features: dict) -> Dict[str, str]:
    X = pd.DataFrame([input_features])
    pred = model.predict(X)[0]
    return {"predicted_class": str(pred)}

if __name__ == "__main__":
    results = train_and_save_models()
    for model_name, metrics in results.items():
        print(f"=== {model_name} Evaluation ===")
        for k, v in metrics.items():
            if k == "confusion_matrix":
                print(f"{k}:\n{pd.DataFrame(v, index=metrics['labels'], columns=metrics['labels'])}")
            else:
                print(f"{k}: {v}")
        print("\n")
        