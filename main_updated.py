import os
import pandas as pd

from ml_model_updated import (
    train_and_save_models,
    load_models,
    predict_class,
    DEFAULT_DATASET_PATH,
)
from rule_engine_updated import AdvisingRuleEngine


def main():
    print("=== Smart Academic Support System (CLI) ===")
    print("Part A: ML Classification (KNN + MLP)")
    print("Part B: Expert System Advising")

    if not os.path.exists(DEFAULT_DATASET_PATH):
        raise FileNotFoundError(
            f"❌ Dataset not found: {DEFAULT_DATASET_PATH}\n"
            "✅ Put your real dataset CSV at data/student_performance.csv"
        )

    # Train if models not found
    try:
        knn, mlp, meta = load_models()
    except Exception:
        print("Models not found. Training now...")
        results = train_and_save_models(dataset_csv=DEFAULT_DATASET_PATH)
        print("Training done. Results:")
        print(results)
        knn, mlp, meta = load_models()

    df = pd.read_csv(DEFAULT_DATASET_PATH)

    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Target column: {meta.target_col}")
    print(f"Classes: {meta.class_names}")

    # Pick a row to test (simple)
    idx = int(input(f"Select a row index (0..{len(df)-1}): ").strip())
    row = df.iloc[idx].to_dict()

    # Prepare features
    features = dict(row)
    if meta.target_col in features:
        features.pop(meta.target_col)

    knn_pred = predict_class(knn, features)["predicted_class"]
    mlp_pred = predict_class(mlp, features)["predicted_class"]

    print("\n=== ML Predictions ===")
    print("KNN Predicted Class:", knn_pred)
    print("MLP Predicted Class:", mlp_pred)

    # Expert system advising uses MLP prediction (default)
    engine = AdvisingRuleEngine()
    advising = engine.advise(student_profile=row, predicted_class=mlp_pred)

    print("\n=== Expert System Advice ===")
    print("Predicted Class (used for rules):", mlp_pred)
    print("Recommended Max Credits:", advising["max_credits"])

    print("\nWarnings:")
    if advising["warnings"]:
        for w in advising["warnings"]:
            print("-", w)
    else:
        print("- None")

    print("\nActions:")
    for a in advising["actions"]:
        print("-", a)

    print("\nSupport/Interventions:")
    for s in advising["support"]:
        print("-", s)

    print("\nForward chaining trace:")
    for t in advising["forward_chaining_trace"]:
        print("-", t)

    print("\nBackward chaining demo (reduce_credits):")
    bc = advising["backward_chaining"]["reduce_credits"]
    print("Result:", bc["result"])
    for t in bc["trace"]:
        print("-", t)


if __name__ == "__main__":
    main()
