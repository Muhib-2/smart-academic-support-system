import os
import streamlit as st
import pandas as pd

from ml_model_updated import (
    train_and_save_models,
    load_model,
    predict_class,
    DEFAULT_DATASET_PATH,
)
from rule_engine_updated import AdvisingRuleEngine

# -------------------- Helpers --------------------
def read_csv_smart(file_or_path):
    """
    Reads CSV from uploaded file OR path.
    Auto-detects delimiter (',' or ';') to avoid the 1-column problem.
    """
    try:
        df = pd.read_csv(file_or_path)
        # If it became 1 column but header contains ';', re-read with sep=';'
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(file_or_path, sep=";")
        return df
    except Exception:
        # fallback
        return pd.read_csv(file_or_path, sep=";")

def derive_expert_facts_from_uci(student_row: dict):
    """
    Make the expert-system facts visible in the UI:
    - cgpa from G3 (0..20 -> 0..4)
    - attendance from absences
    - failed_courses from failures
    """
    facts = dict(student_row)

    # cgpa from G3
    g3 = facts.get("G3", 0)
    try:
        g3 = float(g3)
    except Exception:
        g3 = 0.0
    cgpa = round((g3 / 20.0) * 4.0, 2)

    # failed_courses from failures
    failures = facts.get("failures", 0)
    try:
        failures = int(failures)
    except Exception:
        failures = 0

    # attendance from absences (simple mapping)
    absences = facts.get("absences", 0)
    try:
        absences = float(absences)
    except Exception:
        absences = 0.0
    attendance = max(0.0, 100.0 - (absences * 2.0))

    return {
        "cgpa": cgpa,
        "attendance": round(attendance, 1),
        "failed_courses": failures,
        "G3": g3,
        "absences": absences,
        "failures": failures,
    }

# -------------------- UI --------------------
st.set_page_config(page_title="Smart Academic Support System", layout="centered")

st.title("🎓 Smart Academic Support System")
st.caption("✅ Part A: Machine Learning (KNN + MLP Classification)  |  ✅ Part B: Expert System (Rule-Based Advising)")

st.info(
    "Important: You MUST use a real existing dataset (UCI/Kaggle). "
    "Place it in data/student_performance.csv OR upload it here."
)

# ---------------- Dataset input ----------------
uploaded = st.file_uploader("Upload your dataset CSV (recommended)", type=["csv"])

if uploaded is not None:
    df = read_csv_smart(uploaded)
    st.success(f"Loaded uploaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    if os.path.exists(DEFAULT_DATASET_PATH):
        df = read_csv_smart(DEFAULT_DATASET_PATH)
        st.warning(f"Using default file found at: {DEFAULT_DATASET_PATH}")
    else:
        st.error("No dataset found. Upload a CSV OR put it at data/student_performance.csv")
        st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# ---------------- Train models ----------------
st.subheader("🧠 Train ML Models (KNN + MLP)")

train_clicked = st.button("Train & Save Model")
if train_clicked:
    os.makedirs("data", exist_ok=True)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)

    results = train_and_save_models(dataset_csv=DEFAULT_DATASET_PATH)
    st.success("Training completed. Models saved.")
    st.write("### Results")
    st.json(results)

# ---------------- Load models ----------------
try:
    knn_model, mlp_model, meta = load_model()
except Exception:
    st.warning("Models not trained yet. Click **Train & Save Model** first.")
    st.stop()

# ---------------- Pick a sample row for prediction ----------------
st.subheader("🔎 Predict Student Performance Category")

row_index = st.number_input(
    "Select student row index",
    min_value=0,
    max_value=max(len(df) - 1, 0),
    value=0,
    step=1
)

student_row = df.iloc[int(row_index)].to_dict()

# Show student information clearly
st.markdown("### 👤 Student Information")
st.write(f"**Student Number (Row Index):** {int(row_index)}")
st.dataframe(pd.DataFrame([student_row]).T.rename(columns={0: "value"}), use_container_width=True)

# Build input features: everything except the target column (meta.target_col)
input_features = dict(student_row)
if meta.target_col in input_features:
    input_features.pop(meta.target_col)

col1, col2 = st.columns(2)
with col1:
    st.write("**KNN Prediction**")
    knn_pred = predict_class(knn_model, input_features)["predicted_class"]
    st.success(knn_pred)

with col2:
    st.write("**MLP Prediction**")
    mlp_pred = predict_class(mlp_model, input_features)["predicted_class"]
    st.success(mlp_pred)

st.caption(f"Target column used during training: **{meta.target_col}** | Classes: {meta.class_names}")

# ---------------- Expert System Advising ----------------
st.subheader("🤖 Expert System Advising (Rule-Based)")

engine = AdvisingRuleEngine()

# Choose which prediction to use for rules
pred_for_rules = mlp_pred

# Show derived facts used by expert system
derived = derive_expert_facts_from_uci(student_row)
st.markdown("### 🧾 Facts Used By Expert System (Derived)")
c1, c2, c3 = st.columns(3)
c1.metric("CGPA (derived from G3)", derived["cgpa"])
c2.metric("Attendance % (derived)", derived["attendance"])
c3.metric("Failed Courses (from failures)", derived["failed_courses"])
st.caption(f"G3={derived['G3']} | absences={derived['absences']} | failures={derived['failures']}")

advising = engine.advise(student_profile=student_row, predicted_class=pred_for_rules)

st.write(f"**Predicted Category (used for rules):** {pred_for_rules}")
st.write(f"**Recommended Max Credits:** {advising['max_credits']}")

st.write("### ⚠️ Warnings")
if advising["warnings"]:
    for w in advising["warnings"]:
        st.warning(w)
else:
    st.success("No warnings.")

st.write("### ✅ Actions")
for a in advising["actions"]:
    st.write("- " + a)

st.write("### 🧩 Support / Interventions")
if advising["support"]:
    for s in advising["support"]:
        st.write("- " + s)
else:
    st.write("- None")

# Optional course recommendation
if advising["selected_courses"]:
    st.write("### 📚 Recommended Courses (optional)")
    for c in advising["selected_courses"]:
        st.write(f"- **{c['course_code']}** ({c['course_name']}) — {c['credits']} credits")
    st.write(f"**Total Credits Selected:** {advising['total_credits']}")

with st.expander("Show Forward Chaining Trace (Rules Fired)"):
    for t in advising["forward_chaining_trace"]:
        st.write("- " + t)

with st.expander("Show Backward Chaining Demonstration"):
    st.json(advising["backward_chaining"])
