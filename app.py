"""
Streamlit app - Breast Cancer Wisconsin (Diagnostic) classification.

The user uploads labelled TEST data (CSV: 30 feature columns + 'target'),
picks one of the five pre-trained models from a dropdown, and the app shows
evaluation metrics, a confusion matrix and a classification report.

Models are loaded from the saved .pkl files in models/ - nothing is
retrained on the uploaded data; it is only used for prediction/evaluation.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
SAMPLE_DATA = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}
TARGET_COL = "target"
CLASS_NAMES = ["malignant", "benign"]  # 0 = malignant, 1 = benign

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


@st.cache_data
def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


def evaluate(model, X: pd.DataFrame, y_true: pd.Series):
    """Predict with an already-trained model and compute the 6 metrics (no fitting)."""
    y_pred = model.predict(X)
    # AUC needs a continuous score: class-1 probability, or the SVM's decision function.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


models = load_models()

# ---------------------------------------------------------------------------
# Sidebar: (a) test data upload, (b) model selection dropdown
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader(
    "CSV with all 30 feature columns and a 'target' column (0 = malignant, 1 = benign)",
    type="csv",
)
st.sidebar.download_button(
    "Download sample test data",
    data=SAMPLE_DATA.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
    help="The 114-row held-out test split (20%) that the models never saw during training.",
)

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES))

st.sidebar.caption(
    "All models were trained offline and are loaded from saved .pkl files. "
    "Uploaded data is used only for prediction and evaluation - never for retraining."
)

# ---------------------------------------------------------------------------
# Load and validate test data
# ---------------------------------------------------------------------------
st.title("🩺 Breast Cancer Diagnosis - Model Evaluation Dashboard")
st.caption(
    "Breast Cancer Wisconsin (Diagnostic) dataset, UCI ML Repository - classifying breast "
    "masses as **malignant (0)** or **benign (1)** from 30 cell-nucleus features."
)

if uploaded is not None:
    try:
        df = load_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read **{uploaded.name}** as a CSV file: {exc}")
        st.stop()
else:
    df = load_csv(SAMPLE_DATA)

feature_cols = list(next(iter(models.values())).feature_names_in_)
missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(
        f"The CSV is missing {len(missing)} required column(s): "
        f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}. "
        "The test file must contain all 30 input features and the 'target' column."
    )
    st.stop()

bad_labels = set(df[TARGET_COL].unique()) - {0, 1}
if bad_labels:
    st.error(f"'target' must contain only 0 (malignant) or 1 (benign); found {sorted(bad_labels)}.")
    st.stop()
non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
if non_numeric:
    st.error(f"These feature columns must be numeric: {', '.join(non_numeric[:6])}.")
    st.stop()
if df[feature_cols].isna().any().any():
    st.error("The feature columns contain missing values - please upload complete test data.")
    st.stop()

if uploaded is not None:
    st.success(f"Evaluating on uploaded file **{uploaded.name}** ({len(df)} rows)")
else:
    st.info(
        f"No file uploaded - evaluating on the bundled sample test data ({len(df)} rows). "
        "Upload your own test CSV from the sidebar."
    )

X_test = df[feature_cols]
y_test = df[TARGET_COL]

tab_eval, tab_compare, tab_data = st.tabs(
    ["Selected Model", "Compare All Models", "Test Data Preview"]
)

# ---------------------------------------------------------------------------
# (c) metrics and (d) confusion matrix + classification report for one model
# ---------------------------------------------------------------------------
with tab_eval:
    metrics, y_pred = evaluate(models[model_name], X_test, y_test)

    st.subheader(model_name)
    st.markdown("##### Evaluation metrics")
    for col, (k, v) in zip(st.columns(6), metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Confusion matrix")
        st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=[0, 1])))
    with right:
        st.markdown("##### Classification report")
        report = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    with st.expander("Row-level predictions"):
        out = df.copy()
        out.insert(0, "correct", y_pred == y_test)
        out.insert(0, "predicted", y_pred)
        out.insert(0, "actual", y_test)
        st.dataframe(out, width="stretch", height=300)
        st.download_button(
            "Download predictions",
            data=out.to_csv(index=False).encode(),
            file_name=f"predictions_{MODEL_FILES[model_name].removesuffix('.pkl')}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Results of all 5 saved models on the same test data
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("All 5 models on the current test data")
    comparison = pd.DataFrame(
        {name: evaluate(m, X_test, y_test)[0] for name, m in models.items()}
    ).T
    comparison.index.name = "Model"
    st.dataframe(
        comparison.style.highlight_max(axis=0, color="lightgreen").format("{:.4f}"),
        width="stretch",
    )
    best_mcc = comparison["MCC"].max()
    best = comparison.index[comparison["MCC"] == best_mcc].tolist()
    st.markdown(f"**Best by MCC:** {', '.join(best)} (MCC = {best_mcc:.4f})")
    metric_to_plot = st.radio(
        "Metric to chart", list(comparison.columns), index=5, horizontal=True
    )
    st.bar_chart(comparison[metric_to_plot])

with tab_data:
    st.subheader("Uploaded test data")
    st.write(f"{len(df)} rows x {len(feature_cols)} features + target")
    counts = y_test.value_counts().reindex([0, 1], fill_value=0)
    c1, c2 = st.columns(2)
    c1.metric("Malignant (0)", int(counts[0]))
    c2.metric("Benign (1)", int(counts[1]))
    st.dataframe(df, width="stretch", height=350)
