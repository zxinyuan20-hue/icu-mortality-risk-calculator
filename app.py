"""Streamlit clinical calculator for the frozen GI perforation ICU mortality model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from scipy.special import expit, logit

# Importing this class before loading preprocessing.pkl resolves its stable pickle reference.
from model_artifacts import Winsorizer  # noqa: F401

APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "model"
if not MODEL_DIR.is_dir():
    MODEL_DIR = APP_ROOT
st.set_page_config(page_title="ICU Mortality Risk Calculator", page_icon="+", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_DIR / "lightgbm_final.pkl")
    preprocessing = joblib.load(MODEL_DIR / "preprocessing.pkl")
    metadata = json.loads((MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    recalibration = json.loads((MODEL_DIR / "recalibration_parameters.json").read_text(encoding="utf-8"))
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    return model, preprocessing, metadata, recalibration, explainer


def recalibrate_probability(probability: float, intercept: float, slope: float) -> float:
    return float(expit(intercept + slope * logit(np.clip(probability, 1e-6, 1 - 1e-6))))


def risk_category(probability: float, rules: dict) -> str:
    if probability < rules["low_upper"]:
        return "Low risk"
    if probability < rules["intermediate_upper"]:
        return "Intermediate risk"
    return "High risk"


def predict_patient(values: dict[str, float]):
    model, preprocessing, metadata, recalibration, explainer = load_artifacts()
    feature_columns = metadata["feature_columns"]
    input_frame = pd.DataFrame([[values[name] for name in feature_columns]], columns=feature_columns)
    transformed = preprocessing.transform(input_frame)
    original = float(model.predict_proba(transformed)[0, 1])
    recalibrated = recalibrate_probability(original, recalibration["calibration_intercept"], recalibration["calibration_slope"])
    raw_score = float(model.predict(transformed, raw_score=True)[0])
    explanation = explainer(transformed)
    shap_values = explanation.values[0]
    if not np.isclose(float(explanation.base_values[0] + shap_values.sum()), raw_score, atol=1e-6):
        raise RuntimeError("SHAP additivity check failed for this prediction.")
    return original, recalibrated, shap_values, metadata, recalibration


def contribution_rows(values: dict[str, float], shap_values: np.ndarray, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature, contribution in zip(metadata["feature_columns"], shap_values):
        rows.append({"Feature": metadata["display_names"][feature], "Input": values[feature], "SHAP value": float(contribution)})
    table = pd.DataFrame(rows)
    increasing = table.loc[table["SHAP value"] > 0].sort_values("SHAP value", ascending=False).head(5)
    protective = table.loc[table["SHAP value"] < 0].sort_values("SHAP value").head(5)
    return increasing, protective


def run_app() -> None:
    model, preprocessing, metadata, recalibration, explainer = load_artifacts()
    st.title("ICU Mortality Risk Calculator based on Explainable LightGBM Model")
    st.caption("Gastrointestinal perforation surgery ICU cohort. Research use only; not a substitute for clinical judgment.")

    with st.form("risk_calculator"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age (years)", min_value=18.0, max_value=120.0, value=65.0, step=1.0)
            sofa = st.number_input("SOFA score", min_value=0.0, max_value=24.0, value=5.0, step=1.0)
            lactate = st.number_input("Lactate (mmol/L)", min_value=0.1, max_value=30.0, value=2.0, step=0.1, format="%.1f")
            platelet = st.number_input("Platelet count (10^9/L)", min_value=1.0, max_value=2000.0, value=200.0, step=1.0)
        with right:
            bun = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=20.0, step=1.0)
            urine_output = st.number_input("Urine output (mL/24 h)", min_value=0.0, max_value=20000.0, value=1500.0, step=25.0)
            vasopressor = st.selectbox("Vasopressor use", options=["No", "Yes"], index=0)
            submitted = st.form_submit_button("Calculate ICU mortality risk", use_container_width=True)

    if submitted:
        input_values = {
            "Age": age, "Lactate": lactate, "Platelet": platelet, "Urine Output": urine_output,
            "BUN": bun, "SOFA": sofa, "Vasopressor": 1 if vasopressor == "Yes" else 0,
        }
        original, calibrated, shap_values, metadata, recalibration = predict_patient(input_values)
        category = risk_category(calibrated, metadata["risk_category_rules"]["recalibrated_probability"])
        result, original_metric = st.columns([2, 1])
        with result:
            st.metric("Predicted ICU mortality risk", f"{calibrated * 100:.1f}%")
            st.markdown(f"**Risk category:** {category}")
            st.caption("The displayed risk is the external-cohort recalibrated probability.")
        with original_metric:
            st.metric("Original model probability", f"{original * 100:.1f}%")
            st.caption("Frozen LightGBM output before probability recalibration.")

        with st.expander("Prediction explanation", expanded=True):
            increasing, protective = contribution_rows(input_values, shap_values, metadata)
            first, second = st.columns(2)
            with first:
                st.subheader("Top contributors increasing predicted risk")
                if increasing.empty:
                    st.write("No positive SHAP contributions for this profile.")
                else:
                    st.dataframe(increasing.style.format({"Input": "{:.2f}", "SHAP value": "{:+.3f}"}), use_container_width=True, hide_index=True)
            with second:
                st.subheader("Top protective contributors")
                if protective.empty:
                    st.write("No negative SHAP contributions for this profile.")
                else:
                    st.dataframe(protective.style.format({"Input": "{:.2f}", "SHAP value": "{:+.3f}"}), use_container_width=True, hide_index=True)
            st.caption("SHAP values are calculated on the LightGBM raw-score (log-odds) scale. Positive values increase, and negative values decrease, the original model output.")

    with st.expander("Model information and limitations"):
        st.markdown(
            "- Development cohort: MIMIC-III and MIMIC-IV; external validation cohort: eICU and NWICU.\n"
            "- The LightGBM model and feature set are frozen; this calculator does not retrain or update the base model.\n"
            "- The displayed calibrated probability applies an external-cohort logistic recalibration mapping and should be interpreted as a secondary transportability analysis.\n"
            "- Use only for adults undergoing surgery for gastrointestinal perforation who require ICU care; prospective clinical validation is required before clinical deployment."
        )


if __name__ == "__main__":
    run_app()
