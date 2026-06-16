"""Streamlit app for diabetes risk prediction."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Prédiction du diabète", page_icon="🩺", layout="centered")

st.title("Prédiction du risque de diabète")
st.markdown(
    "Application de démonstration basée sur le projet Machine Learning "
    "(Pima Indians Diabetes — classification binaire)."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Modèle introuvable. Exécutez d'abord `notebooks/final_project.ipynb`.")
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

with st.form("prediction_form"):
    st.subheader("Mesures cliniques")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Grossesses", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Pression artérielle (mmHg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Épaisseur de la peau (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insuline (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("IMC (BMI)", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Âge", min_value=21, max_value=100, value=33)

    submitted = st.form_submit_button("Prédire")

if submitted:
    raw = pd.DataFrame(
        [{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }]
    )

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from preprocessing import add_engineered_features, get_feature_columns

    features = add_engineered_features(raw)[get_feature_columns(include_engineered=True)]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"Risque de diabète détecté — probabilité : {probability:.1%}")
        st.info("Consultez un professionnel de santé pour un diagnostic confirmé.")
    else:
        st.success(f"Pas de diabète prédit — probabilité : {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0), text="Score de risque")

st.caption("Projet ML — Emna Bouharb | Outil pédagogique, pas un diagnostic médical.")
