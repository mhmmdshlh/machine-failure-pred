import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Prediksi Kerusakan Mesin", layout="centered")

st.title("Prediksi Kerusakan Mesin")
st.markdown("Masukkan parameter mesin untuk memprediksi kemungkinan kerusakan.")

scaler = joblib.load("models/scaler.joblib")
rf = joblib.load("models/random_forest_model.joblib")
xgb = joblib.load("models/xgboost_model.joblib")
ann = keras.models.load_model("models/ann_model.keras")

type_map = {"L": 1.0, "M": 2.0, "H": 3.0}

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_type = st.selectbox("Tipe Produk", options=["L", "M", "H"])
        air_temp = st.number_input("Suhu Udara [K]", min_value=290.0, max_value=320.0, value=300.0, step=0.1)
        process_temp = st.number_input("Suhu Proses [K]", min_value=290.0, max_value=330.0, value=310.0, step=0.1)

    with col2:
        rot_speed = st.number_input("Kecepatan Rotasi [rpm]", min_value=1000.0, max_value=3000.0, value=1500.0, step=1.0)
        torque = st.number_input("Torsi [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        tool_wear = st.number_input("Keausan Pahat [min]", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

if submitted:
    features = np.array([[type_map[product_type], air_temp, process_temp, rot_speed, torque, tool_wear]])
    features_scaled = scaler.transform(features)

    rf_pred = rf.predict(features_scaled)[0]
    rf_prob = rf.predict_proba(features_scaled)[0]

    xgb_pred = xgb.predict(features_scaled)[0]
    xgb_prob = xgb.predict_proba(features_scaled)[0]

    ann_prob = ann.predict(features_scaled, verbose=0)[0][0]
    ann_pred = 1 if ann_prob >= 0.5 else 0

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col_a, col_b, col_c = st.columns(3)

    def color_status(pred):
        return f'<span style="color:red;font-weight:bold;font-size:1.5em">Rusak</span>' if pred == 1 else f'<span style="color:green;font-weight:bold;font-size:1.5em">Normal</span>'

    with col_a:
        st.markdown(f"**Random Forest**<br>{color_status(rf_pred)}", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"**XGBoost**<br>{color_status(xgb_pred)}", unsafe_allow_html=True)

    with col_c:
        st.markdown(f"**ANN**<br>{color_status(ann_pred)}", unsafe_allow_html=True)

    n_fail = sum([rf_pred, xgb_pred, ann_pred])
    if n_fail >= 2:
        st.error("Mayoritas model memprediksi **rusak**. Inspeksi disarankan.")
    elif n_fail == 1:
        st.warning("Satu model memprediksi kerusakan — pantau secara berkala.")
    else:
        st.success("Semua model memprediksi **normal**.")
