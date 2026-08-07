"""
Streamlit app: Rain Tomorrow Predictor
Run with:  streamlit run app.py
Requires:  rain_model.pkl (produced by the training notebook) in the same folder.
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Rain Tomorrow Predictor", page_icon="\U0001F327", layout="centered")

st.title("\U0001F327 Will It Rain Tomorrow?")
st.write(
    "Enter today's weather conditions and a trained Random Forest model will estimate "
    "the chance of rain tomorrow."
)


# ----------------------------------------------------------------------------
# Load the trained model bundle (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_bundle(path: str = "rain_model.pkl"):
    return joblib.load(path)


bundle = load_bundle()
model = bundle["model"]
le_season = bundle["le_season"]
le_raintoday = bundle["le_raintoday"]
le_target = bundle["le_target"]
feature_cols = bundle["feature_cols"]

# ----------------------------------------------------------------------------
# Sidebar inputs — one widget per feature the model expects
# ----------------------------------------------------------------------------
st.sidebar.header("Today\'s Conditions")

season = st.sidebar.selectbox("Season", options=list(le_season.classes_))
min_temp = st.sidebar.slider("Min Temperature (\u00b0C)", -10.0, 30.0, 10.0, 0.5)
max_temp = st.sidebar.slider("Max Temperature (\u00b0C)", -5.0, 40.0, 20.0, 0.5)
avg_temp = st.sidebar.slider("Average Temperature (\u00b0C)", -10.0, 35.0, 15.0, 0.5)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 65)
pressure = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, 1013.0, 0.5)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 60.0, 12.0, 0.5)
cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
precipitation = st.sidebar.slider("Precipitation Today (mm)", 0.0, 60.0, 0.0, 0.5)
rain_today = st.sidebar.selectbox("Did it rain today?", options=list(le_raintoday.classes_))

predict_clicked = st.sidebar.button("\U0001F52E Predict", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Build the feature row exactly the way the training data was encoded
# ----------------------------------------------------------------------------
def build_input_row():
    row = {
        "Season_enc": le_season.transform([season])[0],
        "MinTemp_C": min_temp,
        "MaxTemp_C": max_temp,
        "AvgTemp_C": avg_temp,
        "Humidity_pct": humidity,
        "Pressure_hPa": pressure,
        "WindSpeed_kmh": wind_speed,
        "CloudCover_pct": cloud_cover,
        "Precipitation_mm": precipitation,
        "RainToday_enc": le_raintoday.transform([rain_today])[0],
    }
    return pd.DataFrame([row])[feature_cols]


# ----------------------------------------------------------------------------
# Main panel: show a summary of inputs, then the prediction once requested
# ----------------------------------------------------------------------------
with st.expander("Show the exact values being sent to the model"):
    st.write(build_input_row())

if predict_clicked:
    X_input = build_input_row()
    pred_encoded = model.predict(X_input)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(X_input)[0]
    proba_dict = {cls: p for cls, p in zip(le_target.classes_, proba)}

    st.subheader("Prediction")
    if pred_label == "Yes":
        st.error(f"\u2614 Rain expected tomorrow  —  {proba_dict['Yes']*100:.1f}% probability")
    else:
        st.success(f"\u2600\ufe0f No rain expected tomorrow  —  {proba_dict['No']*100:.1f}% probability of staying dry")

    st.write("**Full probability breakdown:**")
    st.bar_chart(pd.Series(proba_dict, name="Probability"))

    st.caption(
        "This model was trained on synthetic historical weather data using a Random Forest "
        "classifier. Treat this as a learning demo, not a real forecast."
    )
else:
    st.info("Set today\'s conditions in the sidebar, then click **Predict**.")
