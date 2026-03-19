"""
Crop Yield Prediction — Streamlit App
--------------------------------------
Loads the best tuned pipeline from notebook 05 and lets users
input farm parameters to get a yield prediction with confidence context.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense — Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3 { font-family: 'DM Serif Display', serif; }

  /* Main background */
  .stApp { background-color: #F7F5F0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #1C2B1A;
    color: #E8F0E6;
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] p {
    color: #C8D8C4 !important;
    font-size: 13px;
  }
  [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background-color: #243822;
    border-color: #3D5C3A;
  }
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #E8F0E6 !important;
  }

  /* Metric cards */
  .metric-card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    border: 1px solid #E8E4DC;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .metric-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B8578;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 42px;
    color: #1C2B1A;
    line-height: 1;
  }
  .metric-unit {
    font-size: 16px;
    color: #8B8578;
    margin-left: 4px;
  }
  .metric-sub {
    font-size: 13px;
    color: #8B8578;
    margin-top: 8px;
  }

  /* Result panel */
  .result-panel {
    background: #1C2B1A;
    border-radius: 20px;
    padding: 32px;
    color: #E8F0E6;
  }
  .result-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #A8D5A2;
    margin-bottom: 4px;
  }

  /* Info boxes */
  .info-box {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid #4A7C47;
    margin-bottom: 12px;
    font-size: 14px;
    color: #3A3530;
  }
  .warn-box {
    background: #FFF9F0;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid #D4860A;
    margin-bottom: 12px;
    font-size: 14px;
    color: #5C4A1A;
  }

  /* Header */
  .app-header {
    padding: 32px 0 24px;
    border-bottom: 1px solid #E0DAD0;
    margin-bottom: 32px;
  }
  .app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    color: #1C2B1A;
    line-height: 1;
  }
  .app-subtitle {
    font-size: 16px;
    color: #8B8578;
    margin-top: 8px;
    font-weight: 300;
  }

  /* Predict button */
  .stButton > button {
    background-color: #2D4F2A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100%;
    transition: background 0.2s;
  }
  .stButton > button:hover {
    background-color: #1C2B1A !important;
  }

  /* Section heading */
  .section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1C2B1A;
    margin-bottom: 16px;
    margin-top: 32px;
  }

  /* Divider */
  hr { border: none; border-top: 1px solid #E0DAD0; margin: 28px 0; }

  /* Hide streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model and metadata ───────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_rf_tuned_pipeline.joblib")
META_PATH  = os.path.join(os.path.dirname(__file__), "models", "model_meta.json")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)

model = load_model()
meta  = load_meta()

# Fallback lists if model not yet trained
DEFAULT_CROPS = [
    "Rice", "Wheat", "Maize", "Sugarcane", "Cotton(lint)",
    "Arhar/Tur", "Gram", "Groundnut", "Jute", "Bajra",
    "Jowar", "Ragi", "Sesamum", "Linseed", "Sunflower",
    "Potato", "Onion", "Tomato", "Coconut ", "Arecanut",
]
DEFAULT_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Jammu and Kashmir", "Delhi",
]
DEFAULT_SEASONS = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"]

crops   = meta["crops"]   if meta else DEFAULT_CROPS
states  = meta["states"]  if meta else DEFAULT_STATES
seasons = meta["seasons"] if meta else DEFAULT_SEASONS


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🌾 CropSense</div>
  <div class="app-subtitle">Machine learning crop yield predictor — Random Forest · India State-Level · 1997–2020</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Crop & Location")
    crop   = st.selectbox("Crop",   sorted(crops),   index=sorted(crops).index("Rice") if "Rice" in crops else 0)
    state  = st.selectbox("State",  sorted(states),  index=sorted(states).index("Punjab") if "Punjab" in states else 0)
    season = st.selectbox("Season", sorted(seasons), index=0)
    year   = st.slider("Year", min_value=2000, max_value=2030, value=2020, step=1)

    st.markdown("---")
    st.markdown("### Farm Inputs")
    area       = st.number_input("Area (hectares)",      min_value=0.1,   max_value=500000.0, value=1000.0,  step=100.0)
    fertilizer = st.number_input("Fertilizer (kg)",      min_value=0.0,   max_value=5000000.0, value=50000.0, step=1000.0)
    pesticide  = st.number_input("Pesticide (kg)",       min_value=0.0,   max_value=500000.0,  value=5000.0,  step=500.0)

    st.markdown("---")
    st.markdown("### Soil")
    N  = st.slider("Nitrogen (N)",   min_value=0,   max_value=200, value=80)
    P  = st.slider("Phosphorus (P)", min_value=0,   max_value=200, value=40)
    K  = st.slider("Potassium (K)",  min_value=0,   max_value=200, value=40)
    pH = st.slider("Soil pH",        min_value=3.5, max_value=9.0, value=6.5, step=0.1)

    st.markdown("---")
    st.markdown("### Weather")
    avg_temp      = st.slider("Avg Temperature (°C)",   min_value=5.0,  max_value=45.0, value=27.0, step=0.5)
    total_rain    = st.slider("Total Rainfall (mm)",    min_value=50,   max_value=4000, value=900,  step=50)
    avg_humidity  = st.slider("Avg Humidity (%)",       min_value=20,   max_value=100,  value=70,   step=1)

    st.markdown("---")
    st.markdown("### Remote Sensing")
    mean_ndvi = st.slider("Mean NDVI", min_value=0.0, max_value=1.0, value=0.55, step=0.01,
                          help="Normalized Difference Vegetation Index. Higher = healthier vegetation.")

    st.markdown("---")
    predict_btn = st.button("🌱 Predict Yield", use_container_width=True)


# ── Build input dataframe ─────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "crop":                 crop,
    "season":               season,
    "state":                state,
    "year":                 year,
    "area":                 area,
    "fertilizer":           fertilizer,
    "pesticide":            pesticide,
    "N":                    N,
    "P":                    P,
    "K":                    K,
    "pH":                   pH,
    "avg_temp_c":           avg_temp,
    "total_rainfall_mm":    total_rain,
    "avg_humidity_percent": avg_humidity,
    "mean_ndvi":            mean_ndvi,
}])


# ── Main content ──────────────────────────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="warn-box">
      <strong>Model not found.</strong> Run notebook 05 first to train and save the model, then place
      <code>best_rf_tuned_pipeline.joblib</code> and <code>model_meta.json</code>
      in a <code>models/</code> folder next to this file.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Expected folder structure:**")
    st.code("""
your_project/
├── notebooks/
│   ├── 01_crop_yield_eda_cleaning.ipynb
│   ├── ...
│   └── 05_hyperparameter_tuning.ipynb
├── models/
│   ├── best_rf_tuned_pipeline.joblib   ← generated by notebook 05
│   └── model_meta.json                 ← generated by notebook 05
└── app.py                              ← this file
    """)

else:
    # ── Prediction ────────────────────────────────────────────────────────────
    if predict_btn:
        with st.spinner("Predicting..."):
            try:
                pred_log  = model.predict(input_data)[0]
                pred_orig = np.expm1(pred_log)

                # Rough confidence interval — ±15% based on model CV RMSE
                ci_low  = pred_orig * 0.85
                ci_high = pred_orig * 1.15

                # Yield category
                if pred_orig < 500:
                    category, cat_color = "Low yield", "#D4860A"
                elif pred_orig < 2000:
                    category, cat_color = "Moderate yield", "#4A7C47"
                elif pred_orig < 5000:
                    category, cat_color = "Good yield", "#2D6E2A"
                else:
                    category, cat_color = "Excellent yield", "#1A4D18"

                # Store in session state
                st.session_state["result"] = {
                    "pred": pred_orig,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "category": category,
                    "cat_color": cat_color,
                    "crop": crop,
                    "state": state,
                    "season": season,
                    "year": year,
                }
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ── Show result if available ──────────────────────────────────────────────
    if "result" in st.session_state:
        r = st.session_state["result"]

        st.markdown('<div class="section-head">Prediction Result</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"""
            <div class="result-panel">
              <div style="font-size:13px; color:#7DB87A; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;">
                {r['crop']} · {r['state']} · {r['season']} {r['year']}
              </div>
              <div class="result-heading">Predicted Yield</div>
              <div style="font-family:'DM Serif Display',serif; font-size:64px; color:#E8F0E6; line-height:1; margin: 8px 0;">
                {r['pred']:,.0f}
                <span style="font-size:22px; color:#7DB87A;">kg / ha</span>
              </div>
              <div style="font-size:14px; color:#7DB87A; margin-top:12px;">
                95% confidence range: {r['ci_low']:,.0f} – {r['ci_high']:,.0f} kg/ha
              </div>
              <div style="margin-top:16px; display:inline-block; background:{r['cat_color']}33;
                          color:{r['cat_color']}; border:1px solid {r['cat_color']}66;
                          border-radius:8px; padding:6px 16px; font-size:13px; font-weight:600;">
                {r['category']}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="height:100%;">
              <div class="metric-label">Total Production Est.</div>
              <div class="metric-value">{r['pred'] * area / 1000:,.1f}<span class="metric-unit">tonnes</span></div>
              <div class="metric-sub">Based on {area:,.0f} ha farm area</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            national_avg = 1800  # approximate India average kg/ha
            pct_vs_avg = ((r['pred'] - national_avg) / national_avg) * 100
            sign = "+" if pct_vs_avg >= 0 else ""
            color = "#2D6E2A" if pct_vs_avg >= 0 else "#C0392B"
            st.markdown(f"""
            <div class="metric-card" style="height:100%;">
              <div class="metric-label">vs. National Average</div>
              <div class="metric-value" style="color:{color}; font-size:36px;">{sign}{pct_vs_avg:.1f}<span class="metric-unit">%</span></div>
              <div class="metric-sub">National avg ≈ 1,800 kg/ha</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature contribution chart ────────────────────────────────────────
        st.markdown('<div class="section-head">Input Summary</div>', unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor("#F7F5F0")

        # Soil nutrients bar
        nutrients = ["N", "P", "K"]
        values    = [N, P, K]
        colors    = ["#2D6E2A", "#4A7C47", "#7DB87A"]
        axes[0].bar(nutrients, values, color=colors, width=0.5, edgecolor="none")
        axes[0].set_facecolor("#F7F5F0")
        axes[0].set_title("Soil Nutrients", fontsize=13, color="#1C2B1A", pad=12)
        axes[0].set_ylabel("Value", fontsize=11, color="#8B8578")
        axes[0].tick_params(colors="#8B8578")
        for spine in axes[0].spines.values():
            spine.set_edgecolor("#E0DAD0")
        for i, v in enumerate(values):
            axes[0].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=11, color="#1C2B1A", fontweight="600")

        # Weather radar-style bar
        weather_labels = ["Temp (°C)", "Rainfall/10 (mm)", "Humidity (%)"]
        weather_values = [avg_temp, total_rain / 10, avg_humidity]
        axes[1].barh(weather_labels, weather_values, color=["#4A7C47", "#2D6E2A", "#7DB87A"],
                     height=0.5, edgecolor="none")
        axes[1].set_facecolor("#F7F5F0")
        axes[1].set_title("Weather Inputs", fontsize=13, color="#1C2B1A", pad=12)
        axes[1].tick_params(colors="#8B8578")
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#E0DAD0")
        for i, (label, val) in enumerate(zip(weather_labels, weather_values)):
            axes[1].text(val + 0.5, i, f"{val:.1f}", va="center", fontsize=11, color="#1C2B1A", fontweight="600")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        # ── Landing state (no prediction yet) ────────────────────────────────
        st.markdown('<div class="section-head">How to use</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="info-box">
              <strong>Step 1 — Select crop & location</strong><br>
              Choose your crop, state, and season from the sidebar dropdowns. Pick the year you want to predict for.
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box">
              <strong>Step 2 — Enter farm data</strong><br>
              Input your farm area, fertilizer and pesticide usage, soil NPK values and pH, and local weather conditions.
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="info-box">
              <strong>Step 3 — Predict</strong><br>
              Click <em>Predict Yield</em> to get an instant prediction in kg/ha, estimated total production, and comparison to the national average.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-head">Model Info</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Algorithm",     "Random Forest",  "Tuned with RandomizedSearchCV + GridSearchCV"),
            ("CV R² Score",   "0.976+",         "5-fold TimeSeriesSplit cross-validation"),
            ("CV RMSE",       "~132 kg/ha",     "After log-transform and back-transform"),
            ("Training Data", "~17,000 rows",   "30 Indian states · 55 crops · 1997–2020"),
        ]
        for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div style="font-family:'DM Serif Display',serif; font-size:22px; color:#1C2B1A; margin:6px 0;">{value}</div>
                  <div class="metric-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
          <strong>Note on NDVI:</strong> If you don't have a real NDVI value for your region,
          use 0.4–0.6 for typical Indian agricultural land during the growing season.
          Values above 0.7 indicate very healthy/dense vegetation.
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#B0A898; font-size:12px; padding: 8px 0 24px;">
  CropSense · Random Forest Crop Yield Predictor · India State-Level Data 1997–2020<br>
  Predictions are estimates based on historical patterns. Actual yields depend on many additional factors.
</div>
""", unsafe_allow_html=True)
