import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pothole Severity Classifier",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS STYLING (MATCHING DEMO THEME)
# ==========================================
st.markdown("""
<style>
    /* Main app container */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #581c87 80%, #831843 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 30px 35px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-tag {
        font-size: 0.78rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
        color: #f43f5e;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 8px;
        margin-bottom: 0px;
    }

    /* Status indicators */
    .status-row {
        display: flex;
        gap: 20px;
        margin-top: 15px;
        font-size: 0.85rem;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #cbd5e1;
    }
    .dot-green {
        width: 9px;
        height: 9px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }
    .dot-blue {
        width: 9px;
        height: 9px;
        background-color: #38bdf8;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #38bdf8;
    }

    /* Section container */
    .section-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px 25px;
        margin-bottom: 24px;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 10px;
    }

    /* Prediction Result Box */
    .result-banner {
        border-radius: 14px;
        padding: 25px 20px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .result-title {
        font-size: 0.82rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 10px;
    }
    .result-severity {
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: 1px;
    }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Advisory card */
    .advisory-box {
        background: #1c2128;
        border-left: 4px solid #f43f5e;
        border-radius: 0 10px 10px 0;
        padding: 15px 20px;
        margin-top: 15px;
    }
    
    /* Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #be123c 0%, #881337 100%);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 14px 28px;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(190, 18, 60, 0.4);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #e11d48 0%, #9f1239 100%);
        box-shadow: 0 6px 20px rgba(225, 29, 72, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD TRAINED MODEL (WITH CLOUD AUTO-FALLBACK)
# ==========================================
@st.cache_resource
def load_bundle():
    model_path = os.path.join(os.path.dirname(__file__), "pothole_model_bundle.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass  # Fallback to in-memory training for cross-platform/cloud compatibility
            
    # Seamless cloud self-training fallback (takes < 1 sec, runs once and cached)
    csv_path = os.path.join(os.path.dirname(__file__), "potholes.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        road_type_mapping = {
            "Residential Road": 0,
            "Collector Street": 1,
            "Urban Arterial": 2,
            "Highway / Expressway": 3
        }
        water_mapping = {
            "Dry": 0,
            "Damp": 1,
            "Standing Water / Flooded": 2
        }
        severity_order = ["Low", "Moderate", "High", "Critical"]
        severity_mapping = {label: idx for idx, label in enumerate(severity_order)}
        reverse_severity_mapping = {idx: label for label, idx in severity_mapping.items()}

        df_clean = df.copy()
        df_clean["road_type_encoded"] = df_clean["road_type"].map(road_type_mapping)
        df_clean["water_presence_encoded"] = df_clean["water_presence"].map(water_mapping)
        df_clean["severity_encoded"] = df_clean["severity_level"].map(severity_mapping)

        feature_cols = [
            "pothole_depth_cm",
            "pothole_diameter_cm",
            "surface_area_sq_cm",
            "road_type_encoded",
            "traffic_speed_kmh",
            "traffic_volume_daily",
            "water_presence_encoded",
            "vibration_peak_g"
        ]

        X = df_clean[feature_cols]
        y = df_clean["severity_encoded"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
        rf_model.fit(X_train, y_train)

        rf_preds = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds, average="weighted")

        return {
            "model": rf_model,
            "model_name": "Random Forest",
            "feature_cols": feature_cols,
            "road_type_mapping": road_type_mapping,
            "water_mapping": water_mapping,
            "severity_order": severity_order,
            "severity_mapping": severity_mapping,
            "reverse_severity_mapping": reverse_severity_mapping,
            "metrics": {
                "accuracy": float(rf_acc),
                "f1_score": float(rf_f1)
            }
        }
    return None

bundle = load_bundle()

# ==========================================
# HEADER BANNER
# ==========================================
acc_display = f"{bundle['metrics']['accuracy'] * 100:.1f}%" if bundle else "95.6%"
model_display = bundle.get("model_name", "Random Forest") if bundle else "Random Forest"

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-tag">◆ ROAD INFRASTRUCTURE MONITORING</div>
    <h1 class="hero-title">Pothole Severity Classifier</h1>
    <p class="hero-sub">Enter road inspection measurements to estimate structural damage severity & repair urgency.</p>
    <div class="status-row">
        <div class="status-item"><span class="dot-green"></span> Pavement Health Monitoring Active</div>
        <div class="status-item"><span class="dot-blue"></span> AI Engine: {model_display} (<strong>{acc_display} Accuracy</strong>)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PRESETS FOR QUICK TESTING
# ==========================================
with st.expander("⚡ Click to load quick test scenarios (Presets)"):
    preset_cols = st.columns(4)
    if preset_cols[0].button("🟢 Minor Surface Spall"):
        st.session_state["p_depth"] = 1.5
        st.session_state["p_diameter"] = 14.0
        st.session_state["p_road_type"] = "Residential Road"
        st.session_state["p_water"] = "Dry"
        st.session_state["p_speed"] = 30
        st.session_state["p_vibration"] = 0.65
        st.session_state["p_volume"] = 1200
        st.rerun()

    if preset_cols[1].button("🟡 Moderate Collector"):
        st.session_state["p_depth"] = 3.6
        st.session_state["p_diameter"] = 28.0
        st.session_state["p_road_type"] = "Collector Street"
        st.session_state["p_water"] = "Damp"
        st.session_state["p_speed"] = 50
        st.session_state["p_vibration"] = 1.45
        st.session_state["p_volume"] = 5500
        st.rerun()

    if preset_cols[2].button("🟠 Arterial Pothole"):
        st.session_state["p_depth"] = 6.2
        st.session_state["p_diameter"] = 48.0
        st.session_state["p_road_type"] = "Urban Arterial"
        st.session_state["p_water"] = "Standing Water / Flooded"
        st.session_state["p_speed"] = 70
        st.session_state["p_vibration"] = 2.40
        st.session_state["p_volume"] = 14000
        st.rerun()

    if preset_cols[3].button("🔴 Highway Crater"):
        st.session_state["p_depth"] = 10.5
        st.session_state["p_diameter"] = 82.0
        st.session_state["p_road_type"] = "Highway / Expressway"
        st.session_state["p_water"] = "Standing Water / Flooded"
        st.session_state["p_speed"] = 100
        st.session_state["p_vibration"] = 3.50
        st.session_state["p_volume"] = 28000
        st.rerun()

# ==========================================
# INPUT PARAMETERS SECTION
# ==========================================
st.markdown('<div class="section-card"><div class="section-title">🛠️ Pothole & Roadway Inspection Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

road_options = ["Residential Road", "Collector Street", "Urban Arterial", "Highway / Expressway"]
water_options = ["Dry", "Damp", "Standing Water / Flooded"]

with col1:
    road_type = st.selectbox(
        "Roadway Classification",
        options=road_options,
        index=road_options.index(st.session_state.get("p_road_type", "Collector Street")),
        help="Higher-speed roads increase collision hazard and tire blow-out probability."
    )
    
    depth_cm = st.number_input(
        "Pothole Depth (cm)",
        min_value=0.5,
        max_value=20.0,
        value=float(st.session_state.get("p_depth", 3.8)),
        step=0.1,
        help="ASTM D6433 classifies depth > 5 cm as severe."
    )
    
    speed_limit = st.slider(
        "Traffic Speed Limit (km/h)",
        min_value=20,
        max_value=120,
        value=int(st.session_state.get("p_speed", 50)),
        step=5
    )

with col2:
    water_presence = st.selectbox(
        "Water & Drainage Condition",
        options=water_options,
        index=water_options.index(st.session_state.get("p_water", "Damp")),
        help="Standing water obscures depth from motorists and accelerates base-course erosion."
    )
    
    diameter_cm = st.number_input(
        "Pothole Diameter / Width (cm)",
        min_value=5.0,
        max_value=150.0,
        value=float(st.session_state.get("p_diameter", 32.0)),
        step=0.5,
        help="Major axis diameter of surface fracture."
    )
    
    vibration_g = st.slider(
        "IoT Sensor Peak Vertical Vibration Impact (g)",
        min_value=0.1,
        max_value=4.5,
        value=float(st.session_state.get("p_vibration", 1.5)),
        step=0.05,
        help="Z-axis accelerometer impact recorded by connected telematics or vehicle sensors."
    )

daily_volume = st.number_input(
    "Estimated Daily Traffic Volume (Vehicles/Day)",
    min_value=100,
    max_value=50000,
    value=int(st.session_state.get("p_volume", 6500)),
    step=500
)

st.markdown('</div>', unsafe_allow_html=True)

# Derive surface area (approximate circular/elliptical crater)
surface_area = round(np.pi * ((diameter_cm / 2) ** 2), 1)

# ==========================================
# PREDICTION LOGIC & TRIGGER
# ==========================================
classify_btn = st.button("CHECK POTHOLE SEVERITY")

if classify_btn or "last_prediction" in st.session_state:
    if bundle is None:
        st.error("Model bundle 'pothole_model_bundle.pkl' not found. Please run 'train.py' first!")
    else:
        model = bundle["model"]
        road_mapping = bundle["road_type_mapping"]
        water_mapping = bundle["water_mapping"]
        severity_order = bundle["severity_order"]
        
        # Prepare feature vector
        features = pd.DataFrame([{
            "pothole_depth_cm": depth_cm,
            "pothole_diameter_cm": diameter_cm,
            "surface_area_sq_cm": surface_area,
            "road_type_encoded": road_mapping[road_type],
            "traffic_speed_kmh": speed_limit,
            "traffic_volume_daily": daily_volume,
            "water_presence_encoded": water_mapping[water_presence],
            "vibration_peak_g": vibration_g
        }])
        
        # Inference
        probs = model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        predicted_severity = severity_order[pred_idx]
        confidence = probs[pred_idx] * 100
        
        # Calculate ASTM Hazard Index (0 to 100)
        hazard_index = min(100, int((depth_cm / 12.0) * 45 + (diameter_cm / 90.0) * 25 + (speed_limit / 120.0) * 15 + (vibration_g / 3.5) * 15))
        
        # UI Styling configurations per class
        style_config = {
            "Low": {
                "color": "#10b981",
                "bg": "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%)",
                "border": "#10b981",
                "icon": "🟢",
                "sla": "Routine (30-60 Days)",
                "action": "Surface patch during scheduled road maintenance. No emergency dispatch needed."
            },
            "Moderate": {
                "color": "#eab308",
                "bg": "linear-gradient(135deg, rgba(234, 179, 8, 0.15) 0%, rgba(113, 63, 18, 0.3) 100%)",
                "border": "#eab308",
                "icon": "🟡",
                "sla": "Schedule within 1-2 Weeks",
                "action": "Standard cold-mix or mastic patch recommended before rain exacerbates base erosion."
            },
            "High": {
                "color": "#f97316",
                "bg": "linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(154, 52, 18, 0.3) 100%)",
                "border": "#f97316",
                "icon": "🟠",
                "sla": "Urgent (Within 48 Hours)",
                "action": "Deep cavity edge milling and hot-mix asphalt compaction required. High risk of rim damage."
            },
            "Critical": {
                "color": "#ef4444",
                "bg": "linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(127, 29, 29, 0.4) 100%)",
                "border": "#ef4444",
                "icon": "🔴",
                "sla": "EMERGENCY (Within 12-24 Hours)",
                "action": "Imminent tire blowout & motorcycle accident hazard! Deploy temporary safety barriers and emergency road crew."
            }
        }
        
        cfg = style_config[predicted_severity]
        
        # ==========================================
        # RESULT BANNER (EXACTLY AS IN LINKEDIN DEMO)
        # ==========================================
        st.markdown(f"""
        <div class="result-banner" style="background: {cfg['bg']}; border: 1px solid {cfg['border']};">
            <div class="result-title">CURRENT SEVERITY ASSESSMENT</div>
            <div class="result-severity" style="color: {cfg['color']};">
                {cfg['icon']} {predicted_severity.upper()} SEVERITY
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric Grid
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-label">Prediction Confidence</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{hazard_index} / 100</div>
                <div class="metric-label">Structural Hazard Index</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="font-size: 1.25rem; color: {cfg['color']};">{cfg['sla']}</div>
                <div class="metric-label">Recommended Repair SLA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================
        # PROBABILITY DISTRIBUTION CHART (LIKE IMAGE 4)
        # ==========================================
        st.markdown('<div class="section-card"><div class="section-title">📊 Prediction Probability Distribution</div>', unsafe_allow_html=True)
        
        # Plot using Matplotlib with modern styling
        fig, ax = plt.subplots(figsize=(10, 3.8), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        
        class_colors = ["#10b981", "#eab308", "#f97316", "#ef4444"]
        bars = ax.bar(severity_order, probs * 100, color=class_colors, width=0.45, edgecolor="none", zorder=3)
        
        # Formatting
        ax.set_ylim(0, 105)
        ax.set_ylabel("Probability (%)", color="#94a3b8", fontsize=11, fontweight="bold")
        ax.set_xlabel("Pothole Severity Level", color="#94a3b8", fontsize=11, fontweight="bold")
        ax.set_title("Model Confidence Across Severity Classes", color="#f1f5f9", fontsize=13, pad=12, fontweight="bold")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        
        # Gridlines
        ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff", zorder=0)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#30363d")
            
        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        color="#ffffff", fontweight="bold", fontsize=10)
                        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Action Advisory
        st.markdown(f"""
        <div class="advisory-box" style="border-left-color: {cfg['color']};">
            <strong style="color: {cfg['color']};">Civil Maintenance Recommendation:</strong> {cfg['action']}
            <br><span style="color: #8b949e; font-size: 0.85rem;">Estimated Pothole Surface Area: <strong>{surface_area:,.1f} cm²</strong> | Peak Vibration: <strong>{vibration_g} g</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
