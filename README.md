# 🚧 Road Pothole Severity Classification ML System

An end-to-end Machine Learning and interactive Web Application for **Road Pothole Severity Assessment & Maintenance Scheduling**, modeled after modern intelligent transportation systems.

---

## 📌 Project Overview
Potholes pose significant threats to motorist safety, cause vehicle damage, and accelerate pavement degradation. This system predicts the **Severity Level** (`Low`, `Moderate`, `High`, or `Critical`) of detected potholes based on physical dimensions, road classification, traffic volume, environmental moisture, and telematics vibration impact (IoT accelerometer data).

### Features Used:
- **Pothole Depth (`cm`)**: Measured vertical depth from road surface.
- **Pothole Diameter / Width (`cm`)**: Major axis diameter across the cavity.
- **Surface Area (`cm²`)**: Calculated surface fracture footprint.
- **Roadway Type**: `Residential Road`, `Collector Street`, `Urban Arterial`, `Highway / Expressway`.
- **Traffic Speed Limit (`km/h`)**: Speed threshold affecting collision severity.
- **Daily Traffic Volume (`Vehicles/Day`)**: Exposure load factor.
- **Water Presence**: `Dry`, `Damp`, `Standing Water / Flooded` (accelerates sub-base erosion).
- **Peak Vertical Vibration (`g`)**: Z-axis accelerometer impact recorded by connected telematics sensors.

---

## 🗂️ Project Structure
```
road_pothole_severity_ml/
│
├── potholes.csv                 # Dataset (2,500 records aligned with ASTM D6433 standards)
├── generate_dataset.py          # Script to reproduce/augment dataset
├── train.py                     # ML training pipeline (Decision Tree & Random Forest)
├── pothole_model_bundle.pkl     # Exported model, encoders, and evaluation metadata
├── app.py                       # Modern Streamlit Web Dashboard (matches the reference UI)
├── requirements.txt             # Required Python libraries
└── README.md                    # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or use `py -m pip install -r requirements.txt` on Windows)*

### 2. Train the Model
```bash
python train.py
```
*(Or `py train.py`)*

Outputs:
- Model benchmark comparing Decision Tree (89.8%), Random Forest (**95.6%**), and Gradient Boosting (**95.6%**)
- Detailed precision, recall, and F1-score across all 4 severity classes
- Confusion matrix and feature importance ranking
- Exports champion model into `pothole_model_bundle.pkl`

### 3. Launch the Streamlit Web Application
```bash
py -m streamlit run app.py
```
The app will automatically open in your browser at: `http://localhost:8501`

---

## 📊 Severity Levels & Civil Engineering SLAs
| Level | Severity Badge | Description | Repair SLA |
| :--- | :--- | :--- | :--- |
| **Low** | 🟢 Low Severity | Shallow surface spall (<2.5 cm depth) | Routine maintenance (30-60 days) |
| **Moderate** | 🟡 Moderate Severity | Noticeable cavity (2.5 - 5 cm depth) | Schedule within 1-2 weeks |
| **High** | 🟠 High Severity | Deep rupture (>5 cm depth), risk of rim damage | Urgent repair within 48 hours |
| **Critical** | 🔴 Critical Hazard | Extreme crater on arterial/highway, blowout danger | Emergency dispatch (12-24 hours) |
Deployement link : `https://roadpatholeseveritymodel-lnjsmm4z8w7rlm7b83du8f.streamlit.app/`