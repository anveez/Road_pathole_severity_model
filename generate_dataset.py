import numpy as np
import pandas as pd

def generate_pothole_dataset(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    road_types = ["Residential Road", "Collector Street", "Urban Arterial", "Highway / Expressway"]
    water_conditions = ["Dry", "Damp", "Standing Water / Flooded"]
    
    # Generate base features
    road_type_choices = np.random.choice(road_types, size=n_samples, p=[0.30, 0.30, 0.25, 0.15])
    water_choices = np.random.choice(water_conditions, size=n_samples, p=[0.55, 0.25, 0.20])
    
    # Speed limit based on road type
    speed_map = {
        "Residential Road": (25, 40),
        "Collector Street": (40, 60),
        "Urban Arterial": (60, 80),
        "Highway / Expressway": (80, 120)
    }
    traffic_vol_map = {
        "Residential Road": (300, 2500),
        "Collector Street": (2500, 8000),
        "Urban Arterial": (8000, 20000),
        "Highway / Expressway": (15000, 35000)
    }
    
    speed_limits = []
    traffic_volumes = []
    for rt in road_type_choices:
        s_min, s_max = speed_map[rt]
        v_min, v_max = traffic_vol_map[rt]
        speed_limits.append(np.random.randint(s_min, s_max + 1))
        traffic_volumes.append(np.random.randint(v_min, v_max + 1))
        
    speed_limits = np.array(speed_limits)
    traffic_volumes = np.array(traffic_volumes)
    
    # Realistic depth (cm) and diameter (cm)
    # Most potholes are between 1 cm and 12 cm deep, 10 to 90 cm wide
    depths = np.clip(np.random.exponential(scale=3.8, size=n_samples) + 0.8, 0.8, 14.5)
    diameters = np.clip(depths * np.random.uniform(4.5, 9.5, size=n_samples) + np.random.normal(5, 4, size=n_samples), 8.0, 110.0)
    
    # Approximate surface area in sq cm (ellipse/circle area: pi * (d/2)^2 * aspect_ratio)
    aspect_ratio = np.random.uniform(0.7, 1.3, size=n_samples)
    surface_areas = np.round(np.pi * (diameters / 2) * (diameters * aspect_ratio / 2), 1)
    
    # Accelerometer / telematics peak vertical vibration impact (g)
    # Correlated with depth and vehicle speed: higher speed + deeper hole = higher vibration shock
    vibration_peak_g = np.round(0.2 + (depths / 12.0) * 1.8 + (speed_limits / 120.0) * 1.2 + np.random.normal(0, 0.15, size=n_samples), 2)
    vibration_peak_g = np.clip(vibration_peak_g, 0.15, 4.2)
    
    # Calculate severity according to civil engineering criteria (ASTM D6433 adapted)
    # Score calculation
    severity_scores = []
    for i in range(n_samples):
        d = depths[i]
        dia = diameters[i]
        speed = speed_limits[i]
        rt = road_type_choices[i]
        water = water_choices[i]
        vib = vibration_peak_g[i]
        
        # Base severity score (0 to 100)
        # Depth is the single highest predictor:
        score = (d / 12.0) * 45.0 + (dia / 90.0) * 25.0
        
        # Speed & Road type hazard multiplier
        if speed >= 80 or rt == "Highway / Expressway":
            score += 15.0
        elif speed >= 60 or rt == "Urban Arterial":
            score += 10.0
            
        # Water accumulation increases deterioration & accident risk
        if water == "Standing Water / Flooded":
            score += 12.0
        elif water == "Damp":
            score += 4.0
            
        # Vibration impact from telematics
        score += (vib / 3.5) * 10.0
        
        # Controlled realistic inspection measurement noise
        score += np.random.normal(0, 0.7)
        severity_scores.append(score)
        
    severity_scores = np.array(severity_scores)
    
    # Map score to 4 severity categories
    labels = []
    for s in severity_scores:
        if s < 30:
            labels.append("Low")
        elif s < 54:
            labels.append("Moderate")
        elif s < 76:
            labels.append("High")
        else:
            labels.append("Critical")
            
    df = pd.DataFrame({
        "pothole_depth_cm": np.round(depths, 2),
        "pothole_diameter_cm": np.round(diameters, 2),
        "surface_area_sq_cm": surface_areas,
        "road_type": road_type_choices,
        "traffic_speed_kmh": speed_limits,
        "traffic_volume_daily": traffic_volumes,
        "water_presence": water_choices,
        "vibration_peak_g": vibration_peak_g,
        "severity_level": labels
    })
    
    return df

if __name__ == "__main__":
    df = generate_pothole_dataset(n_samples=2500)
    output_path = "C:/Users/WE4YOU/.gemini/antigravity/scratch/road_pothole_severity_ml/potholes.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully with {len(df)} records at {output_path}")
    print("\nClass distribution:")
    print(df["severity_level"].value_counts())
