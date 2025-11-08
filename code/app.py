"""
Streamlit Web Application for Crop Recommendation and Rotation Planning
Corrected and consolidated version of app.py to ensure prediction model
receives the exact expected features and to robustly handle missing artifacts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import difflib

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Configuration -----------------
BASE = Path(__file__).resolve().parent
SAVED_DIR = BASE / "saved_models"
SAVED_DIR.mkdir(exist_ok=True)
MODEL_PATHS = {
    "best_model": SAVED_DIR / "best_model.pkl",
    "random_forest": SAVED_DIR / "random_forest.pkl",
    "rf_model": SAVED_DIR / "rf_model.pkl",
    "scaler": SAVED_DIR / "scaler.pkl",
    "label_encoder": SAVED_DIR / "label_encoder.pkl",
    "target_encoder": SAVED_DIR / "target_encoder.pkl",
    "label_encoders": SAVED_DIR / "label_encoders.pkl",
    "metadata": SAVED_DIR / "metadata.pkl"
}

# Update CANONICAL_FEATURES to match training data
CANONICAL_FEATURES = [
    "Temperature", "Rainfall", "Light_Intensity", 
    "Nitrogen", "Phosphorus", "Potassium",
    "Season", "Soil_Type", "Impact", "Fertility"
]

# ----------------- UI CSS -----------------
st.markdown(
        """
        <style>
        :root {
            --accent: #2E7D32;
            --muted: #6b7280;
            --card-bg: #ffffff;
            --glass: rgba(255,255,255,0.85);
        }
        body {
            background: linear-gradient(180deg, #f6fbf6 0%, #ffffff 60%);
        }
        .main-header {
            font-size: 2.4rem; color: var(--accent); font-weight:800; text-align:center; margin-bottom:0.25rem;
            letter-spacing: 0.6px;
        }
        .subtitle { text-align:center; color:var(--muted); margin-top:0; margin-bottom:1rem }
        .sub-header { font-size: 1.2rem; color: #388E3C; margin-top: 1rem; }
        .result-box { padding: 1rem; background:var(--glass); border-left:6px solid var(--accent); border-radius:10px; box-shadow:0 6px 18px rgba(46,125,50,0.06); }

        /* Card for crop recommendation */
        .crop-card { background: var(--card-bg); border-radius:12px; padding:12px; text-align:center; box-shadow:0 8px 24px rgba(13, 71, 11, 0.06); }
        .crop-emoji { font-size:4rem; margin-bottom:6px; }
        .crop-name { font-weight:700; color:#075e23; font-size:1.05rem; }
        .crop-score { color: #444; font-size:0.95rem; margin-top:6px }

        /* Sidebar tweaks */
        .stSidebar .css-1d391kg { padding-top: 1rem; }

        /* Button consistent style */
        .big-button { background: linear-gradient(90deg,#4caf50,#2e7d32); color: white !important; border-radius:6px; padding:8px 12px }
    
        @media (max-width: 600px){ .crop-emoji { font-size:3rem } }
        </style>
        """, unsafe_allow_html=True)

# Simple emoji map
CROP_EMOJIS = {
    # Unique, related emoji per crop (aiming for distinct icons across the set)
    'Tomatoes': '🍅',       # tomato
    'Potatoes': '🥔',       # potato
    'Sweet Potato': '🍠',   # sweet potato
    'Lettuce': '🥗',        # salad/lettuce
    'Cabbage': '🥬',        # cabbage / leafy green
    'Spinach': '🌿',        # herb/leaf for spinach
    'Cucumber': '🥒',       # cucumber
    'Carrot': '🥕',         # carrot
    'Eggplant': '🍆',       # eggplant
    'Corn': '🌽',           # corn
    'Pepper': '🫑',         # bell pepper
    'Chili': '🌶️',          # hot chili
    'Garlic': '🧄',         # garlic
    'Onion': '🧅',          # onion
    'Mushroom': '🍄',       # mushroom
    'Beet': '🟣',           # purple circle as beet surrogate
    'Rice': '🍚',           # cooked rice
    'Wheat': '🌾',          # sheaf of wheat
    'Peas': '🫛',           # pea pod
    'Beans': '🫘',          # beans
    'Broccoli': '🥦',       # broccoli
    'Cauliflower': '🥣',    # bowl as a surrogate for cauliflower/veg
    'Cauliflowers': '🥣',   # plural dataset variant
    'Asparagus': '🌿',      # asparagus (use herb/leaf emoji)
    'Peanut': '🥜',         # peanut
    'Radish': '�',         # basket as surrogate for root veg
    'default': '🌱'         # seedling for unknown crops
}

def get_crop_emoji(name: str) -> str:
    """Get emoji for crop name.

    Matching strategy (robust to case and simple pluralization):
    1. Exact key match.
    2. Lowercase match.
    3. If name ends with 's', try singular by stripping final 's'.
    4. Try common plural -> singular transform (ies->y).
    5. Fuzzy match against known keys (difflib) with a high cutoff.
    Falls back to the 'default' emoji.
    """
    if not name:
        return CROP_EMOJIS.get('default', '🌱')

    key = str(name).strip()
    # 1) direct exact
    if key in CROP_EMOJIS:
        return CROP_EMOJIS[key]

    lower_key = key.lower()
    # prepare normalized lookup dict (lowercased keys)
    normalized = {k.lower(): v for k, v in CROP_EMOJIS.items()}

    # 2) lowercase direct
    if lower_key in normalized:
        return normalized[lower_key]

    # 3) naive singularization: remove trailing 's'
    if lower_key.endswith('s'):
        singular = lower_key[:-1]
        if singular in normalized:
            return normalized[singular]

    # 4) ies -> y (e.g., 'cherries' -> 'cherry')
    if lower_key.endswith('ies'):
        cand = lower_key[:-3] + 'y'
        if cand in normalized:
            return normalized[cand]

    # 5) fuzzy match against known keys
    choices = list(normalized.keys())
    match = difflib.get_close_matches(lower_key, choices, n=1, cutoff=0.78)
    if match:
        return normalized[match[0]]

    # fallback
    return CROP_EMOJIS.get('default', '🌱')

# ----------------- Model Loading -----------------
@st.cache_resource
@st.cache_resource
def load_app_models() -> Dict[str, Any]:
    """
    Load persisted model artifacts with caching for performance.
    Returns a dict containing:
    - model, scaler, target_encoder, label_encoders (dict), metadata (dict)
    Provides robust fallbacks (dummy model / encoders) so UI does not crash.
    """
    models = {
        "model": None,
        "scaler": None,
        "target_encoder": None,
        "label_encoders": {},
        "metadata": {"feature_cols": CANONICAL_FEATURES, "classes": [], "accuracy": 0.0, "best_model": "Unknown"}
    }

    try:
        # Prefer a combined best_model if present
        if MODEL_PATHS["best_model"].exists():
            with open(MODEL_PATHS["best_model"], "rb") as f:
                models["model"] = pickle.load(f)
        elif MODEL_PATHS["random_forest"].exists():
            with open(MODEL_PATHS["random_forest"], "rb") as f:
                models["model"] = pickle.load(f)
        elif MODEL_PATHS["rf_model"].exists():
            with open(MODEL_PATHS["rf_model"], "rb") as f:
                models["model"] = pickle.load(f)

        if MODEL_PATHS["scaler"].exists():
            with open(MODEL_PATHS["scaler"], "rb") as f:
                models["scaler"] = pickle.load(f)

        # target encoder (label encoder for output)
        if MODEL_PATHS["target_encoder"].exists():
            with open(MODEL_PATHS["target_encoder"], "rb") as f:
                models["target_encoder"] = pickle.load(f)
        elif MODEL_PATHS["label_encoder"].exists():
            with open(MODEL_PATHS["label_encoder"], "rb") as f:
                models["target_encoder"] = pickle.load(f)

        # per-column label encoders mapping
        if MODEL_PATHS["label_encoders"].exists():
            with open(MODEL_PATHS["label_encoders"], "rb") as f:
                models["label_encoders"] = pickle.load(f)

        if MODEL_PATHS["metadata"].exists():
            with open(MODEL_PATHS["metadata"], "rb") as f:
                md = pickle.load(f)
                # merge minimal keys
                models["metadata"].update({k: md.get(k, v) for k, v in models["metadata"].items()})
                # ensure feature_cols present
                if "feature_cols" in md and md["feature_cols"]:
                    models["metadata"]["feature_cols"] = md["feature_cols"]
                if "classes" in md and md["classes"]:
                    models["metadata"]["classes"] = md["classes"]
                if "accuracy" in md:
                    models["metadata"]["accuracy"] = md.get("accuracy", 0.0)
                if "best_model" in md:
                    models["metadata"]["best_model"] = md.get("best_model", models["metadata"]["best_model"])
    except Exception:
        # swallow -- we'll create fallbacks below
        pass

    # Ensure model is not None
    if models["model"] is None:
        class DummyModel:
            def __init__(self, classes: List[str]):
                self.classes_ = np.asarray(classes if classes else ["Unknown"])
            def predict_proba(self, X):
                n = len(self.classes_)
                probs = np.ones((len(X), n)) / float(n)
                return probs
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
        fallback_classes = models["metadata"].get("classes") or ["Tomatoes", "Potatoes", "Lettuce"]
        models["model"] = DummyModel(fallback_classes)
        models["metadata"]["classes"] = list(models["model"].classes_)

    # Ensure target_encoder mapping if not present, create a simple one matching metadata classes
    if models["target_encoder"] is None:
        class SimpleLE:
            def __init__(self, classes):
                self.classes_ = np.asarray(classes)
                self._map = {c: i for i, c in enumerate(self.classes_)}
            def transform(self, arr):
                return np.array([self._map.get(x, 0) for x in arr])
            def inverse_transform(self, ints):
                return np.array([self.classes_[int(i)] for i in ints])
        models["target_encoder"] = SimpleLE(models["metadata"].get("classes", list(models["model"].classes_)))
        models["metadata"]["classes"] = list(models["target_encoder"].classes_)

    return models

# ----------------- Dataset helpers -----------------
@st.cache_data
def read_dataset_first_available() -> pd.DataFrame:
    data_paths = [BASE / "soil.impact.csv", BASE / "data" / "crop_data_clean.csv"]
    for p in data_paths:
        try:
            if p.exists():
                return pd.read_csv(p)
        except Exception:
            continue
    # fallback small dataset
    return pd.DataFrame([
        ["Tomatoes", "neutral", "Loamy"],
        ["Potatoes", "depleting", "Loamy"],
        ["Lettuce", "restorative", "Sandy"],
    ], columns=["Name", "Impact", "Soil_Type"])

@st.cache_data
def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive statistical analysis of the dataset."""
    stats = {
        "basic_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum()
        },
        "numerical_stats": {},
        "categorical_stats": {},
        "correlations": None
    }
    
    # Numerical columns analysis
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        stats["numerical_stats"] = df[num_cols].describe()
        stats["correlations"] = df[num_cols].corr()
    
    # Categorical columns analysis
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        most_common = value_counts.head(5).to_dict()
        stats["categorical_stats"][col] = {
            "unique_values": unique_count,
            "most_common": most_common
        }
    
    return stats

@st.cache_data
@st.cache_data
def compute_data_ranges() -> Dict[str, Dict[str, float]]:
    df = read_dataset_first_available()
    ranges = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        ranges[col] = {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
    # defaults matching actual training data distribution from soil.impact.csv
    defaults = {
        "Temperature": {"min": 9.4, "max": 40.0, "mean": 20.8},
        "Rainfall": {"min": 410.0, "max": 2510.0, "mean": 949.0},
        "pH": {"min": 4.5, "max": 8.5, "mean": 6.5},
        "Light_Hours": {"min": 5.0, "max": 16.0, "mean": 10.0},
        "Light_Intensity": {"min": 69.0, "max": 986.0, "mean": 398.0},
        "Rh": {"min": 30.0, "max": 100.0, "mean": 70.0},
        "Nitrogen": {"min": 40, "max": 410, "mean": 140},
        "Phosphorus": {"min": 13, "max": 360, "mean": 108},
        "Potassium": {"min": 35, "max": 580, "mean": 180},
        "Yield": {"min": 0.5, "max": 67.0, "mean": 22.0}
    }
    for k, v in defaults.items():
        ranges.setdefault(k, v)
    return ranges

@st.cache_data
def get_valid_categorical_values() -> Dict[str, List[str]]:
    """Get valid categorical values from dataset with caching."""
    df = read_dataset_first_available()
    valid_values = {}
    for col in ("Soil_Type", "Season", "Fertility", "Impact"):
        if col in df.columns:
            valid_values[col] = list(pd.unique(df[col].astype(str)))
    valid_values.setdefault("Soil_Type", ["Loam", "Sandy", "Clay"])
    valid_values.setdefault("Season", ["Spring", "Summer", "Autumn", "Winter"])
    valid_values.setdefault("Fertility", ["Low", "Moderate", "High"])
    valid_values.setdefault("Impact", ["depleting", "restorative", "neutral", "enriching"])
    return valid_values

# ----------------- Encoding & Feature Vector -----------------
def safe_encode_column(col_name: str, value: Any, models_dict: Dict[str, Any]) -> int:
    """Encode a categorical value using available encoders or deterministic fallback."""
    # try per-column encoder
    label_encoders = models_dict.get("label_encoders", {}) or {}
    target_encoder = models_dict.get("target_encoder")
    if col_name in label_encoders and label_encoders[col_name] is not None:
        try:
            out = label_encoders[col_name].transform([value])
            return int(out[0])
        except Exception:
            pass
    # try target_encoder if it contains the categories
    try:
        if target_encoder is not None:
            # some encoders expect array-like
            out = target_encoder.transform([value])
            return int(out[0])
    except Exception:
        pass
    # try to map from dataset
    try:
        df = read_dataset_first_available()
        if col_name in df.columns:
            uniques = list(pd.unique(df[col_name].astype(str)))
            mapping = {v: i for i, v in enumerate(uniques)}
            return int(mapping.get(str(value), 0))
    except Exception:
        pass
    # deterministic hash fallback
    if isinstance(value, str):
        return int(abs(hash(value)) % 1000)
    try:
        return int(value)
    except Exception:
        return 0

def build_feature_vector(input_data: Dict[str, Any], models_dict: Dict[str, Any]) -> np.ndarray:
    """
    Build feature vector in the exact order expected by the model.
    Optimized for performance - no debug overhead.
    """
    feature_cols = models_dict.get("metadata", {}).get("feature_cols") or CANONICAL_FEATURES
    row = []
    
    for feat in feature_cols:
        val = input_data.get(feat, 0)
        if feat in ("Fertility", "Soil_Type", "Season", "Impact"):
            encoded = safe_encode_column(feat, val, models_dict)
            row.append(float(encoded))
        else:
            try:
                row.append(float(val))
            except Exception:
                row.append(0.0)
    
    X = np.atleast_2d(np.asarray(row, dtype=float))
    
    # Apply scaler if present
    scaler = models_dict.get("scaler")
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    
    return X

def get_top_3_predictions(models_dict: Dict[str, Any], input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get top 3 predictions with temperature-scaled probabilities for better confidence."""
    model = models_dict.get("model")
    if model is None:
        st.error("Model artifact missing.")
        return None

    X = build_feature_vector(input_data, models_dict)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            
            # Apply temperature scaling to improve confidence spread
            # Lower temperature (0.5-0.7) makes the model more confident
            temperature = 0.6
            scaled_logits = np.log(probs + 1e-10) / temperature
            # Re-normalize with softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs_scaled = exp_logits / exp_logits.sum()
            
            top_3_idx = np.argsort(probs_scaled)[::-1][:3]

            # Get class labels
            classes = []
            te = models_dict.get("target_encoder")
            if te is not None:
                try:
                    classes = list(te.classes_)
                except Exception:
                    classes = models_dict.get("metadata", {}).get("classes", [])
            if not classes:
                classes = [f"class_{i}" for i in range(probs_scaled.shape[0])]

            results = []
            for rank, idx in enumerate(top_3_idx, start=1):
                score = float(probs_scaled[idx])
                results.append({
                    "rank": rank,
                    "crop": classes[idx] if idx < len(classes) else str(idx),
                    "score": score,
                    "raw_score": float(probs[idx])  # keep original for debugging
                })
            return results
        else:
            preds = model.predict(X)[0]
            return [{"rank": 1, "crop": str(preds), "score": 1.0, "raw_score": 1.0}]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# ----------------- Rotation Plan -----------------
def generate_rotation_plan(years: int, crops_per_year: int, soil_type: str, season: str) -> Dict[int, List[Dict[str, str]]]:
    """Generate year-wise rotation plan."""
    try:
        df = read_dataset_first_available()
        if "Soil_Type" in df.columns:
            pool = df[df["Soil_Type"] == soil_type]
            if pool.empty:
                pool = df
        else:
            pool = df

        crops = pool["Name"].unique().tolist() if "Name" in pool.columns else ["Tomatoes", "Potatoes", "Lettuce"]
        
        # Organize by year
        plan = {}
        for year in range(1, years + 1):
            year_plan = []
            for season_num in range(crops_per_year):
                idx = ((year-1) * crops_per_year + season_num) % len(crops)
                crop = crops[idx]
                crop_info = {
                    "Season": f"Season {season_num + 1}",
                    "Crop": crop,
                    "Soil_Type": soil_type
                }
                year_plan.append(crop_info)
            plan[year] = year_plan
        return plan
    except Exception as e:
        st.error(f"Rotation generation failed: {e}")
        return None


def render_crop_card(crop: str, score: float) -> str:
        """Return HTML for a crop recommendation card."""
        pct = int(round(score * 100)) if score is not None else 0
        emoji = get_crop_emoji(crop)
        # simple progress bar using inline styles
        bar = f"<div style='background:#eee;border-radius:8px;height:8px;width:100%'><div style='width:{pct}%;height:8px;background:linear-gradient(90deg,#4caf50,#2e7d32);border-radius:8px'></div></div>"
        html = f"""
        <div class='crop-card'>
            <div class='crop-emoji'>{emoji}</div>
            <div class='crop-name'>{crop}</div>
            <div class='crop-score'>{pct}% confidence</div>
            <div style='margin-top:8px'>{bar}</div>
        </div>
        """
        return html

# ----------------- Main App -----------------
def main():
    models_dict = load_app_models()
    df_sample = read_dataset_first_available()
    data_ranges = compute_data_ranges()

    # Sidebar — quick app info + model status
    with st.sidebar:
        st.markdown("<div style='text-align:center;font-size:2.2rem'>🌾</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;margin:0'>SmartCrop Advisor</h3>", unsafe_allow_html=True)
        md = models_dict.get("metadata", {})
        st.write(f"**Model:** {md.get('best_model', 'Unknown')}")
        try:
            st.write(f"**Accuracy:** {float(md.get('accuracy',0.0)):.2f}")
        except Exception:
            st.write(f"**Accuracy:** {md.get('accuracy',0.0)}")
        st.markdown("---")
        st.markdown("<small>Quick actions</small>", unsafe_allow_html=True)
        if st.button("Retrain model (run main.py)", key='retrain'):
            st.warning("Run training locally: python main.py — this UI cannot run training from Streamlit safely.")
        st.markdown("<small style='color:var(--muted)'>Tip: run training to populate `saved_models/` for better predictions.</small>", unsafe_allow_html=True)

    # Get valid categorical values (cached)
    valid_values = get_valid_categorical_values()

    st.markdown("<div class='main-header'>🌾 Crop Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#666'>Optimizing Crop Selection for Small-Scale Farmers</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌱 Crop Recommendation", "🔄 Crop Rotation Plan", "📊 Dataset Info"])

    with tab1:
        st.markdown('<div class="sub-header">Inputs</div>', unsafe_allow_html=True)
        
        with st.form(key='prediction_form'):
            left, right = st.columns([2, 1])

            with left:
                temperature = st.slider("Temperature (°C)", float(data_ranges["Temperature"]["min"]),
                                        float(data_ranges["Temperature"]["max"]), float(data_ranges["Temperature"]["mean"]))
                rainfall = st.slider("Rainfall (mm)", float(data_ranges["Rainfall"]["min"]),
                                     float(data_ranges["Rainfall"]["max"]), float(data_ranges["Rainfall"]["mean"]), step=10.0)
                ph = st.slider("pH", float(data_ranges["pH"]["min"]), float(data_ranges["pH"]["max"]), float(data_ranges["pH"]["mean"]), step=0.1)
                light_hours = st.slider("Light Hours per Day", float(data_ranges["Light_Hours"]["min"]),
                                        float(data_ranges["Light_Hours"]["max"]), float(data_ranges["Light_Hours"]["mean"]), step=0.1)
                light_intensity = st.slider("Light Intensity", float(data_ranges["Light_Intensity"]["min"]),
                                            float(data_ranges["Light_Intensity"]["max"]), float(data_ranges["Light_Intensity"]["mean"]), step=10.0)
                rh = st.slider("Relative Humidity (%)", float(data_ranges["Rh"]["min"]), float(data_ranges["Rh"]["max"]), float(data_ranges["Rh"]["mean"]), step=1.0)

            with right:
                nitrogen = st.number_input("Nitrogen (mg/ha)", int(data_ranges["Nitrogen"]["min"]), int(data_ranges["Nitrogen"]["max"]), int(data_ranges["Nitrogen"]["mean"]))
                phosphorus = st.number_input("Phosphorus (mg/ha)", int(data_ranges["Phosphorus"]["min"]), int(data_ranges["Phosphorus"]["max"]), int(data_ranges["Phosphorus"]["mean"]))
                potassium = st.number_input("Potassium (mg/ha)", int(data_ranges["Potassium"]["min"]), int(data_ranges["Potassium"]["max"]), int(data_ranges["Potassium"]["mean"]))
                yield_val = st.number_input("Expected Yield", float(data_ranges["Yield"]["min"]), float(data_ranges["Yield"]["max"]), float(data_ranges["Yield"]["mean"]), step=0.1)

                fertility = st.selectbox("Fertility", valid_values["Fertility"])
                soil_type = st.selectbox("Soil Type", valid_values["Soil_Type"])
                season = st.selectbox("Season", valid_values["Season"])
                impact = st.selectbox("Impact", valid_values["Impact"])

            # Form submit button
            submit_button = st.form_submit_button("🌱 Get Crop Recommendations", use_container_width=True)
        
        # Process after form submission
        if submit_button:
            input_data = {
                "Temperature": temperature,
                "Rainfall": rainfall,
                "Light_Intensity": light_intensity,
                "Nitrogen": nitrogen,
                "Phosphorus": phosphorus,
                "Potassium": potassium,
                "Season": season,
                "Soil_Type": soil_type,
                "Impact": impact,
                "Fertility": fertility
            }
            with st.spinner("Analyzing..."):
                results = get_top_3_predictions(models_dict, input_data)
            if results is None:
                st.error("Prediction failed. Consider retraining the models.")
            else:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("<h3 style='text-align:center;color:#2E7D32'>Recommended Crops</h3>", unsafe_allow_html=True)

                # Display as styled cards
                cols = st.columns(len(results))
                for c, res in zip(cols, results):
                    with c:
                        html = render_crop_card(res.get('crop', 'Unknown'), res.get('score', 0.0))
                        st.markdown(html, unsafe_allow_html=True)

                # allow download of results
                try:
                    import json
                    csv_bytes = json.dumps(results, indent=2).encode('utf-8')
                    st.download_button("Download recommendations (JSON)", data=csv_bytes, file_name="recommendations.json", mime='application/json')
                except Exception:
                    pass
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="sub-header">Generate Rotation Plan</div>', unsafe_allow_html=True)
        years = st.slider("Years", 2, 5, 3)
        crops_per_year = st.slider("Crops per year", 1, 4, 2)
        rot_soil = st.selectbox("Soil Type (rotation)", valid_values["Soil_Type"])
        rot_season = st.selectbox("Primary Season", valid_values["Season"])
        if st.button("Generate Rotation"):
            plan = generate_rotation_plan(years, crops_per_year, rot_soil, rot_season)
            if plan:
                st.markdown("<h3 style='text-align:center;color:#2E7D32'>Year-wise Rotation Plan</h3>", 
                   unsafe_allow_html=True)
                for year, crops in plan.items():
                    st.markdown(f"### Year {year}")
                    df_year = pd.DataFrame(crops)
                    st.dataframe(df_year, use_container_width=True)
                    st.markdown("---")

    with tab3:
        st.markdown('<div class="sub-header">Dataset & Model Information</div>', unsafe_allow_html=True)
        df = read_dataset_first_available()
        stats = get_dataset_statistics(df)
        
        # Basic Information
        st.markdown("### 📊 Basic Information")
        basic_info = stats["basic_info"]
        cols = st.columns(4)
        cols[0].metric("Total Rows", basic_info["rows"])
        cols[1].metric("Total Columns", basic_info["columns"])
        cols[2].metric("Missing Values", basic_info["missing_values"])
        cols[3].metric("Duplicate Rows", basic_info["duplicate_rows"])
        
        # Numerical Analysis
        if stats["numerical_stats"] is not None:
            st.markdown("### 📈 Numerical Features Analysis")
            st.dataframe(stats["numerical_stats"], use_container_width=True)
            
            # Correlation Matrix
            if stats["correlations"] is not None:
                st.markdown("#### Correlation Matrix")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(stats["correlations"], annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
        
        # Categorical Analysis
        if stats["categorical_stats"]:
            st.markdown("### 🧾 Categorical Features Analysis")
            for col, info in stats["categorical_stats"].items():
                st.markdown(f"#### {col}")
                st.write(f"Unique values: {info.get('unique_values', 0)}")
                most_common = info.get("most_common", {})
                if most_common:
                    mc_df = pd.DataFrame(list(most_common.items()), columns=[col, "count"])
                    st.dataframe(mc_df, use_container_width=True)
                else:
                    st.write("No categorical distribution available.")
        
        # Model metadata
        st.markdown("### 🧠 Model Metadata")
        md = models_dict.get("metadata", {})
        st.write(f"Best model: {md.get('best_model', 'Unknown')}")
        try:
            acc = float(md.get("accuracy", 0.0))
            st.write(f"Training accuracy: {acc:.2f}")
        except Exception:
            st.write(f"Training accuracy: {md.get('accuracy', 0.0)}")
        st.write(f"Classes: {md.get('classes', [])}")

if __name__ == "__main__":
    main()
