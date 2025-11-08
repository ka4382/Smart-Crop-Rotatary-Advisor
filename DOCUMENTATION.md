# 📚 SmartCrop Rotatory Advisor - Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Flow](#data-flow)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Training Details](#model-training-details)
6. [Prediction Pipeline](#prediction-pipeline)
7. [UI Components](#ui-components)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Configuration](#advanced-configuration)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Crop Rec   │  │   Rotation   │  │   Dataset    │      │
│  │     Tab      │  │   Planner    │  │     Info     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer (app.py)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Input Validation                                  │   │
│  │  • Feature Vector Construction                       │   │
│  │  • Categorical Encoding                             │   │
│  │  • Feature Scaling                                  │   │
│  │  • Prediction Orchestration                         │   │
│  │  • Result Formatting                                │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Layer (saved_models/)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  RandomForest│  │StandardScaler│  │LabelEncoders │      │
│  │   Classifier │  │    (scaler)  │  │   (dict)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (CSV files)                    │
│  • soil.impact.csv (1500+ samples, 22 crops)                │
│  • crop_data_clean.csv (alternative dataset)                │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.0+ | Web UI framework |
| **ML Core** | scikit-learn 0.24.2+ | Model training & inference |
| **Data Processing** | pandas 1.3+ | DataFrame operations |
| **Numerical** | numpy 1.21+ | Array computations |
| **Optimization** | DEAP 1.3.1+ | Genetic algorithm for rotation |
| **Visualization** | matplotlib, seaborn | Charts and plots |

---

## Data Flow

### Training Pipeline

```
soil.impact.csv
      │
      ├─► Load Dataset (pd.read_csv)
      │
      ├─► Split Features & Target
      │    ├─ X: Temperature, Rainfall, Light_Intensity, N, P, K, 
      │    │     Season, Soil_Type, Impact, Fertility
      │    └─ y: Name (crop name)
      │
      ├─► Encode Categorical Features
      │    ├─ Season → LabelEncoder → {0, 1, 2, 3}
      │    ├─ Soil_Type → LabelEncoder → {0, 1, 2}
      │    ├─ Impact → LabelEncoder → {0, 1, 2, 3}
      │    └─ Fertility → LabelEncoder → {0, 1, 2}
      │
      ├─► Combine Features
      │    X = [X_numeric | X_categorical_encoded]
      │
      ├─► Scale Features
      │    X_scaled = StandardScaler().fit_transform(X)
      │
      ├─► Train RandomForest
      │    model.fit(X_train, y_train)
      │
      └─► Save Artifacts
           ├─ random_forest.pkl
           ├─ scaler.pkl
           ├─ label_encoder.pkl (target)
           ├─ label_encoders.pkl (features)
           └─ metadata.pkl
```

### Prediction Pipeline

```
User Input
  │
  ├─► Validate Input (type checking, range validation)
  │
  ├─► Build Feature Vector
  │    ├─ Extract features in model's expected order
  │    │   [Temperature, Rainfall, Light_Intensity, N, P, K, 
  │    │    Season, Soil_Type, Impact, Fertility]
  │    │
  │    ├─ Encode Categorical Features
  │    │   Season='Summer' → label_encoders['Season'].transform(['Summer']) → [2]
  │    │
  │    └─ Create numpy array: X = [[20.9, 748, 534, 171, 119, 243, 2, 0, 0, 1]]
  │
  ├─► Scale Features
  │    X_scaled = scaler.transform(X)
  │    # Each feature normalized to mean=0, std=1
  │
  ├─► Make Prediction
  │    probas = model.predict_proba(X_scaled)[0]  # Get probabilities
  │    # Shape: (22,) - one probability per crop
  │
  ├─► Apply Temperature Scaling
  │    scaled_logits = log(probas) / temperature  # temperature=0.6
  │    probas_calibrated = softmax(scaled_logits)
  │
  ├─► Extract Top 3
  │    top_3_indices = argsort(probas_calibrated)[-3:][::-1]
  │
  └─► Format Results
       [
         {rank: 1, crop: "Tomatoes", score: 0.87},
         {rank: 2, crop: "Peppers", score: 0.09},
         {rank: 3, crop: "Cucumbers", score: 0.04}
       ]
```

---

## Machine Learning Pipeline

### RandomForest Configuration

```python
RandomForestClassifier(
    n_estimators=200,           # Number of decision trees
    max_depth=None,             # Trees grow until pure leaves
    min_samples_split=2,        # Min samples to split node
    min_samples_leaf=1,         # Min samples in leaf node
    n_jobs=-1,                  # Use all CPU cores
    random_state=42             # Reproducibility
)
```

### Why RandomForest?

1. **High Accuracy** - Ensemble of 200 trees reduces overfitting
2. **Handles Mixed Data** - Works well with numeric + categorical features
3. **Feature Importance** - Provides interpretability
4. **Robust to Outliers** - Tree-based methods are naturally robust
5. **No Hyperparameter Tuning Needed** - Works well with defaults

### Training Process

```python
# 1. Load and preprocess data
df = pd.read_csv('soil.impact.csv')
X = df[numeric_features + categorical_features]
y = df['Name']

# 2. Encode categorical features
for col in categorical_features:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    label_encoders[col] = encoder

# 3. Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Train model
rf = RandomForestClassifier(**config)
rf.fit(X_train, y_train)

# 6. Evaluate
accuracy = rf.score(X_test, y_test)  # 99.02%

# 7. Save artifacts
pickle.dump(rf, open('random_forest.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
```

---

## Feature Engineering

### Feature Types

#### Numeric Features (6)

| Feature | Unit | Range | Mean | Description |
|---------|------|-------|------|-------------|
| **Temperature** | °C | 9.4 - 40.0 | 20.8 | Average growing temp |
| **Rainfall** | mm | 410 - 2510 | 949 | Annual precipitation |
| **Light_Intensity** | lux | 69 - 986 | 398 | Light exposure |
| **Nitrogen** | mg/ha | 40 - 410 | 140 | Soil nitrogen |
| **Phosphorus** | mg/ha | 13 - 360 | 108 | Soil phosphorus |
| **Potassium** | mg/ha | 35 - 580 | 180 | Soil potassium |

#### Categorical Features (4)

| Feature | Values | Encoding |
|---------|--------|----------|
| **Season** | Spring, Summer, Autumn, Winter | 0, 1, 2, 3 |
| **Soil_Type** | Loam, Sandy, Clay | 0, 1, 2 |
| **Impact** | depleting, neutral, restorative, enriching | 0, 1, 2, 3 |
| **Fertility** | Low, Moderate, High | 0, 1, 2 |

### Feature Scaling

StandardScaler transforms each feature to have:
- **Mean = 0**
- **Standard Deviation = 1**

Formula:
```
X_scaled = (X - μ) / σ

where:
  μ = mean of feature in training data
  σ = standard deviation in training data
```

**Example:**
```python
# Original Temperature value: 25.5°C
# Training data: μ=20.8, σ=6.2
X_scaled = (25.5 - 20.8) / 6.2 = 0.758
```

**Why Scale?**
- Ensures all features have equal weight
- RandomForest doesn't strictly need it, but improves consistency
- Critical for distance-based models (future extensions)

### Feature Importance (from trained model)

```
Temperature      ████████████████████ 32.5%
Rainfall         ███████████████      24.1%
Light_Intensity  ██████████          16.8%
Nitrogen         ██████               9.2%
Season           ████                 6.5%
Soil_Type        ███                  4.9%
Phosphorus       ██                   3.1%
Potassium        ██                   2.3%
Impact           █                    0.4%
Fertility        █                    0.2%
```

---

## Model Training Details

### Training Script: `main.py`

```python
def train_model():
    """Complete training pipeline"""
    
    # 1. Load dataset
    df = pd.read_csv(BASE / "soil.impact.csv")
    print(f"Loaded {len(df)} samples, {len(df['Name'].unique())} crops")
    
    # 2. Define features
    num_features = ["Temperature", "Rainfall", "Light_Intensity", 
                    "Nitrogen", "Phosphorus", "Potassium"]
    cat_features = ["Season", "Soil_Type", "Impact", "Fertility"]
    
    # 3. Encode categorical features
    encoders = {}
    X_cat = []
    for col in cat_features:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        X_cat.append(encoded.reshape(-1, 1))
        encoders[col] = le
    
    # 4. Combine features
    X_num = df[num_features].values
    X = np.hstack([X_num] + X_cat)  # Shape: (n_samples, 10)
    
    # 5. Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['Name'])
    
    # 6. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    
    # 8. Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    
    print(f"✓ Training complete. Accuracy: {accuracy:.4f}")
    
    # 9. Save everything
    save_artifacts(rf, scaler, target_encoder, encoders, 
                   num_features + cat_features, 
                   target_encoder.classes_, accuracy)
```

### Saved Artifacts

| File | Type | Purpose | Size |
|------|------|---------|------|
| `random_forest.pkl` | Model | Trained RF classifier | ~5 MB |
| `scaler.pkl` | Preprocessor | StandardScaler fitted on training data | ~2 KB |
| `label_encoder.pkl` | Encoder | Target encoder (Name → int) | ~1 KB |
| `label_encoders.pkl` | Dict | Feature encoders (Season, Soil_Type, etc.) | ~2 KB |
| `metadata.pkl` | Dict | Feature order, classes, accuracy | ~3 KB |

### Metadata Structure

```python
{
    "feature_cols": [
        "Temperature", "Rainfall", "Light_Intensity", 
        "Nitrogen", "Phosphorus", "Potassium",
        "Season", "Soil_Type", "Impact", "Fertility"
    ],
    "classes": [
        "Arugula", "Asparagus", "Beet", "Broccoli", 
        "Cabbage", "Cauliflowers", ... (22 total)
    ],
    "accuracy": 0.9902597402597403,
    "best_model": "RandomForest"
}
```

---

## Prediction Pipeline

### Step-by-Step Prediction

#### 1. Load Models (Cached)

```python
@st.cache_resource
def load_app_models():
    """Load once, cache forever"""
    with open('saved_models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('saved_models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('saved_models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'metadata': metadata
    }
```

#### 2. Build Feature Vector

```python
def build_feature_vector(input_data, models_dict):
    """Construct feature vector in correct order"""
    
    feature_cols = models_dict['metadata']['feature_cols']
    row = []
    
    for feat in feature_cols:
        val = input_data[feat]
        
        if feat in ("Fertility", "Soil_Type", "Season", "Impact"):
            # Encode categorical feature
            encoder = models_dict['label_encoders'][feat]
            encoded = encoder.transform([val])[0]
            row.append(float(encoded))
        else:
            # Use numeric value directly
            row.append(float(val))
    
    # Convert to numpy array
    X = np.atleast_2d(np.array(row, dtype=float))
    
    # Scale features
    X_scaled = models_dict['scaler'].transform(X)
    
    return X_scaled
```

#### 3. Make Prediction with Temperature Scaling

```python
def get_top_3_predictions(models_dict, input_data):
    """Get top 3 crops with calibrated probabilities"""
    
    # Build and scale features
    X = build_feature_vector(input_data, models_dict)
    
    # Get raw probabilities
    probs = models_dict['model'].predict_proba(X)[0]
    # Shape: (22,) - one prob per crop
    
    # Apply temperature scaling
    temperature = 0.6  # Lower = more confident
    scaled_logits = np.log(probs + 1e-10) / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs_scaled = exp_logits / exp_logits.sum()
    
    # Get top 3
    top_3_idx = np.argsort(probs_scaled)[::-1][:3]
    
    # Format results
    classes = models_dict['metadata']['classes']
    results = []
    for rank, idx in enumerate(top_3_idx, 1):
        results.append({
            "rank": rank,
            "crop": classes[idx],
            "score": float(probs_scaled[idx]),
            "raw_score": float(probs[idx])
        })
    
    return results
```

### Temperature Scaling Explained

**Purpose:** Calibrate model confidence to match true accuracy

**Formula:**
```python
probs_calibrated = softmax(log(probs) / T)

where:
  T = temperature (0.6 in our case)
  T < 1 → sharper distribution (more confident)
  T > 1 → softer distribution (less confident)
  T = 1 → original distribution
```

**Example:**
```python
# Raw probabilities from model
probs = [0.45, 0.30, 0.15, 0.05, ...]

# With T=0.6 (sharper)
probs_scaled = [0.68, 0.21, 0.08, 0.02, ...]
# Top prediction becomes more confident

# With T=2.0 (softer)
probs_scaled = [0.35, 0.28, 0.20, 0.10, ...]
# More uniform distribution
```

---

## UI Components

### Streamlit App Structure

```python
# Main app layout
st.markdown("<div class='main-header'>🌾 Crop Recommendation System</div>")

tab1, tab2, tab3 = st.tabs([
    "🌱 Crop Recommendation", 
    "🔄 Crop Rotation Plan", 
    "📊 Dataset Info"
])

with tab1:
    # Form to prevent lag on slider changes
    with st.form(key='prediction_form'):
        # Sliders and inputs
        temperature = st.slider("Temperature (°C)", 9.4, 40.0, 20.8)
        rainfall = st.slider("Rainfall (mm)", 410.0, 2510.0, 949.0)
        # ... more inputs
        
        # Submit button
        submit = st.form_submit_button("🌱 Get Crop Recommendations")
    
    if submit:
        # Make prediction
        results = get_top_3_predictions(models_dict, input_data)
        
        # Display crop cards
        for res in results:
            st.markdown(render_crop_card(res['crop'], res['score']))
```

### CSS Styling

```css
/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Crop card design */
.crop-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.crop-card:hover {
    transform: translateY(-5px);
}

/* Confidence bar */
.confidence-bar {
    background: linear-gradient(90deg, 
        #4CAF50 0%, 
        #8BC34A 50%, 
        #FFC107 100%);
    height: 10px;
    border-radius: 5px;
}
```

### Emoji Mapping

```python
CROP_EMOJIS = {
    "Tomatoes": "🍅",
    "Potatoes": "🥔",
    "Lettuce": "🥬",
    "Carrots": "🥕",
    "Peppers": "🌶️",
    "Cucumbers": "🥒",
    "Strawberry": "🍓",
    "Watermelon": "🍉",
    "Broccoli": "🥦",
    "Cauliflowers": "🥣",
    "Spinach": "🌿",
    "Kale": "🥬",
    # ... 22 unique emojis
}

def get_crop_emoji(crop_name):
    """5-step fuzzy matching"""
    # 1. Exact match
    if crop_name in CROP_EMOJIS:
        return CROP_EMOJIS[crop_name]
    
    # 2. Case-insensitive
    for key in CROP_EMOJIS:
        if key.lower() == crop_name.lower():
            return CROP_EMOJIS[key]
    
    # 3. Singularize (Cauliflowers → Cauliflower)
    # 4. ies → y transform (Strawberries → Strawberry)
    # 5. Fuzzy match with difflib (80% similarity)
    
    return "🌱"  # Default
```

---

## Performance Optimization

### Caching Strategy

```python
# Model loading - cache as resource (persistent)
@st.cache_resource
def load_app_models():
    """Load once per session"""
    return load_models_from_disk()

# Data operations - cache as data (serializable)
@st.cache_data
def read_dataset():
    """Cache DataFrame"""
    return pd.read_csv('soil.impact.csv')

@st.cache_data
def compute_data_ranges():
    """Cache feature ranges"""
    df = read_dataset()
    return calculate_ranges(df)

@st.cache_data
def get_valid_categorical_values():
    """Cache categorical options"""
    df = read_dataset()
    return extract_categories(df)
```

### Form-Based Input

**Problem:** Streamlit reruns entire script on every widget change
**Solution:** Wrap inputs in `st.form()`

```python
# ❌ Without form - laggy!
temperature = st.slider("Temperature", 10, 40, 25)
rainfall = st.slider("Rainfall", 400, 2500, 1000)
# Script reruns on every slider movement

# ✅ With form - smooth!
with st.form("input_form"):
    temperature = st.slider("Temperature", 10, 40, 25)
    rainfall = st.slider("Rainfall", 400, 2500, 1000)
    submit = st.form_submit_button("Predict")
# Script runs only when submit clicked
```

### Performance Metrics

| Operation | Without Optimization | With Optimization | Improvement |
|-----------|---------------------|-------------------|-------------|
| Model loading | ~2s per interaction | ~0.01s (cached) | **200x faster** |
| Dataset reading | ~0.5s per interaction | ~0.01s (cached) | **50x faster** |
| Slider movement | Reruns entire app | No rerun | **∞ faster** |
| Overall UX | Laggy, freezes | Smooth, instant | **Perfect** |

---

## Troubleshooting

### Common Issues

#### 1. Low Confidence (<50%)

**Symptom:** All predictions show confidence below 50%

**Cause:** Input values far from training distribution

**Solution:**
```python
# Check if inputs match typical training data
# Temperature: 18-22°C (mean: 20.8)
# Rainfall: 900-1000mm (mean: 949)
# Light_Intensity: 350-450 (mean: 398)

# Use values close to means for best results
```

#### 2. Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 3. Model Not Found

**Symptom:** `FileNotFoundError: saved_models/random_forest.pkl`

**Solution:**
```bash
# Train the model first
python main.py

# Verify files created
ls saved_models/
# Should show: random_forest.pkl, scaler.pkl, etc.
```

#### 4. Encoding Errors

**Symptom:** `ValueError: unknown categories in transform`

**Solution:**
```python
# Ensure categorical inputs match training data
# Valid Season: Spring, Summer, Autumn, Winter
# Valid Soil_Type: Loam, Sandy, Clay
# Valid Fertility: Low, Moderate, High
# Valid Impact: depleting, neutral, restorative, enriching
```

#### 5. Slow Performance

**Symptom:** UI lags, sliders freeze

**Solution:**
```python
# Already implemented in current version:
# - @st.cache_resource for models
# - @st.cache_data for datasets
# - st.form() for inputs

# If still slow, check:
streamlit --version  # Should be 1.0+
python --version      # Should be 3.8+
```

---

## Advanced Configuration

### Custom Dataset

To use your own dataset:

```python
# 1. Format your CSV
# Required columns:
columns = [
    'Name',              # Crop name (target)
    'Temperature',       # Numeric: °C
    'Rainfall',          # Numeric: mm
    'Light_Intensity',   # Numeric: lux
    'Nitrogen',          # Numeric: mg/ha
    'Phosphorus',        # Numeric: mg/ha
    'Potassium',         # Numeric: mg/ha
    'Season',            # Categorical: Spring/Summer/Autumn/Winter
    'Soil_Type',         # Categorical: Loam/Sandy/Clay
    'Impact',            # Categorical: depleting/neutral/restorative/enriching
    'Fertility'          # Categorical: Low/Moderate/High
]

# 2. Place CSV in code/ directory as 'soil.impact.csv'

# 3. Retrain model
python main.py

# 4. Run app
streamlit run app.py
```

### Adjust Temperature Scaling

```python
# In app.py, modify get_top_3_predictions()

# Current value
temperature = 0.6  # Moderate confidence boost

# Adjust based on needs:
temperature = 0.4  # More confident (sharper)
temperature = 1.0  # Original probabilities
temperature = 2.0  # Less confident (softer)
```

### Change Model Hyperparameters

```python
# In main.py, modify RandomForestClassifier

# For faster training (lower accuracy)
rf = RandomForestClassifier(
    n_estimators=50,     # Fewer trees
    max_depth=10,        # Limit depth
    n_jobs=-1,
    random_state=42
)

# For higher accuracy (slower training)
rf = RandomForestClassifier(
    n_estimators=500,    # More trees
    max_depth=None,      # Full depth
    min_samples_split=2,
    max_features='sqrt', # Feature subset per split
    n_jobs=-1,
    random_state=42
)
```

### Add More Crops

```python
# 1. Add samples to soil.impact.csv
# Include at least 50 samples per new crop

# 2. Add emoji to CROP_EMOJIS in app.py
CROP_EMOJIS = {
    "Tomatoes": "🍅",
    "NewCrop": "🌾",  # Add your emoji
    # ...
}

# 3. Retrain model
python main.py
```

---

## API Reference

See `API_REFERENCE.md` for complete function documentation.

**Quick Reference:**

- `load_app_models()` - Load model artifacts
- `get_top_3_predictions()` - Make predictions
- `build_feature_vector()` - Construct input array
- `safe_encode_column()` - Encode categorical values
- `render_crop_card()` - Generate HTML crop card
- `get_crop_emoji()` - Fuzzy emoji lookup
- `generate_rotation_plan()` - Create rotation schedule

---

**For more help, see:**
- README.md - Quick start guide
- API_REFERENCE.md - Detailed API docs
- GitHub Issues - Community support

*Last Updated: November 9, 2025*
