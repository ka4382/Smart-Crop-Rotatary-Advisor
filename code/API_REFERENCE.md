# 🔧 API Reference - SmartCrop Rotatory Advisor

Complete API documentation for all public functions and classes.

---

## Table of Contents

- [Model Loading](#model-loading)
- [Prediction Functions](#prediction-functions)
- [Feature Engineering](#feature-engineering)
- [Data Processing](#data-processing)
- [UI Rendering](#ui-rendering)
- [Utility Functions](#utility-functions)
- [Rotation Planning](#rotation-planning)

---

## Model Loading

### `load_app_models()`

Load and cache all trained model artifacts.

**Signature:**
```python
@st.cache_resource
def load_app_models() -> Dict[str, Any]
```

**Returns:**
```python
{
    "model": RandomForestClassifier,           # Trained model
    "scaler": StandardScaler,                  # Feature scaler
    "target_encoder": LabelEncoder,            # Target encoder
    "label_encoders": Dict[str, LabelEncoder], # Feature encoders
    "metadata": {
        "feature_cols": List[str],             # Feature order
        "classes": List[str],                  # Crop names
        "accuracy": float,                     # Test accuracy
        "best_model": str                      # Model name
    }
}
```

**Example:**
```python
models_dict = load_app_models()
print(f"Model accuracy: {models_dict['metadata']['accuracy']:.2%}")
print(f"Supported crops: {len(models_dict['metadata']['classes'])}")
```

**Notes:**
- Cached with `@st.cache_resource` - loads once per session
- Returns DummyModel if artifacts not found
- Automatically handles missing files gracefully

---

## Prediction Functions

### `get_top_3_predictions()`

Generate top 3 crop predictions with calibrated confidence scores.

**Signature:**
```python
def get_top_3_predictions(
    models_dict: Dict[str, Any], 
    input_data: Dict[str, Any]
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models_dict` | `Dict[str, Any]` | Model dictionary from `load_app_models()` |
| `input_data` | `Dict[str, Any]` | User input features |

**Input Data Format:**
```python
input_data = {
    "Temperature": 20.8,           # float: °C
    "Rainfall": 949.0,             # float: mm
    "Light_Intensity": 398.0,      # float: lux
    "Nitrogen": 140,               # int: mg/ha
    "Phosphorus": 108,             # int: mg/ha
    "Potassium": 180,              # int: mg/ha
    "Season": "Summer",            # str: Spring|Summer|Autumn|Winter
    "Soil_Type": "Loam",           # str: Loam|Sandy|Clay
    "Impact": "neutral",           # str: depleting|neutral|restorative|enriching
    "Fertility": "Moderate"        # str: Low|Moderate|High
}
```

**Returns:**
```python
[
    {
        "rank": 1,
        "crop": "Tomatoes",
        "score": 0.872,          # Calibrated probability (0-1)
        "raw_score": 0.654       # Original model probability
    },
    {
        "rank": 2,
        "crop": "Peppers",
        "score": 0.089,
        "raw_score": 0.123
    },
    {
        "rank": 3,
        "crop": "Cucumbers",
        "score": 0.039,
        "raw_score": 0.078
    }
]
```

**Example:**
```python
models_dict = load_app_models()

input_data = {
    "Temperature": 22.5,
    "Rainfall": 850.0,
    "Light_Intensity": 420.0,
    "Nitrogen": 150,
    "Phosphorus": 110,
    "Potassium": 200,
    "Season": "Summer",
    "Soil_Type": "Loam",
    "Impact": "neutral",
    "Fertility": "Moderate"
}

results = get_top_3_predictions(models_dict, input_data)

for res in results:
    print(f"{res['rank']}. {res['crop']}: {res['score']:.1%}")
```

**Output:**
```
1. Tomatoes: 87.2%
2. Peppers: 8.9%
3. Cucumbers: 3.9%
```

**Notes:**
- Applies temperature scaling (T=0.6) for calibrated probabilities
- Returns `None` if model is missing
- Confidence scores sum to ~100%

---

## Feature Engineering

### `build_feature_vector()`

Construct and scale feature vector in model's expected order.

**Signature:**
```python
def build_feature_vector(
    input_data: Dict[str, Any], 
    models_dict: Dict[str, Any]
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_data` | `Dict[str, Any]` | Raw input features |
| `models_dict` | `Dict[str, Any]` | Model artifacts with encoders |

**Returns:**
- `np.ndarray` - Shape `(1, 10)`, scaled feature vector

**Process:**
1. Extract features in correct order (from metadata)
2. Encode categorical features using LabelEncoders
3. Apply StandardScaler transformation
4. Return 2D numpy array

**Example:**
```python
input_data = {
    "Temperature": 20.8,
    "Season": "Summer",
    # ... other features
}

X = build_feature_vector(input_data, models_dict)
print(X.shape)  # (1, 10)
print(X[0])     # [0.01, -0.59, 0.71, ...]
```

**Internal Flow:**
```python
# Step 1: Extract in order
feature_cols = ['Temperature', 'Rainfall', ..., 'Fertility']
row = []
for feat in feature_cols:
    if feat in categorical_features:
        encoded = label_encoders[feat].transform([input_data[feat]])[0]
        row.append(float(encoded))
    else:
        row.append(float(input_data[feat]))

# Step 2: Convert to numpy
X = np.array([row])  # Shape: (1, 10)

# Step 3: Scale
X_scaled = scaler.transform(X)

return X_scaled
```

---

### `safe_encode_column()`

Safely encode categorical feature with fallback handling.

**Signature:**
```python
def safe_encode_column(
    col_name: str, 
    value: Any, 
    models_dict: Dict[str, Any]
) -> int
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `col_name` | `str` | Feature name (e.g., "Season") |
| `value` | `Any` | Categorical value (e.g., "Summer") |
| `models_dict` | `Dict` | Contains label_encoders |

**Returns:**
- `int` - Encoded value (0, 1, 2, ...)

**Encoding Strategy:**

1. **Try label_encoders[col_name]** - Primary encoder
2. **Try target_encoder** - Fallback if column encoder missing
3. **Dataset mapping** - Extract from dataset unique values
4. **Hash fallback** - Deterministic hash of string

**Example:**
```python
encoded = safe_encode_column("Season", "Summer", models_dict)
print(encoded)  # 2

# Encoding mapping (example):
# Spring  → 0
# Summer  → 1
# Autumn  → 2
# Winter  → 3
```

**Error Handling:**
```python
# Unknown value
encoded = safe_encode_column("Season", "InvalidSeason", models_dict)
# Returns: hash(value) % 1000 (deterministic fallback)
```

---

## Data Processing

### `read_dataset_first_available()`

Load dataset from available sources.

**Signature:**
```python
@st.cache_data
def read_dataset_first_available() -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame` - Loaded dataset

**Search Order:**
1. `soil.impact.csv`
2. `data/crop_data_clean.csv`
3. Fallback minimal DataFrame

**Example:**
```python
df = read_dataset_first_available()
print(f"Loaded {len(df)} samples")
print(df.columns.tolist())
```

**Output:**
```
Loaded 1547 samples
['Name', 'Fertility', 'Temperature', 'Rainfall', ...]
```

---

### `compute_data_ranges()`

Calculate feature ranges from dataset.

**Signature:**
```python
@st.cache_data
def compute_data_ranges() -> Dict[str, Dict[str, float]]
```

**Returns:**
```python
{
    "Temperature": {"min": 9.4, "max": 40.0, "mean": 20.8},
    "Rainfall": {"min": 410.0, "max": 2510.0, "mean": 949.0},
    "Light_Intensity": {"min": 69.0, "max": 986.0, "mean": 398.0},
    # ... all numeric features
}
```

**Example:**
```python
ranges = compute_data_ranges()
temp_range = ranges["Temperature"]
print(f"Temperature: {temp_range['min']}-{temp_range['max']}°C (avg: {temp_range['mean']}°C)")
```

**Output:**
```
Temperature: 9.4-40.0°C (avg: 20.8°C)
```

---

### `get_valid_categorical_values()`

Get valid categorical feature values from dataset.

**Signature:**
```python
@st.cache_data
def get_valid_categorical_values() -> Dict[str, List[str]]
```

**Returns:**
```python
{
    "Soil_Type": ["Loam", "Sandy", "Clay"],
    "Season": ["Spring", "Summer", "Autumn", "Winter"],
    "Fertility": ["Low", "Moderate", "High"],
    "Impact": ["depleting", "restorative", "neutral", "enriching"]
}
```

**Example:**
```python
valid_values = get_valid_categorical_values()
print(f"Valid seasons: {valid_values['Season']}")
```

**Output:**
```
Valid seasons: ['Spring', 'Summer', 'Autumn', 'Winter']
```

---

### `get_dataset_statistics()`

Generate comprehensive dataset statistics.

**Signature:**
```python
@st.cache_data
def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Dataset to analyze |

**Returns:**
```python
{
    "basic_info": {
        "rows": 1547,
        "columns": 14,
        "missing_values": 0,
        "duplicate_rows": 3
    },
    "numerical_stats": {
        "Temperature": {
            "mean": 20.8,
            "std": 6.2,
            "min": 9.4,
            "max": 40.0
        },
        # ... all numeric columns
    },
    "categorical_stats": {
        "Season": {
            "unique": 4,
            "most_common": "Summer",
            "counts": {"Summer": 420, "Spring": 390, ...}
        },
        # ... all categorical columns
    },
    "correlations": np.ndarray  # Correlation matrix
}
```

**Example:**
```python
df = read_dataset_first_available()
stats = get_dataset_statistics(df)

print(f"Dataset size: {stats['basic_info']['rows']} samples")
print(f"Temperature range: {stats['numerical_stats']['Temperature']['min']}-{stats['numerical_stats']['Temperature']['max']}°C")
```

---

## UI Rendering

### `render_crop_card()`

Generate HTML crop card with emoji and confidence visualization.

**Signature:**
```python
def render_crop_card(crop_name: str, confidence: float) -> str
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `crop_name` | `str` | Crop name (e.g., "Tomatoes") |
| `confidence` | `float` | Confidence score (0-1) |

**Returns:**
- `str` - HTML string for crop card

**Example:**
```python
html = render_crop_card("Tomatoes", 0.87)
st.markdown(html, unsafe_allow_html=True)
```

**Generated HTML:**
```html
<div class="crop-card">
    <div class="crop-emoji">🍅</div>
    <div class="crop-name">Tomatoes</div>
    <div class="crop-confidence">87.0%</div>
    <div class="confidence-bar-container">
        <div class="confidence-bar" style="width: 87.0%"></div>
    </div>
</div>
```

---

### `get_crop_emoji()`

Get emoji for crop with fuzzy matching.

**Signature:**
```python
def get_crop_emoji(crop_name: str) -> str
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `crop_name` | `str` | Crop name to lookup |

**Returns:**
- `str` - Emoji character (or "🌱" default)

**Matching Strategy:**

1. **Exact match**: `"Tomatoes"` → `"🍅"`
2. **Case-insensitive**: `"tomatoes"` → `"🍅"`
3. **Singularize**: `"Cauliflowers"` → `"🥣"` (plural → singular)
4. **ies→y transform**: `"Strawberries"` → `"🍓"`
5. **Fuzzy match**: `difflib.get_close_matches()` with 78% threshold

**Example:**
```python
print(get_crop_emoji("Tomatoes"))      # 🍅
print(get_crop_emoji("tomatoes"))      # 🍅
print(get_crop_emoji("Cauliflowers"))  # 🥣
print(get_crop_emoji("Strawberries"))  # 🍓
print(get_crop_emoji("Tomat"))         # 🍅 (fuzzy match)
print(get_crop_emoji("UnknownCrop"))   # 🌱 (default)
```

---

## Rotation Planning

### `generate_rotation_plan()`

Generate multi-year crop rotation plan using genetic algorithm.

**Signature:**
```python
def generate_rotation_plan(
    years: int, 
    crops_per_year: int, 
    soil_type: str, 
    season: str
) -> Dict[int, List[Dict]]
```

**Parameters:**

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| `years` | `int` | Number of years | 2-5 |
| `crops_per_year` | `int` | Crops per season | 1-4 |
| `soil_type` | `str` | Soil classification | Loam/Sandy/Clay |
| `season` | `str` | Primary season | Spring/Summer/Autumn/Winter |

**Returns:**
```python
{
    1: [
        {"crop": "Tomatoes", "season": "Summer", "impact": "depleting"},
        {"crop": "Lettuce", "season": "Summer", "impact": "neutral"}
    ],
    2: [
        {"crop": "Peas", "season": "Summer", "impact": "enriching"},
        {"crop": "Carrots", "season": "Summer", "impact": "neutral"}
    ],
    # ... up to specified years
}
```

**Example:**
```python
plan = generate_rotation_plan(
    years=3, 
    crops_per_year=2, 
    soil_type="Loam", 
    season="Summer"
)

for year, crops in plan.items():
    print(f"Year {year}:")
    for crop in crops:
        print(f"  - {crop['crop']} ({crop['impact']})")
```

**Output:**
```
Year 1:
  - Tomatoes (depleting)
  - Lettuce (neutral)
Year 2:
  - Peas (enriching)
  - Carrots (neutral)
Year 3:
  - Broccoli (depleting)
  - Spinach (neutral)
```

**Algorithm:**
- Uses DEAP genetic algorithm
- Optimizes for soil impact diversity
- Avoids repeating crops consecutively
- Balances depleting/enriching/neutral impacts

---

## Utility Functions

### `CANONICAL_FEATURES`

Global constant defining feature order.

**Type:** `List[str]`

**Value:**
```python
CANONICAL_FEATURES = [
    "Temperature", 
    "Rainfall", 
    "Light_Intensity", 
    "Nitrogen", 
    "Phosphorus", 
    "Potassium",
    "Season", 
    "Soil_Type", 
    "Impact", 
    "Fertility"
]
```

**Usage:**
```python
# Ensure input has all features
for feat in CANONICAL_FEATURES:
    assert feat in input_data, f"Missing feature: {feat}"
```

---

### `CROP_EMOJIS`

Emoji mapping for all supported crops.

**Type:** `Dict[str, str]`

**Value:**
```python
CROP_EMOJIS = {
    "Tomatoes": "🍅",
    "Potatoes": "🥔",
    "Lettuce": "🥬",
    "Carrots": "🥕",
    "Peppers": "🌶️",
    "Chilli Peppers": "🌶️",
    "Cucumbers": "🥒",
    "Strawberry": "🍓",
    "Watermelon": "🍉",
    "Broccoli": "🥦",
    "Cauliflowers": "🥣",
    "Cauliflower": "🥣",
    "Spinach": "🌿",
    "Kale": "🥬",
    "Arugula": "🥬",
    "Asparagus": "🌿",
    "Beet": "🌰",
    "Cabbage": "🥬",
    "Chard": "🥬",
    "Cress": "🌿",
    "Okra": "🥒",
    "Peas": "🫛",
    "Radish": "🌰",
    "Squash": "🎃",
    "Turnip": "🌰",
    "Zucchini": "🥒"
}
```

---

## Error Handling

All functions include error handling:

```python
try:
    results = get_top_3_predictions(models_dict, input_data)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    results = None
```

**Common Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Model files missing | Run `python main.py` |
| `KeyError` | Invalid feature name | Check `CANONICAL_FEATURES` |
| `ValueError` | Invalid categorical value | Use `get_valid_categorical_values()` |
| `AttributeError` | Model not loaded | Call `load_app_models()` first |

---

## Type Hints

All functions include type hints for better IDE support:

```python
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def function_example(
    param1: str,
    param2: int,
    param3: Optional[float] = None
) -> Dict[str, Any]:
    """Function with type hints"""
    return {"result": "value"}
```

---

## Performance Considerations

### Caching Decorators

| Decorator | Use Case | Invalidation |
|-----------|----------|--------------|
| `@st.cache_resource` | Model objects, connections | Manual only |
| `@st.cache_data` | DataFrames, computations | On data change |

### Memory Usage

| Object | Memory | Notes |
|--------|--------|-------|
| RandomForest model | ~5 MB | Cached once |
| StandardScaler | ~2 KB | Minimal overhead |
| Dataset (1500 rows) | ~500 KB | Cached |
| Label encoders | ~2 KB | Small dict |

### Execution Time

| Operation | Time | Cached |
|-----------|------|--------|
| Model loading | 2s | 0.01s ✓ |
| Dataset loading | 0.5s | 0.01s ✓ |
| Feature encoding | 0.01s | N/A |
| Prediction | 0.05s | N/A |
| UI render | 0.1s | N/A |

---

## Version History

### v1.1 (Current)
- Added temperature scaling for probability calibration
- Implemented form-based UI for smooth interaction
- Enhanced caching strategy for 200x performance improvement
- Added 22 unique crop emojis with fuzzy matching

### v1.0
- Initial release with RandomForest classifier
- Basic Streamlit UI
- Dataset analytics features

---

**For more information:**
- See `DOCUMENTATION.md` for technical details
- See `README.md` for quick start guide
- Report issues on GitHub

*Last Updated: November 9, 2025*
