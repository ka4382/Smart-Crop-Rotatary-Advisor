# 🌾 SmartCrop Rotatory Advisor

> AI-Powered Crop Recommendation and Rotation Planning System for Small-Scale Farmers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

SmartCrop Rotatory Advisor is an intelligent agricultural decision support system that helps small-scale farmers make data-driven decisions about crop selection and rotation planning. Using machine learning algorithms trained on soil composition, climate data, and historical yield patterns, the system provides personalized crop recommendations with confidence scores.

### Key Benefits

- ✅ **Optimized Crop Selection** - Get top 3 crop recommendations based on your soil and climate conditions
- 📊 **High Accuracy** - 99%+ prediction accuracy using RandomForest classifier
- 🔄 **Rotation Planning** - Generate multi-year crop rotation plans to maintain soil health
- ⚡ **Real-time Results** - Instant predictions with beautiful, intuitive UI
- 🌱 **22 Crop Support** - Covers major vegetables and crops suitable for small farms

## ✨ Features

### 1. Crop Recommendation Engine
- Input 10 environmental and soil parameters
- Get top 3 crop suggestions with confidence scores
- Visual confidence indicators and emoji-based crop identification
- Temperature-scaled probability for better prediction distribution

### 2. Crop Rotation Planner
- Generate 2-5 year rotation plans
- Specify crops per season
- Optimized for soil fertility and pest management
- Export plans in JSON format

### 3. Dataset Analytics
- Comprehensive dataset statistics
- Feature distribution visualizations
- Correlation analysis
- Download processed data

### 4. Model Management
- Cached model loading for fast performance
- Automatic retraining capability
- Model metadata display (accuracy, classes, features)

## 🛠 Tech Stack

### Core Technologies
- **Python 3.8+** - Programming language
- **Streamlit** - Web application framework
- **scikit-learn 0.24.2+** - Machine learning library
- **pandas 1.3+** - Data manipulation
- **numpy 1.21+** - Numerical computing

### ML Pipeline
- **RandomForest Classifier** - Primary prediction model (99.02% accuracy)
- **StandardScaler** - Feature normalization
- **LabelEncoder** - Categorical feature encoding
- **DEAP** - Genetic algorithms for rotation optimization

### Visualization
- **matplotlib 3.4.2+** - Static plots
- **seaborn 0.11.1+** - Statistical visualizations
- **Custom CSS** - Modern, responsive UI design

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/SmartCropRotatoryAdvisor.git
cd SmartCropRotatoryAdvisor/code
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import streamlit; import sklearn; import pandas; print('✓ All dependencies installed successfully!')"
```

## 🚀 Usage

### Quick Start

1. **Train the Model** (First time only)
```bash
python main.py
```
This creates the `saved_models/` directory with trained model artifacts.

2. **Launch Web Application**
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

3. **Make Predictions**
   - Navigate to **Crop Recommendation** tab
   - Adjust sliders for your environmental conditions:
     - Temperature, Rainfall, Light Intensity
     - Soil nutrients (N, P, K)
     - Soil type, Season, Fertility, Impact
   - Click **"🌱 Get Crop Recommendations"**
   - View top 3 recommended crops with confidence scores

### Advanced Usage

#### Custom Training
```bash
# Train with specific configuration
python main.py --mode full  # Slower but more accurate
```

#### Dataset Requirements
Your dataset should be a CSV file with these columns:
- **Name** - Crop name (target variable)
- **Temperature** - Average temperature (°C)
- **Rainfall** - Annual rainfall (mm)
- **Light_Intensity** - Light intensity (lux)
- **Nitrogen, Phosphorus, Potassium** - Soil nutrients (mg/ha)
- **Season** - Growing season (Spring/Summer/Autumn/Winter)
- **Soil_Type** - Soil classification (Loam/Sandy/Clay)
- **Impact** - Crop impact on soil (depleting/enriching/neutral)
- **Fertility** - Soil fertility level (Low/Moderate/High)

## 📁 Project Structure

```
SmartCropRotatoryAdvisor/
├── code/
│   ├── app.py                    # Main Streamlit application ⭐
│   ├── main.py                   # Model training script
│   ├── requirements.txt          # Python dependencies
│   ├── soil.impact.csv          # Training dataset (1500+ samples)
│   ├── crop_data_clean.csv      # Alternative dataset
│   └── saved_models/            # Trained model artifacts
│       ├── random_forest.pkl    # Trained RandomForest model
│       ├── scaler.pkl           # StandardScaler object
│       ├── label_encoder.pkl    # Target encoder
│       ├── label_encoders.pkl   # Feature encoders (dict)
│       └── metadata.pkl         # Model metadata
├── README.md                     # This file
├── DOCUMENTATION.md              # Detailed documentation
└── API_REFERENCE.md             # API and code reference

```

### Key Files Explained

| File | Purpose | Size |
|------|---------|------|
| `app.py` | Main Streamlit web application with UI and prediction logic | ~700 lines |
| `main.py` | Model training pipeline with RandomForest configuration | ~200 lines |
| `soil.impact.csv` | Primary training dataset with 22 crop classes | ~500 KB |
| `requirements.txt` | Python package dependencies | ~1 KB |

## 📊 Model Performance

### Training Results

```
Model: RandomForest Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:        99.02%
Estimators:      200 trees
Max Depth:       None (full growth)
Features:        10 (6 numeric + 4 categorical)
Classes:         22 crops
Training Time:   ~3-5 seconds
```

### Supported Crops (22)

1. Arugula
2. Asparagus
3. Beet
4. Broccoli
5. Cabbage
6. Cauliflowers
7. Chard
8. Chilli Peppers
9. Cress
10. Cucumbers
11. Kale
12. Lettuce
13. Okra
14. Peas
15. Radish
16. Spinach
17. Squash
18. Strawberry
19. Tomatoes
20. Turnip
21. Watermelon
22. Zucchini

### Feature Importance

| Feature | Importance | Description |
|---------|------------|-------------|
| Temperature | High | Average growing temperature (°C) |
| Rainfall | High | Annual precipitation (mm) |
| Light_Intensity | High | Light exposure (lux) |
| Nitrogen | Medium | Soil nitrogen content (mg/ha) |
| Season | Medium | Growing season category |
| Soil_Type | Medium | Soil classification |

### Confidence Calibration

The model uses **temperature scaling** (T=0.6) to calibrate probabilities:
- Raw model probabilities are sharpened using softmax with reduced temperature
- Typical confidence ranges: 70-95% for in-distribution inputs
- Below 50% indicates input values far from training distribution

## 🔧 API Documentation

### Core Functions

#### `load_app_models()`
```python
@st.cache_resource
def load_app_models() -> Dict[str, Any]
```
Loads and caches trained model artifacts.

**Returns:**
- `dict` containing model, scaler, encoders, metadata

**Cache:** Resource cache (persistent across reruns)

#### `get_top_3_predictions(models_dict, input_data)`
```python
def get_top_3_predictions(models_dict: Dict, input_data: Dict) -> List[Dict]
```
Generate top 3 crop predictions with confidence scores.

**Parameters:**
- `models_dict` - Dictionary from `load_app_models()`
- `input_data` - Dictionary of input features

**Returns:**
```python
[
    {"rank": 1, "crop": "Tomatoes", "score": 0.87, "raw_score": 0.65},
    {"rank": 2, "crop": "Peppers", "score": 0.09, "raw_score": 0.12},
    {"rank": 3, "crop": "Cucumbers", "score": 0.04, "raw_score": 0.08}
]
```

#### `build_feature_vector(input_data, models_dict)`
```python
def build_feature_vector(input_data: Dict, models_dict: Dict) -> np.ndarray
```
Constructs and scales feature vector in correct order.

**Process:**
1. Extract features in model's expected order
2. Encode categorical features using LabelEncoders
3. Apply StandardScaler transformation
4. Return 2D numpy array ready for prediction

### Performance Optimizations

All expensive operations are cached:
```python
@st.cache_data          # For data operations
@st.cache_resource      # For model objects
```

**Cached Functions:**
- `load_app_models()` - Model loading
- `read_dataset_first_available()` - Dataset reading
- `compute_data_ranges()` - Feature range calculation
- `get_valid_categorical_values()` - Categorical options
- `get_dataset_statistics()` - Statistical analysis

## 🎨 UI Features

### Modern Design Elements
- Gradient backgrounds with glassmorphism
- Responsive crop cards with confidence bars
- Inline progress visualizations
- Color-coded confidence indicators
- Emoji-based crop identification (22 unique emojis)

### Form-Based Inputs
All inputs are wrapped in `st.form()` to prevent lag:
- Sliders update instantly without rerunning
- Predictions execute only on button click
- Smooth, responsive user experience

### Accessibility
- Clear visual hierarchy
- High-contrast text
- Intuitive slider controls
- Helpful tooltips and hints

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/SmartCropRotatoryAdvisor.git
cd SmartCropRotatoryAdvisor/code

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
streamlit run app.py

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where possible
- Test changes thoroughly before committing

### Areas for Contribution
- 🌾 Add support for more crops
- 🗺️ Regional crop variety datasets
- 📱 Mobile-responsive UI improvements
- 🌐 Internationalization (i18n)
- 📊 Advanced visualization features
- 🧪 Unit tests and integration tests

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 SmartCrop Rotatory Advisor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/SmartCropRotatoryAdvisor/issues)
- **Email**: support@smartcrop.example.com
- **Documentation**: See `DOCUMENTATION.md` for detailed guides

## 🙏 Acknowledgments

- Dataset sources: Agricultural research institutions
- ML frameworks: scikit-learn, pandas, numpy
- UI framework: Streamlit
- Community: Open source contributors

## 📈 Roadmap

### Version 2.0 (Planned)
- [ ] Weather API integration for real-time data
- [ ] GPS-based location recommendations
- [ ] Historical yield tracking
- [ ] Pest and disease prediction
- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support

### Version 1.1 (Current)
- [x] Temperature-scaled probability calibration
- [x] Form-based UI for smooth interaction
- [x] Comprehensive caching for performance
- [x] 22 unique crop emojis
- [x] Download predictions as JSON

---

**Built with ❤️ for farmers worldwide**

*Last Updated: November 9, 2025*
