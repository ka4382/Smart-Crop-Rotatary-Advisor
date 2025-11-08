# 🌾 SmartCrop Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model (First Time Only)
```bash
python main.py
```
*Wait ~5 seconds for training to complete*

### 3. Launch App
```bash
streamlit run app.py
```
*Opens at http://localhost:8501*

---

## 🎯 Making Predictions

### Input Parameters

| Parameter | Range | Unit | Example |
|-----------|-------|------|---------|
| **Temperature** | 9.4 - 40.0 | °C | 20.8 |
| **Rainfall** | 410 - 2510 | mm | 949 |
| **Light Intensity** | 69 - 986 | lux | 398 |
| **Nitrogen** | 40 - 410 | mg/ha | 140 |
| **Phosphorus** | 13 - 360 | mg/ha | 108 |
| **Potassium** | 35 - 580 | mg/ha | 180 |

**Categorical:**
- **Season**: Spring, Summer, Autumn, Winter
- **Soil Type**: Loam, Sandy, Clay  
- **Fertility**: Low, Moderate, High
- **Impact**: depleting, neutral, restorative, enriching

### Best Practices

✅ **DO:**
- Use values close to the means for best confidence
- Select categorical values from provided options
- Adjust sliders freely before clicking predict

❌ **DON'T:**
- Enter values outside the valid ranges
- Mix incompatible soil types and seasons
- Expect high confidence for extreme values

---

## 📊 Understanding Results

### Confidence Scores

| Score | Meaning |
|-------|---------|
| **70-95%** | ✅ High confidence - Strong recommendation |
| **50-70%** | ⚠️ Moderate - Consider alternatives |
| **<50%** | ❌ Low - Input may be unusual |

### Top 3 Crops
- **Rank 1**: Best match for your conditions
- **Rank 2**: Good alternative
- **Rank 3**: Backup option

---

## 🔄 Rotation Planning

### Steps:
1. Go to **Crop Rotation Plan** tab
2. Set:
   - **Years**: 2-5 (how many years to plan)
   - **Crops per year**: 1-4 (how many crops per season)
   - **Soil Type**: Your primary soil
   - **Season**: Primary growing season
3. Click **Generate Rotation**
4. Download plan as JSON

### Rotation Benefits:
- 🌱 Maintains soil health
- 🐛 Reduces pest buildup
- 💪 Balances nutrient depletion/enrichment
- 📈 Optimizes long-term yields

---

## 🛠️ Troubleshooting

### Low Confidence (<50%)

**Solution:**
```
Try these typical values:
- Temperature: 20°C
- Rainfall: 950mm
- Light Intensity: 400
- N: 140, P: 108, K: 180
- Season: Summer
- Soil: Loam
- Fertility: Moderate
- Impact: neutral
```

### "Model Not Found" Error

**Solution:**
```bash
# Train the model first
python main.py

# Then run app
streamlit run app.py
```

### App is Slow/Laggy

**Current version has optimizations:**
- ✅ Cached model loading
- ✅ Form-based inputs (no lag on sliders)
- ✅ Instant predictions

**If still slow:**
```bash
# Update Streamlit
pip install --upgrade streamlit

# Restart app
streamlit run app.py
```

---

## 📦 Project Files

### Keep These:
- ✅ `app.py` - Main application
- ✅ `main.py` - Training script
- ✅ `soil.impact.csv` - Dataset
- ✅ `requirements.txt` - Dependencies
- ✅ `saved_models/` - Trained models

### Can Delete:
- ❌ `__pycache__/` - Python cache
- ❌ `test_*.py` - Test files
- ❌ Old custom_*.py files (if present)

---

## 🚀 Advanced Tips

### Custom Dataset

Replace `soil.impact.csv` with your data:
```csv
Name,Temperature,Rainfall,Light_Intensity,Nitrogen,Phosphorus,Potassium,Season,Soil_Type,Impact,Fertility
Tomatoes,25.0,800,450,150,120,200,Summer,Loam,depleting,Moderate
...
```

Then retrain:
```bash
python main.py
```

### Adjust Confidence

In `app.py`, line ~415:
```python
temperature = 0.6  # Lower = more confident (try 0.4)
                  # Higher = less confident (try 1.0)
```

### Add New Crops

1. Add samples to `soil.impact.csv` (50+ per crop)
2. Add emoji in `app.py`:
```python
CROP_EMOJIS = {
    "NewCrop": "🌾",  # Add here
    # ...
}
```
3. Retrain: `python main.py`

---

## 📞 Support

- 📖 **Full Docs**: See `DOCUMENTATION.md`
- 🔧 **API Reference**: See `API_REFERENCE.md`  
- 🐛 **Issues**: GitHub Issues page
- 📧 **Email**: support@example.com

---

## 🎓 Learning Resources

### Understand the ML Model
- **Algorithm**: RandomForest (ensemble of 200 decision trees)
- **Accuracy**: 99.02% on test set
- **Features**: 10 (6 numeric + 4 categorical)
- **Classes**: 22 crops

### Key Concepts
- **Feature Scaling**: StandardScaler normalizes all inputs
- **Temperature Scaling**: Calibrates probability for better confidence
- **Label Encoding**: Converts categories to numbers (0, 1, 2...)
- **Genetic Algorithm**: Optimizes rotation plans

---

## ✅ Checklist

**Before First Use:**
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model trained (`python main.py`)
- [ ] `saved_models/` folder exists with 5 .pkl files

**For Best Results:**
- [ ] Use realistic input values (close to means)
- [ ] Select valid categorical options
- [ ] Interpret confidence scores correctly
- [ ] Consider top 3 crops, not just #1

---

**Ready to grow! 🌱**

*For detailed information, see README.md and DOCUMENTATION.md*
