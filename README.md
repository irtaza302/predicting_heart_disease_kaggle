# Heart Disease Prediction - Kaggle Playground Series S6E2

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/playground-series-s6e2)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Overview

This project predicts the likelihood of heart disease using machine learning models as part of the Kaggle Playground Series S6E2 competition. The evaluation metric is **ROC-AUC**.

**Target**: Achieve ROC-AUC > 0.954

## 🎯 Results

| Model | CV AUC Score |
|-------|-------------|
| XGBoost | ~0.95+ |
| LightGBM | ~0.95+ |
| CatBoost | ~0.95+ |
| Random Forest | ~0.94+ |
| Gradient Boosting | ~0.94+ |
| **Stacking Ensemble** | **Best Performance** |

## 📁 Project Structure

```
predicting_heart_disease_kaggle/
├── heart_disease_kaggle.ipynb    # Main Jupyter notebook with advanced techniques
├── heart_disease_prediction.py   # Python script for model training
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore file
└── playground-series-s6e2/       # Data directory (not tracked in git)
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 🔧 Features

### Feature Engineering
- **Age Features**: Age groups, squared, cubed, and square root transformations
- **Blood Pressure Categories**: Binned BP values for risk stratification
- **Cholesterol Categories**: Binned cholesterol levels
- **Risk Score**: Composite risk factor combining multiple indicators
- **Heart Rate Reserve**: Calculated as 220 - Age - Max HR
- **Interaction Features**: Age×BP, Age×Cholesterol, BP×Cholesterol
- **ST Depression Ratio**: ST depression normalized by heart rate

### Models Used
1. **XGBoost** - Gradient boosting with optimized hyperparameters
2. **LightGBM** - Fast gradient boosting implementation
3. **CatBoost** - Gradient boosting with categorical feature support
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Sklearn's gradient boosting classifier

### Ensemble Techniques
- **Weighted Average Ensemble**: Weights based on cross-validation scores
- **Simple Average Ensemble**: Equal weight averaging
- **Stacking Ensemble**: Meta-learner (Logistic Regression) on top of base models

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna
```

### Installation
```bash
git clone https://github.com/irtaza302/predicting_heart_disease_kaggle.git
cd predicting_heart_disease_kaggle
```

### Download Data
Download the competition data from [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e2/data) and place it in the `playground-series-s6e2/` directory.

### Run Training
```bash
# Using Python script
python heart_disease_prediction.py

# Or open the Jupyter notebook
jupyter notebook heart_disease_kaggle.ipynb
```

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Sex | Gender (0: Female, 1: Male) |
| Chest pain type | Type of chest pain (1-4) |
| BP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar > 120 mg/dl |
| ECG results | Resting electrocardiographic results |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina (0: No, 1: Yes) |
| ST depression | ST depression induced by exercise |
| Slope of ST | Slope of the peak exercise ST segment |
| Number of vessels fluro | Number of major vessels colored by fluoroscopy |
| Thallium | Thallium stress test result |

## 📈 Model Performance

The stacking ensemble approach provides the most robust predictions by combining the strengths of multiple base models:

1. **Base Models**: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting
2. **Meta-Learner**: Logistic Regression
3. **Cross-Validation**: 5-fold stratified CV for robust evaluation

## 📝 Submission Files

The script generates multiple submission files:
- `submission_best_model.csv` - Best single model predictions
- `submission_weighted_ensemble.csv` - Weighted average ensemble
- `submission_simple_avg.csv` - Simple average ensemble
- `submission_stacking.csv` - Stacking ensemble (recommended)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **XGBoost** - Gradient boosting
- **LightGBM** - Light gradient boosting
- **CatBoost** - Categorical boosting
- **Optuna** - Hyperparameter optimization
- **Matplotlib & Seaborn** - Visualization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for hosting the competition
- Open-source ML community for the amazing tools and libraries

---

⭐ If you find this project helpful, please give it a star!
