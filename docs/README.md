# Volleyball AI Match Prediction System

A complete machine learning pipeline to predict volleyball match outcomes using XGBoost, trained on PVL (Premier Volleyball League) match data.

## 📋 Project Overview

This project processes volleyball match XML files, extracts relevant statistics, engineers features, and trains an XGBoost model to predict match winners.

## 🗂️ Project Structure

```
VolleyballAIProject/
├── parse_volleyball_data.py    # XML parser for match data
├── feature_engineering.py      # Feature extraction & engineering
├── database_manager.py         # SQLite database management
├── batch_processor.py          # Complete processing pipeline
├── train_xgboost.py           # XGBoost model training
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Match Data

Place your PVL XML match files in the project directory. You can download them like this:

```bash
curl -o PVL2023A-W01-AKAvCMF-XML.xml 'https://dashboard.pvl.ph/assets/match_results/xml/PVL2023A-W01-AKAvCMF-XML.xml'
```

### 3. Process All Match Data

Run the complete pipeline to parse, clean, and prepare your data:

```bash
python batch_processor.py
```

This will:
- Parse all XML files
- Create `volleyball_matches.json`
- Create `volleyball_data.db` (SQLite database)
- Generate `volleyball_features.csv`
- Create `X_features.csv` and `y_target.csv` for ML training

### 4. Train the Model

```bash
python train_xgboost.py
```

This will:
- Train an XGBoost classifier
- Evaluate model performance
- Show feature importance
- Save the model to `volleyball_predictor.pkl`

## 📊 Data Processing Pipeline

### Step 1: Parse XML Files
```python
from parse_volleyball_data import VolleyballDataParser

parser = VolleyballDataParser()
parsed_data = parser.parse_multiple_files(['match1.xml', 'match2.xml'])
parser.save_to_json('volleyball_matches.json')
```

### Step 2: Store in Database
```python
from database_manager import VolleyballDatabase

db = VolleyballDatabase('volleyball_data.db')
db.connect()
db.create_schema()
db.load_from_json('volleyball_matches.json')
```

### Step 3: Feature Engineering
```python
from feature_engineering import VolleyballFeatureEngineer

engineer = VolleyballFeatureEngineer(matches_data)
X, y, metadata, feature_cols = engineer.get_feature_importance_ready_data()
```

### Step 4: Train Model
```python
from train_xgboost import VolleyballPredictor

predictor = VolleyballPredictor()
X, y = predictor.load_data()
X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
predictor.save_model('volleyball_predictor.pkl')
```

## 📈 Features Extracted

The system extracts and engineers numerous features including:

### Match Statistics
- Attack points, faults, and continues
- Block points and faults
- Serve points and faults
- Reception excellence and faults
- Dig statistics
- Set statistics
- Opponent errors

### Derived Metrics
- Attack efficiency rate
- Serve efficiency rate
- Reception efficiency rate
- Error rates for each skill
- Points per set averages

### Historical Features
- Previous match win rate
- Previous set win rate
- Average points scored/conceded
- Team performance trends

## 🎯 Model Performance

After training, the model will display:
- **Accuracy**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: True/false positives and negatives
- **Feature Importance**: Most influential features for predictions

## 📁 Output Files

| File | Description |
|------|-------------|
| `volleyball_matches.json` | Raw parsed match data in JSON format |
| `volleyball_data.db` | SQLite database with structured data |
| `volleyball_features.csv` | Complete feature dataset with metadata |
| `X_features.csv` | Feature matrix for ML (without metadata) |
| `y_target.csv` | Target labels (0 = Team B wins, 1 = Team A wins) |
| `volleyball_predictor.pkl` | Trained XGBoost model |
| `feature_importance.csv` | Feature importance rankings |

## 🔮 Making Predictions

```python
from train_xgboost import VolleyballPredictor

# Load trained model
predictor = VolleyballPredictor()
predictor.load_model('volleyball_predictor.pkl')

# Make prediction (you'll need to format features properly)
result = predictor.predict_match(team_a_features, team_b_features)
print(f"Winner: {result['winner']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## 📥 Downloading More Match Data

To improve model accuracy, add more match XML files:

```bash
# Example: Download multiple matches
curl -o match1.xml 'https://dashboard.pvl.ph/assets/match_results/xml/MATCH_ID_1.xml'
curl -o match2.xml 'https://dashboard.pvl.ph/assets/match_results/xml/MATCH_ID_2.xml'
curl -o match3.xml 'https://dashboard.pvl.ph/assets/match_results/xml/MATCH_ID_3.xml'

# Then rerun the batch processor
python batch_processor.py
```

## 🛠️ Customization

### Adjust XGBoost Parameters

Edit `train_xgboost.py` and modify the `params` dictionary:

```python
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,              # Tree depth
    'learning_rate': 0.1,        # Learning rate
    'n_estimators': 100,         # Number of trees
    'subsample': 0.8,            # Sample ratio
    'colsample_bytree': 0.8,     # Feature ratio
}
```

### Add Custom Features

Edit `feature_engineering.py` to add your own feature calculations in the `_extract_stat_features()` method.

## 📊 Database Schema

The SQLite database includes tables for:
- **tournaments**: Tournament information
- **teams**: Team details and coaches
- **players**: Player roster
- **matches**: Match metadata and results
- **team_match_stats**: Detailed team statistics per match
- **set_scores**: Individual set scores

## 🤝 Contributing

To extend this project:
1. Add more XML files to increase dataset size
2. Engineer new features based on domain knowledge
3. Experiment with different ML models
4. Add player-level features
5. Implement time-series features for recent form

## 📝 Notes

- The model uses historical data up to each match (no data leakage)
- Features are normalized and handle missing values
- More data = better predictions (aim for 50+ matches minimum)
- Team form and momentum are captured through historical features

## 🎓 Next Steps

1. **Collect More Data**: Add more XML match files
2. **Feature Tuning**: Experiment with different feature combinations
3. **Hyperparameter Optimization**: Use GridSearchCV or RandomizedSearchCV
4. **Model Ensemble**: Combine XGBoost with other models
5. **Deploy**: Create a web API to serve predictions
6. **Real-time Updates**: Add live match data integration

## 📧 Support

For issues or questions about the PVL data format or XML structure, refer to the official PVL dashboard.

---

**Good luck with your volleyball match predictions! 🏐**
