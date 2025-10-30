"""
XGBoost Training Pipeline for Volleyball Match Prediction
Train and evaluate XGBoost model to predict match winners
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class VolleyballPredictor:
    """Train XGBoost model to predict volleyball match outcomes"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def load_data(self, features_file: str = 'X_features.csv', target_file: str = 'y_target.csv'):
        """Load features and target from CSV files"""
        X = pd.read_csv(features_file)
        y = pd.read_csv(target_file).squeeze()
        
        # Handle any NaN values
        X = X.fillna(0)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, params=None):
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            print("\n⚠️  XGBoost not installed. Installing required packages...")
            print("Please run: pip install xgboost scikit-learn pandas numpy joblib")
            return None
        
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
        
        print(f"\nTraining XGBoost model with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        
        print("\n✓ Model training complete")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            print("Error: Model not trained yet")
            return
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Team B Wins', 'Team A Wins']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                Predicted")
        print(f"                B Wins  A Wins")
        print(f"Actual B Wins   {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"Actual A Wins   {cm[1][0]:6d}  {cm[1][1]:6d}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self, top_n=20):
        """Get and display feature importance"""
        if self.model is None:
            print("Error: Model not trained yet")
            return
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print(f"{'='*60}")
        for i, row in feature_importance.head(top_n).iterrows():
            print(f"{row['feature']:40s} {row['importance']:.4f}")
        
        return feature_importance
    
    def save_model(self, model_path='volleyball_predictor.pkl'):
        """Save trained model"""
        if self.model is None:
            print("Error: No model to save")
            return
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"\n✓ Model saved to: {model_path}")
    
    def load_model(self, model_path='volleyball_predictor.pkl'):
        """Load trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        print(f"✓ Model loaded from: {model_path}")
    
    def predict_match(self, team_a_features: dict, team_b_features: dict):
        """Predict outcome of a new match"""
        if self.model is None:
            print("Error: Model not trained yet")
            return
        
        # Create feature vector (simplified example)
        # In practice, you'd need to format the features according to your feature engineering
        features = pd.DataFrame([team_a_features])
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        winner = "Team A" if prediction == 1 else "Team B"
        confidence = probability[int(prediction)] * 100
        
        print(f"\nPrediction: {winner} wins")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities: Team B: {probability[0]*100:.2f}%, Team A: {probability[1]*100:.2f}%")
        
        return {
            'winner': winner,
            'confidence': confidence,
            'probabilities': {
                'team_a': probability[1],
                'team_b': probability[0]
            }
        }


def main():
    """Main training pipeline"""
    print("="*60)
    print("VOLLEYBALL MATCH PREDICTION - XGBOOST TRAINING")
    print("="*60)
    
    # Initialize predictor
    predictor = VolleyballPredictor()
    
    # Load data
    try:
        X, y = predictor.load_data()
        print(f"\n✓ Data loaded: {len(X)} samples, {len(X.columns)} features")
    except FileNotFoundError:
        print("\nError: Feature files not found.")
        print("Please run feature_engineering.py first to create X_features.csv and y_target.csv")
        return
    
    # Prepare train/test split
    print(f"\n{'='*60}")
    print("PREPARING DATA")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    
    # Train model
    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    model = predictor.train_model(X_train, y_train)
    
    if model is None:
        return
    
    # Evaluate model
    results = predictor.evaluate_model(X_test, y_test)
    
    # Feature importance
    feature_importance = predictor.get_feature_importance(top_n=20)
    
    # Save model
    predictor.save_model('volleyball_predictor.pkl')
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("✓ Feature importance saved to: feature_importance.csv")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("\nYour model is ready to predict volleyball match outcomes!")
    print("\nNext steps:")
    print("1. Add more match data (XML files) to improve accuracy")
    print("2. Use volleyball_predictor.pkl to make predictions")
    print("3. Fine-tune hyperparameters for better performance")


if __name__ == '__main__':
    main()
