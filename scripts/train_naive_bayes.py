"""
Train Naive Bayes model for volleyball match prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("="*80)
print(" "*20 + "VOLLEYBALL MATCH PREDICTION - NAIVE BAYES TRAINING")
print("="*80)

# Load data
print("\n✓ Data loaded: ", end="")
X = pd.read_csv('X_features.csv')
y = pd.read_csv('y_target.csv').values.ravel()
print(f"{len(X)} samples, {len(X.columns)} features")

# Split data
print("\n" + "="*80)
print("PREPARING DATA")
print("="*80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Class distribution - Train: {{1: {sum(y_train)}, 0: {len(y_train) - sum(y_train)}}}")
print(f"Class distribution - Test: {{1: {sum(y_test)}, 0: {len(y_test) - sum(y_test)}}}")

# Train model
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)
print("\nTraining Gaussian Naive Bayes model...")

model = GaussianNB()
model.fit(X_train, y_train)

print("✓ Model training complete")

# Evaluate
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Team B Wins', 'Team A Wins']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predicted")
print(f"                B Wins  A Wins")
print(f"Actual B Wins     {cm[0][0]:3d}     {cm[0][1]:3d}")
print(f"Actual A Wins     {cm[1][0]:3d}     {cm[1][1]:3d}")

# Feature probabilities (for Naive Bayes interpretation)
print("\n" + "="*80)
print("MODEL INSIGHTS")
print("="*80)

print("\nNaive Bayes assumes feature independence and uses probability distributions.")
print(f"Number of classes: {len(model.classes_)}")
print(f"Class priors: Team B Win = {model.class_prior_[0]:.4f}, Team A Win = {model.class_prior_[1]:.4f}")

# Save model
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_data = {
    'model': model,
    'feature_names': X.columns.tolist()
}

joblib.dump(model_data, 'volleyball_predictor.pkl')
print("✓ Model saved to: volleyball_predictor.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nYour Naive Bayes model is ready to predict volleyball match outcomes!")
print("\nNext steps:")
print("1. Use volleyball_predictor.pkl to make predictions")
print("2. Run simulate_tournament.py for tournament predictions")
print("3. Compare with other algorithms using compare_algorithms.py")
