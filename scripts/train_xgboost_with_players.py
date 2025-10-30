"""
Train XGBoost model with player-level features for volleyball match prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("="*80)
print(" "*15 + "XGBOOST TRAINING WITH PLAYER-LEVEL FEATURES")
print("="*80)

# Load data
print("\n✓ Data loaded: ", end="")
X = pd.read_csv('X_features_with_players.csv')
y = pd.read_csv('y_target_with_players.csv').values.ravel()
print(f"{len(X)} samples, {len(X.columns)} features")

print("\nFeature groups:")
print(f"  Team historical stats: {len([c for c in X.columns if 'win_rate' in c or 'avg' in c and 'starter' not in c and 'libero' not in c and 'scorer' not in c and 'roster' not in c and 'sets_per' not in c and '10plus' not in c])}")
print(f"  Player-based stats: {len([c for c in X.columns if 'starter' in c or 'libero' in c or 'scorer' in c or 'roster' in c or 'sets_per' in c or '10plus' in c])}")

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
print("\nTraining XGBoost classifier with player features...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

print("✓ Model training complete")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluate
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Team B Wins', 'Team A Wins']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predicted")
print(f"                B Wins  A Wins")
print(f"Actual B Wins     {cm[0][0]:3d}     {cm[0][1]:3d}")
print(f"Actual A Wins     {cm[1][0]:3d}     {cm[1][1]:3d}")

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Top 15)")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nMost Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.4f}")

# Save model
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_data = {
    'model': model,
    'feature_names': X.columns.tolist()
}

joblib.dump(model_data, 'volleyball_predictor_with_players.pkl')
print("✓ Model saved to: volleyball_predictor_with_players.pkl")

# Comparison with team-only model
print("\n" + "="*80)
print("COMPARISON: TEAM-ONLY VS PLAYER-ENHANCED MODEL")
print("="*80)

try:
    team_only_model = joblib.load('volleyball_predictor.pkl')
    print("\n📊 Performance Comparison:")
    print(f"  Team-only model (Naive Bayes):     75.25%")
    print(f"  XGBoost with player features:      {accuracy*100:.2f}%")
    
    improvement = (accuracy * 100) - 75.25
    if improvement > 0:
        print(f"\n✨ Improvement: +{improvement:.2f}%")
    else:
        print(f"\n⚠️  Change: {improvement:.2f}%")
except:
    print("\n(No previous model found for comparison)")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nYour enhanced XGBoost model with player features is ready!")
print("\nNext steps:")
print("1. Use volleyball_predictor_with_players.pkl for predictions")
print("2. Run simulate_tournament.py to predict tournament outcomes")
print("3. Compare with other algorithms")
