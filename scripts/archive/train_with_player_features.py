"""
Advanced Training with Player-Level Statistics
Combines enhanced features + player stats for maximum performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')


def merge_enhanced_and_player_features():
    """Combine enhanced features with player-level features"""
    print("="*80)
    print("MERGING ENHANCED FEATURES WITH PLAYER STATISTICS")
    print("="*80)
    
    # Load all feature sets
    print("\nLoading feature sets...")
    
    # Load basic features
    X_basic = pd.read_csv('X_features.csv')
    print(f"✓ Basic features: {len(X_basic.columns)} columns")
    
    # Load enhanced features (momentum, form, etc.)
    try:
        X_enhanced = pd.read_csv('X_features_enhanced.csv')
        print(f"✓ Enhanced features: {len(X_enhanced.columns)} columns")
        has_enhanced = True
    except FileNotFoundError:
        print("⚠️  Enhanced features not found, using basic only")
        X_enhanced = X_basic
        has_enhanced = False
    
    # Load player features
    try:
        X_players = pd.read_csv('X_features_with_players.csv')
        print(f"✓ Player features: {len(X_players.columns)} columns")
        has_players = True
    except FileNotFoundError:
        print("⚠️  Player features not found")
        X_players = X_basic
        has_players = False
    
    if not has_players:
        print("\n❌ Error: Player features are required for this script")
        print("Please ensure X_features_with_players.csv exists")
        return None, None
    
    # Identify unique columns from each set
    basic_cols = set(X_basic.columns)
    enhanced_cols = set(X_enhanced.columns) - basic_cols
    player_cols = set(X_players.columns) - basic_cols
    
    print(f"\nFeature breakdown:")
    print(f"  - Basic features: {len(basic_cols)}")
    print(f"  - Enhanced features (new): {len(enhanced_cols)}")
    print(f"  - Player features (new): {len(player_cols)}")
    
    # Combine all features
    if has_enhanced:
        # Start with enhanced (includes basic + momentum/form)
        X_combined = X_enhanced.copy()
        
        # Add player columns that aren't already there
        for col in player_cols:
            if col in X_players.columns and col not in X_combined.columns:
                X_combined[col] = X_players[col]
    else:
        X_combined = X_players.copy()
    
    # Load target
    y = pd.read_csv('y_target.csv').values.ravel()
    
    # Remove any duplicate columns
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
    
    # Handle missing values
    X_combined = X_combined.fillna(0)
    
    # Remove low variance features
    variance = X_combined.var()
    low_var_cols = variance[variance < 0.001].index
    if len(low_var_cols) > 0:
        print(f"\nRemoving {len(low_var_cols)} near-constant features")
        X_combined = X_combined.drop(columns=low_var_cols)
    
    print(f"\n✓ Combined dataset: {len(X_combined.columns)} total features")
    print(f"✓ Samples: {len(X_combined)}")
    
    return X_combined, y


def train_with_all_features():
    """Train models with all available features"""
    print("\n" + "="*80)
    print("TRAINING WITH COMPLETE FEATURE SET")
    print("(Basic + Enhanced + Player Statistics)")
    print("="*80)
    
    X, y = merge_enhanced_and_player_features()
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(X.columns)}")
    
    # Test models
    print("\n" + "="*80)
    print("TESTING ALGORITHMS WITH ALL FEATURES")
    print("="*80)
    
    models = {
        'XGBoost': XGBClassifier(
            max_depth=7,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'XGBoost (Deep)': XGBClassifier(
            max_depth=9,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=2,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            num_leaves=70,
            max_depth=8,
            learning_rate=0.03,
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=250,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{name}...", end=' ')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        result = {
            'name': name,
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        results.append(result)
        trained_models[name] = model
        
        print(f"Acc: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.3f}")
    
    # Create stacking ensemble
    print("\nStacking Ensemble...", end=' ')
    stacking = StackingClassifier(
        estimators=[
            ('xgb', trained_models['XGBoost']),
            ('lgbm', trained_models['LightGBM']),
            ('cat', trained_models['CatBoost'])
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)
    y_proba_stack = stacking.predict_proba(X_test)[:, 1]
    
    accuracy_stack = accuracy_score(y_test, y_pred_stack)
    f1_stack = f1_score(y_test, y_pred_stack)
    auc_stack = roc_auc_score(y_test, y_proba_stack)
    
    results.append({
        'name': 'Stacking Ensemble',
        'model': stacking,
        'accuracy': accuracy_stack,
        'f1_score': f1_stack,
        'auc': auc_stack,
        'cv_mean': 0,
        'cv_std': 0
    })
    
    print(f"Acc: {accuracy_stack:.4f} | F1: {f1_stack:.4f} | AUC: {auc_stack:.4f}")
    
    # Results
    print("\n" + "="*80)
    print("FINAL RESULTS WITH PLAYER STATISTICS")
    print("="*80)
    
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'F1':<10} {'AUC':<10} {'CV'}")
    print("-"*80)
    
    for i, r in enumerate(results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        cv_str = f"{r['cv_mean']:.3f}±{r['cv_std']:.3f}" if r['cv_mean'] > 0 else "N/A"
        print(f"{medal} {i:<3} {r['name']:<25} "
              f"{r['accuracy']:.4f} ({r['accuracy']*100:5.1f}%)  "
              f"{r['f1_score']:.4f}  "
              f"{r['auc']:.4f}  "
              f"{cv_str}")
    
    # Best model analysis
    best = results[0]
    best_model = best['model']
    
    print(f"\n" + "="*80)
    print(f"🏆 CHAMPION: {best['name']}")
    print("="*80)
    print(f"   Accuracy:  {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"   F1-Score:  {best['f1_score']:.4f}")
    print(f"   AUC-ROC:   {best['auc']:.4f}")
    
    y_pred_best = best_model.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Team B Wins', 'Team A Wins']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"                Predicted")
    print(f"                B Wins  A Wins")
    print(f"Actual B Wins   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Actual A Wins   {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n{'='*80}")
        print(f"TOP 20 MOST IMPORTANT FEATURES (WITH PLAYER STATS)")
        print(f"{'='*80}")
        
        for idx, row in importance.head(20).iterrows():
            bar = '█' * int(row['importance'] * 50)
            # Mark player features
            marker = "👤" if any(x in row['feature'].lower() for x in ['starter', 'libero', 'top_scorer', 'roster']) else "  "
            print(f"{marker} {row['feature']:45s} {row['importance']:.6f} {bar}")
        
        importance.to_csv('feature_importance_with_players.csv', index=False)
        print(f"\n✓ Feature importance saved: feature_importance_with_players.csv")
    
    # Save best model
    model_data = {
        'model': best_model,
        'feature_names': X.columns.tolist(),
        'model_type': 'full_features_with_players'
    }
    joblib.dump(model_data, 'best_model_with_players.pkl')
    
    print(f"\n✓ Best model saved: best_model_with_players.pkl")
    
    # Comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print("\nModel Evolution:")
    print(f"  1. Basic features only:           70.30%")
    print(f"  2. + Enhanced (momentum/form):    70.30%")
    print(f"  3. + Player statistics:           {best['accuracy']*100:.2f}%")
    
    improvement = (best['accuracy'] - 0.7030) * 100
    if improvement > 0:
        print(f"\n🎉 Total improvement: +{improvement:.2f}%")
        if improvement >= 5:
            print("   🚀 SIGNIFICANT IMPROVEMENT!")
        elif improvement >= 2:
            print("   ✅ Good improvement")
        else:
            print("   📊 Marginal improvement")
    else:
        print(f"\n⚠️  Change: {improvement:.2f}%")
        print("   Player features may not be adding much predictive power")
    
    return best_model, best['accuracy'], results


if __name__ == '__main__':
    print("="*80)
    print(" "*15 + "ADVANCED TRAINING WITH PLAYER STATISTICS")
    print("="*80)
    
    model, accuracy, all_results = train_with_all_features()
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal accuracy: {accuracy*100:.2f}%")
    print("Model saved: best_model_with_players.pkl")
    print("\nReady for production predictions! 🏐")
