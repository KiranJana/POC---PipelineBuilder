"""
Layer 3: ML Ensemble
RandomForest classifier with confidence calibration
Handles the "messy middle" cases not covered by rules or patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import json


class MLEnsemble:
    """
    Layer 3: ML Ensemble
    RandomForest with calibration for robust predictions on small datasets
    """
    
    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.feature_cols = None
        self.explainer = None
        self.feature_importance = None
        
        print("[Layer 3] ML Ensemble initialized")
    
    def prepare_training_data(self, df_features, df_rule_results, df_pattern_results, train_on_all_data=True):
        """
        Prepare training data - option to train on all data for maximum accuracy
        or only "messy middle" for consistency with rules
        """
        print("\n[Layer 3] Preparing training data...")

        # Merge results
        df_merged = df_features.copy()
        df_merged = df_merged.merge(df_rule_results[['opportunity_id', 'rule_matched']], on='opportunity_id', how='left')
        df_merged = df_merged.merge(df_pattern_results[['opportunity_id', 'pattern_matched']], on='opportunity_id', how='left')

        # Fill NaN
        df_merged['rule_matched'] = df_merged['rule_matched'].fillna(False)
        df_merged['pattern_matched'] = df_merged['pattern_matched'].fillna(False)

            # STRATEGY: Train on ALL data for robustness, apply only to messy middle
            # Rationale: In production, we may encounter patterns not seen during training.
            # By training on all data (including rule/pattern-covered cases), the ML model
            # learns the full landscape. However, during inference (Layer 4), the model
            # is ONLY applied to opportunities that don't match rules or patterns.
            # This ensures: (1) Rules/patterns take priority, (2) ML is robust when applied
        df_training = df_merged.copy()
        
        print(f"[Layer 3] Training ML on ALL {len(df_training)} opportunities")
        print(f"[Layer 3] Strategy: Learn full landscape, apply only to messy middle in Layer 4")
        print(f"[Layer 3] Rule-covered: {df_merged['rule_matched'].sum()}, Pattern-covered: {df_merged['pattern_matched'].sum()}")


        print(f"[Layer 3] Total opportunities: {len(df_features)}")
        print(f"[Layer 3] Handled by rules: {df_merged['rule_matched'].sum()}")
        print(f"[Layer 3] Handled by patterns: {df_merged['pattern_matched'].sum()}")
        print(f"[Layer 3] Training ML on messy middle only: {len(df_training)} ({len(df_training)/len(df_features):.1%})")

        if len(df_training) < 30:
            print("[Layer 3] WARNING: Insufficient data for ML training (need at least 30 samples)")
            return None, None, None
        
        # Select features
        exclude_cols = [
            'opportunity_id', 'account_id', 'is_won',
            'rule_matched', 'pattern_matched',
            'engagement_trend', 'urgency', 'opportunity_type',  # Categorical
            'is_stalled', 'had_competitor_intent',  # Already used in rules
        ]
        
        feature_cols = [
            col for col in df_training.columns
            if col not in exclude_cols and df_training[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        # Remove constant features
        constant_features = [col for col in feature_cols if df_training[col].nunique() <= 1]
        if constant_features:
            print(f"[Layer 3] Removing {len(constant_features)} constant features")
            feature_cols = [col for col in feature_cols if col not in constant_features]

        self.feature_cols = feature_cols

        X = df_training[feature_cols].fillna(0).astype(float)
        y = df_training['is_won']

        training_context = "all data" if train_on_all_data else "messy middle"
        print(f"[Layer 3] Feature matrix: {X.shape}")
        print(f"[Layer 3] Win rate in {training_context}: {y.mean():.1%}")

        return X, y, df_training
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train LGBM model with calibration
        """
        print(f"\n[Layer 3] Training RandomForest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"[Layer 3] Train set: {len(X_train)} samples")
        print(f"[Layer 3] Test set: {len(X_test)} samples")
        
        # Train base RandomForest model (robust for small datasets, less prone to overfitting)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Conservative depth to prevent overfitting
            min_samples_split=20,  # Higher min samples for robustness
            min_samples_leaf=10,   # Higher min leaf size
            max_features='sqrt',   # Feature subset selection
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        print("[Layer 3] Training base model...")
        self.model.fit(X_train, y_train)
        
        # Calibrate model
        print("[Layer 3] Calibrating confidence scores...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method='sigmoid',
            cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.calibrated_model.predict(X_train)
        y_pred_test = self.calibrated_model.predict(X_test)
        y_proba_test = self.calibrated_model.predict_proba(X_test)[:, 1]
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_auc = roc_auc_score(y_test, y_proba_test)
        
        print(f"\n[Layer 3] Model Performance:")
        print(f"  Training Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy: {test_acc:.1%}")
        print(f"  Test Precision: {test_precision:.1%}")
        print(f"  Test Recall: {test_recall:.1%}")
        print(f"  Test ROC-AUC: {test_auc:.3f}")
        
        if test_acc >= 0.60 and test_acc <= 0.85:
            print(f"[Layer 3] [OK] Performance in realistic range (60-85%)")
        elif test_acc > 0.85:
            print(f"[Layer 3] [WARN] Performance high - may indicate remaining leakage")
        else:
            print(f"[Layer 3] [WARN] Performance below target - check feature quality")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n[Layer 3] Top 10 Features:")
        for idx, row in enumerate(self.feature_importance.head(10).itertuples(), 1):
            print(f"  {idx:2d}. {row.feature:40s} {row.importance:6.1f}")

        print(f"\n[Layer 3] Using feature importance for explanations (SHAP disabled for POC)")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc
        }
    
    def predict(self, X):
        """Predict with calibrated model"""
        if self.calibrated_model is None:
            raise ValueError("Model not trained yet")
        
        return self.calibrated_model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities with calibrated model"""
        if self.calibrated_model is None:
            raise ValueError("Model not trained yet")
        
        return self.calibrated_model.predict_proba(X)
    
    def explain_prediction(self, X_single):
        """
        Generate explanation for a single prediction using feature importance
        """
        if self.feature_importance is None:
            return None

        feature_contributions = []
        for _, row in self.feature_importance.head(10).iterrows():
            feature = row['feature']
            if feature in X_single.columns if isinstance(X_single, pd.DataFrame) else feature in self.feature_cols:
                feature_value = float(X_single[feature].iloc[0]) if isinstance(X_single, pd.DataFrame) else float(X_single[self.feature_cols.index(feature)])
                feature_contributions.append({
                    'feature': feature,
                    'value': feature_value,
                    'importance': float(row['importance']),
                    'explanation_type': 'global_importance'
                })

        return feature_contributions[:10]
    
    def apply_ml_batch(self, df_features):
        """
        Apply ML model to a batch of opportunities
        """
        print(f"\n[Layer 3] Applying ML model to {len(df_features)} opportunities...")
        
        if self.calibrated_model is None:
            print("[Layer 3] ERROR: Model not trained")
            return None
        
        # Prepare features
        X = df_features[self.feature_cols].fillna(0).astype(float)
        
        # Predict
        predictions = self.calibrated_model.predict(X)
        probabilities = self.calibrated_model.predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'opportunity_id': df_features['opportunity_id'],
            'ml_prediction': predictions,
            'ml_probability': probabilities,
            'ml_confidence': probabilities  # Calibrated probabilities are confidence scores
        })
        
        print(f"[Layer 3] ML predictions complete")
        print(f"  Predicted Won: {predictions.sum()} ({predictions.sum()/len(predictions):.1%})")
        print(f"  Average confidence: {probabilities.mean():.1%}")
        
        return results
    
    def save_model(self, model_path='outputs/trained_models/rf_model.pkl'):
        """Save trained model"""
        if self.calibrated_model is None:
            print("[Layer 3] ERROR: No model to save")
            return
        
        joblib.dump(self.calibrated_model, model_path)
        
        # Save feature columns
        feature_path = model_path.replace('.pkl', '_features.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_cols, f)
        
        # Save feature importance
        importance_path = model_path.replace('.pkl', '_importance.csv')
        self.feature_importance.to_csv(importance_path, index=False)
        
        print(f"\n[Layer 3] Model saved:")
        print(f"  Model: {model_path}")
        print(f"  Features: {feature_path}")
        print(f"  Importance: {importance_path}")
    
    def load_model(self, model_path='outputs/trained_models/rf_model.pkl'):
        """Load trained model"""
        self.calibrated_model = joblib.load(model_path)
        
        # Load feature columns
        feature_path = model_path.replace('.pkl', '_features.json')
        with open(feature_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        # Load feature importance
        importance_path = model_path.replace('.pkl', '_importance.csv')
        self.feature_importance = pd.read_csv(importance_path)

        # Use feature importance for explanations (SHAP disabled for POC)
        self.model = self.calibrated_model
        print(f"[Layer 3] Model loaded with feature importance explanations")
        
        print(f"\n[Layer 3] Model loaded from {model_path}")


def create_ml_ensemble():
    """Factory function to create ML ensemble"""
    return MLEnsemble()

