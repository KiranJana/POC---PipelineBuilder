"""
Layer 3: ML Ensemble
LGBM classifier with SHAP explainability and confidence calibration
Handles the "messy middle" cases not covered by rules or patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from lightgbm import LGBMClassifier
import shap
import joblib
import json


class MLEnsemble:
    """
    Layer 3: ML Ensemble
    LightGBM with calibration and SHAP explainability
    """
    
    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.feature_cols = None
        self.explainer = None
        self.feature_importance = None
        
        print("[Layer 3] ML Ensemble initialized")
    
    def prepare_training_data(self, df_features, df_rule_results, df_pattern_results, train_on_all_data=False):
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

        if train_on_all_data:
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
        else:
            # Alternative approach: only train on messy middle
            messy_mask = (~df_merged['rule_matched']) & (~df_merged['pattern_matched'])
            df_training = df_merged[messy_mask].copy()

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
        print(f"\n[Layer 3] Training LGBM model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"[Layer 3] Train set: {len(X_train)} samples")
        print(f"[Layer 3] Test set: {len(X_test)} samples")
        
        # Train base LGBM model (smaller to avoid overfitting)
        self.model = LGBMClassifier(
            objective='binary',
            n_estimators=100,
            max_depth=4,  # Smaller depth
            learning_rate=0.05,
            num_leaves=15,  # Fewer leaves
            min_child_samples=20,  # Higher min samples
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1
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
        
        # Initialize SHAP explainer
        print(f"\n[Layer 3] Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
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
        Generate explanation for a single prediction
        Uses SHAP if available, otherwise falls back to feature importance
        """
        if self.explainer is None:
            # Fallback: Use global feature importance from training
            # Note: This doesn't give instance-specific explanations like SHAP,
            # but shows which features are generally important to the model
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
                        'explanation_type': 'global_importance'  # Not instance-specific
                    })

            return feature_contributions[:10]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_single)
        
        # If binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Get top features
        feature_contributions = []
        for i, feature in enumerate(self.feature_cols):
            feature_contributions.append({
                'feature': feature,
                'value': float(X_single[feature].iloc[0] if isinstance(X_single, pd.DataFrame) else X_single[i]),
                'shap_value': float(shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]),
                'importance': float(self.feature_importance[self.feature_importance['feature'] == feature]['importance'].values[0])
            })
        
        # Sort by absolute SHAP value
        feature_contributions = sorted(feature_contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        
        return feature_contributions[:10]  # Top 10
    
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
    
    def save_model(self, model_path='outputs/trained_models/lgbm_model.pkl'):
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
    
    def load_model(self, model_path='outputs/trained_models/lgbm_model.pkl'):
        """Load trained model"""
        self.calibrated_model = joblib.load(model_path)
        
        # Load feature columns
        feature_path = model_path.replace('.pkl', '_features.json')
        with open(feature_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        # Load feature importance
        importance_path = model_path.replace('.pkl', '_importance.csv')
        self.feature_importance = pd.read_csv(importance_path)
        
        # SHAP explainer disabled for POC to avoid dependency/compatibility issues
        # Reason: SHAP can be slow and requires additional dependencies
        # Alternative: Use feature importance from training (loaded above)
        # For MVP: Consider re-enabling SHAP if detailed ML explainability is required
        self.model = self.calibrated_model
        self.explainer = None
        print(f"[Layer 3] SHAP explainer disabled (using feature importance instead)")
        
        print(f"\n[Layer 3] Model loaded from {model_path}")


def create_ml_ensemble():
    """Factory function to create ML ensemble"""
    return MLEnsemble()

