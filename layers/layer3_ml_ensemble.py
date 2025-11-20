"""
Layer 3: ML Ensemble
RandomForest classifier with confidence calibration
Handles the "messy middle" cases not covered by rules or patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
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

        print(f"[Layer 3] Initial feature count: {len(feature_cols)}")

        # AGGRESSIVE FEATURE SELECTION: Cut to 20-30 most important features
        # This prevents overfitting on small datasets
        if len(feature_cols) > 30:
            print(f"[Layer 3] Applying feature selection (RFE) to reduce from {len(feature_cols)} to ~25 features...")

            X_temp = df_training[feature_cols].fillna(0).astype(float)
            y_temp = df_training['is_won']

            # Use RFE with RandomForest for feature selection
            # n_features_to_select=25 is aggressive but necessary for small datasets
            rfe_selector = RFE(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                n_features_to_select=25,  # Aggressive cut to prevent overfitting
                step=5  # Remove 5 features at a time for efficiency
            )

            try:
                X_selected = rfe_selector.fit_transform(X_temp, y_temp)
                selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe_selector.support_[i]]

                print(f"[Layer 3] RFE selected {len(selected_features)} features:")
                for i, feature in enumerate(selected_features[:10], 1):  # Show top 10
                    print(f"  {i:2d}. {feature}")

                if len(selected_features) > 10:  # Show count if more than 10
                    print(f"     ... and {len(selected_features) - 10} more")

                feature_cols = selected_features

            except Exception as e:
                print(f"[Layer 3] RFE failed ({e}), using all features")
                # Continue with all features if RFE fails

        self.feature_cols = feature_cols

        X = df_training[feature_cols].fillna(0).astype(float)
        y = df_training['is_won']

        training_context = "all data" if train_on_all_data else "messy middle"
        print(f"[Layer 3] Feature matrix: {X.shape}")
        print(f"[Layer 3] Win rate in {training_context}: {y.mean():.1%}")

        return X, y, df_training
    
    def train(self, X, y, test_size=0.2, random_state=42, use_kfold_cv=True, cv_folds=5):
        """
        Train RandomForest model with calibration and robust evaluation

        Args:
            X, y: Training data
            test_size: For backward compatibility single split (if use_kfold_cv=False)
            random_state: Random seed
            use_kfold_cv: Whether to use stratified k-fold CV instead of single split
            cv_folds: Number of CV folds (default 5)
        """
        print(f"\n[Layer 3] Training RandomForest model...")

        if use_kfold_cv:
            print(f"[Layer 3] Using {cv_folds}-fold stratified cross-validation for robust evaluation")
            return self._train_with_kfold_cv(X, y, cv_folds, random_state)
        else:
            print(f"[Layer 3] Using single train/test split (test_size={test_size})")
            return self._train_with_single_split(X, y, test_size, random_state)

    def _train_with_kfold_cv(self, X, y, cv_folds=5, random_state=42):
        """Train with stratified k-fold cross-validation for robust evaluation"""

        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Train base RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Conservative depth to prevent overfitting
            min_samples_split=20,  # Higher min samples for robustness
            min_samples_leaf=10,   # Higher min leaf size
            max_features='sqrt',   # Feature subset selection
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )

        # Cross-validation scores
        cv_accuracy_scores = []
        cv_precision_scores = []
        cv_recall_scores = []
        cv_auc_scores = []

        print(f"[Layer 3] Running {cv_folds}-fold cross-validation...")

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Train on this fold
            self.model.fit(X_train_fold, y_train_fold)

            # Calibrate on this fold
            calibrated_model_fold = CalibratedClassifierCV(
                self.model, method='sigmoid', cv=3
            )
            calibrated_model_fold.fit(X_train_fold, y_train_fold)

            # Evaluate on validation fold
            y_pred_val = calibrated_model_fold.predict(X_val_fold)
            y_proba_val = calibrated_model_fold.predict_proba(X_val_fold)[:, 1]

            fold_accuracy = accuracy_score(y_val_fold, y_pred_val)
            fold_precision = precision_score(y_val_fold, y_pred_val, zero_division=0)
            fold_recall = recall_score(y_val_fold, y_pred_val, zero_division=0)
            fold_auc = roc_auc_score(y_val_fold, y_proba_val)

            cv_accuracy_scores.append(fold_accuracy)
            cv_precision_scores.append(fold_precision)
            cv_recall_scores.append(fold_recall)
            cv_auc_scores.append(fold_auc)

            fold_results.append({
                'fold': fold,
                'accuracy': fold_accuracy,
                'precision': fold_precision,
                'recall': fold_recall,
                'auc': fold_auc
            })

            print(f"  Fold {fold}: Acc={fold_accuracy:.1%}, Prec={fold_precision:.1%}, Rec={fold_recall:.1%}, AUC={fold_auc:.3f}")

        # Calculate CV statistics
        cv_stats = {
            'cv_mean_accuracy': np.mean(cv_accuracy_scores),
            'cv_std_accuracy': np.std(cv_accuracy_scores),
            'cv_mean_precision': np.mean(cv_precision_scores),
            'cv_std_precision': np.std(cv_precision_scores),
            'cv_mean_recall': np.mean(cv_recall_scores),
            'cv_std_recall': np.std(cv_recall_scores),
            'cv_mean_auc': np.mean(cv_auc_scores),
            'cv_std_auc': np.std(cv_auc_scores),
        }

        print(f"\n[Layer 3] Cross-Validation Results (Mean ± Std):")
        print(f"  Accuracy: {cv_stats['cv_mean_accuracy']:.1%} ± {cv_stats['cv_std_accuracy']:.1%}")
        print(f"  Precision: {cv_stats['cv_mean_precision']:.1%} ± {cv_stats['cv_std_precision']:.1%}")
        print(f"  Recall: {cv_stats['cv_mean_recall']:.1%} ± {cv_stats['cv_std_recall']:.1%}")
        print(f"  ROC-AUC: {cv_stats['cv_mean_auc']:.3f} ± {cv_stats['cv_std_auc']:.3f}")

        # Train final model on ALL data for deployment
        print(f"\n[Layer 3] Training final model on all {len(X)} samples...")
        self.model.fit(X, y)

        # Calibrate final model
        print("[Layer 3] Calibrating final model...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='sigmoid', cv=3
        )
        self.calibrated_model.fit(X, y)

        # Performance assessment
        mean_accuracy = cv_stats['cv_mean_accuracy']
        if mean_accuracy >= 0.60 and mean_accuracy <= 0.85:
            print(f"[Layer 3] [OK] Performance in realistic range (60-85%)")
        elif mean_accuracy > 0.85:
            print(f"[Layer 3] [WARN] Performance high - may indicate remaining leakage")
        else:
            print(f"[Layer 3] [WARN] Performance below target - check feature quality")

        # Feature importance from final model
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n[Layer 3] Top 10 Features (from final model):")
        for idx, row in enumerate(self.feature_importance.head(10).itertuples(), 1):
            print(f"  {idx:2d}. {row.feature:40s} {row.importance:6.1f}")

        print(f"\n[Layer 3] Using feature importance for explanations (SHAP disabled for POC)")

        return cv_stats

    def _train_with_single_split(self, X, y, test_size=0.2, random_state=42):
        """Legacy training method with single train/test split"""
        print(f"[Layer 3] Using single train/test split for backward compatibility...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"[Layer 3] Train set: {len(X_train)} samples")
        print(f"[Layer 3] Test set: {len(X_test)} samples")

        # Train base RandomForest model
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
            self.model, method='sigmoid', cv=3
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

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n[Layer 3] Top 10 Features:")
        for idx, row in enumerate(self.feature_importance.head(10).itertuples(), 1):
            print(f"  {idx:2d}. {row.feature:40s} {row.importance:6.1f}")

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

