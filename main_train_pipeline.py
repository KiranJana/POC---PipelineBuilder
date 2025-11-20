"""
Main Training Pipeline
Orchestrates all 5 layers to train the complete hybrid system
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils.data_loader import load_and_preprocess_all
from utils.validation import (
    validate_layer1_rules,
    validate_layer2_patterns,
    validate_layer3_ml,
    validate_overall_system,
    generate_validation_report
)

# Import feature engineering
from features.feature_engineering import (
    engineer_all_features,
    create_interaction_features,
    clean_and_prepare_dataset,
    save_engineered_features
)

# Import layers
from layers.layer1_rules_engine import create_rules_engine
from layers.layer2_pattern_discovery import create_pattern_discovery_engine
from layers.layer3_ml_ensemble import create_ml_ensemble
from layers.layer4_recommendation import create_recommendation_engine
from layers.layer5_llm_explainer import create_llm_explainer
from config.llm_config import get_llm_config


def main():
    """Main training pipeline"""
    
    print("="*80)
    print("PIPELINE BUILDER - ENHANCED HYBRID POC")
    print("Training Pipeline - 5-Layer Architecture")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load and Preprocess Data
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*80)
    
    data_dict = load_and_preprocess_all()
    
    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    df_features = engineer_all_features(data_dict)
    df_features = create_interaction_features(df_features)
    df_features = clean_and_prepare_dataset(df_features)
    
    # Save engineered features
    save_engineered_features(df_features)
    
    print(f"\n[OK] Feature engineering complete")
    print(f"  Total features: {len(df_features.columns)}")
    print(f"  Total opportunities: {len(df_features)}")
    print(f"  Win rate: {df_features['is_won'].mean():.1%}")
    
    # ========================================================================
    # STEP 3: Layer 1 - Universal Rules Engine
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: LAYER 1 - UNIVERSAL RULES ENGINE")
    print("="*80)
    
    rules_engine = create_rules_engine()
    df_rule_results = rules_engine.apply_rules_batch(df_features)
    
    # ========================================================================
    # STEP 4: Layer 2 - Pattern Discovery
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: LAYER 2 - PATTERN DISCOVERY")
    print("="*80)
    
    pattern_engine = create_pattern_discovery_engine(min_support=0.02, min_confidence=0.65)
    
    # Mine patterns from data
    discovered_patterns = pattern_engine.mine_patterns(df_features)
    
    # Save patterns
    if discovered_patterns:
        pattern_engine.save_patterns()
    
    # Apply patterns to all opportunities
    df_pattern_results = pattern_engine.apply_patterns_batch(df_features)

    # VALIDATION: Check pattern diversity and quality
    if discovered_patterns:
        pattern_names = [p['pattern'] for p in discovered_patterns]
        unique_patterns = len(set(pattern_names))

        print(f"[VALIDATION] Pattern diversity: {unique_patterns}/{len(pattern_names)} unique patterns")

        if unique_patterns < len(pattern_names):
            print(f"[WARNING] Duplicate patterns detected - investigate discretization logic")
            duplicates = [p for p in pattern_names if pattern_names.count(p) > 1]
            print(f"[WARNING] Duplicates: {set(duplicates)}")

        # Check for pattern quality
        avg_confidence = np.mean([p['confidence'] for p in discovered_patterns])
        avg_support = np.mean([p['support'] for p in discovered_patterns])
        print(f"[VALIDATION] Pattern quality: Avg confidence {avg_confidence:.1%}, Avg support {avg_support:.0f} deals")

        if avg_confidence < 0.75:
            print(f"[WARNING] Average pattern confidence below 75% - consider stricter thresholds")
    else:
        print(f"[WARNING] No patterns discovered - check data quality and thresholds")
    
    # ========================================================================
    # STEP 5: Layer 3 - ML Ensemble
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: LAYER 3 - ML ENSEMBLE")
    print("="*80)
    
    ml_ensemble = create_ml_ensemble()
    
    # Prepare training data - train on ALL data for maximum learning/variation
    # But still only apply ML to messy middle cases (rules/patterns not covered)
    X, y, df_messy = ml_ensemble.prepare_training_data(df_features, df_rule_results, df_pattern_results, train_on_all_data=True)
    
    if X is not None and len(X) >= 30:
        # Train model with k-fold CV for robust evaluation
        print("[Layer 3] Using stratified 5-fold cross-validation for robust evaluation")
        ml_performance = ml_ensemble.train(X, y, use_kfold_cv=True, cv_folds=5)

        # Save model
        ml_ensemble.save_model()

        # Apply to all messy middle cases
        df_ml_results = ml_ensemble.apply_ml_batch(df_messy)
    else:
        print("[Layer 3] Insufficient data for ML training - skipping")
        df_ml_results = pd.DataFrame({
            'opportunity_id': df_features['opportunity_id'],
            'ml_prediction': 0,
            'ml_probability': 0.5,
            'ml_confidence': 0.5
        })
    
    # ========================================================================
    # STEP 6: Layer 4 - Recommendation Engine
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: LAYER 4 - RECOMMENDATION ENGINE")
    print("="*80)
    
    recommendation_engine = create_recommendation_engine(rules_engine, pattern_engine, ml_ensemble)
    
    # Generate recommendations for all opportunities
    recommendations = recommendation_engine.generate_recommendations_batch(
        df_features,
        df_historical=df_features  # Use same dataset for similarity search
    )
    
    # Save recommendations
    recommendation_engine.save_recommendations(recommendations)

    # VALIDATION: Check recommendation diversity and quality
    if recommendations:
        # Check for action diversity
        all_actions = []
        for rec in recommendations:
            if 'recommended_actions' in rec:
                all_actions.extend([action['action'] for action in rec['recommended_actions']])

        unique_actions = len(set(all_actions))
        total_actions = len(all_actions)
        print(f"[VALIDATION] Action diversity: {unique_actions}/{total_actions} unique actions")

        if unique_actions < total_actions * 0.5:  # Less than 50% unique actions
            print(f"[WARNING] Low action diversity - recommendations may be too generic")
            # Show most common actions
            from collections import Counter
            action_counts = Counter(all_actions).most_common(5)
            print(f"[WARNING] Top 5 most common actions:")
            for action, count in action_counts:
                print(f"  - '{action[:60]}...': {count} times")

        # Check recommendation type diversity
        rec_types = [r.get('recommendation_type', 'Unknown') for r in recommendations]
        unique_types = len(set(rec_types))
        print(f"[VALIDATION] Recommendation types: {unique_types}/{len(recommendations)} unique types")

        if unique_types <= 3:  # Very few different recommendation types
            print(f"[WARNING] Very few recommendation types - system may be too simplistic")
            type_counts = Counter(rec_types).most_common()
            print(f"[WARNING] Recommendation distribution: {dict(type_counts)}")
    
    # ========================================================================
    # STEP 7: Layer 5 - LLM Explanation Generator
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 7: LAYER 5 - LLM EXPLANATION GENERATOR")
    print("="*80)
    
    # Use LLM for explanations (configured in config/llm_config.py)
    llm_config = get_llm_config()
    llm_explainer = create_llm_explainer(**llm_config)
    
    # Generate explanations for first 10 recommendations (for demo)
    sample_recommendations = recommendations[:10]
    explanations = llm_explainer.generate_explanations_batch(sample_recommendations)
    
    # Save explanations
    llm_explainer.save_explanations(explanations)
    
    # Print sample explanation
    if explanations:
        print(f"\n[Layer 5] Sample Explanation:")
        print("="*80)
        try:
            print(explanations[0]['explanation'])
        except UnicodeEncodeError:
            print("[Unicode encoding issue - explanation contains special characters]")
            print(f"Explanation length: {len(explanations[0]['explanation'])} characters")
        print("="*80)
    
    # ========================================================================
    # STEP 8: Validation
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 8: VALIDATION")
    print("="*80)
    
    # Validate each layer
    layer1_results = validate_layer1_rules(df_features, df_rule_results)
    layer2_results = validate_layer2_patterns(df_features, df_pattern_results)
    
    if X is not None and len(X) >= 30:
        layer3_results = validate_layer3_ml(df_messy, df_ml_results)
    else:
        layer3_results = None
    
    overall_results = validate_overall_system(df_features, recommendations)
    
    # Generate validation report
    validation_report = generate_validation_report(
        layer1_results,
        layer2_results,
        layer3_results,
        overall_results
    )
    
    # ========================================================================
    # STEP 9: Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    summary = f"""
[OK] Data Loaded: {len(df_features)} opportunities from 8 data sources

[OK] Features Engineered: {len(df_features.columns)} total features
  • Activity features (calls, emails, meetings, velocity)
  • Intent features (scores, surges, competitive signals)
  • Email features (engagement, response rate, exec involvement)
  • Marketing features (events, downloads, demos, page visits)
  • Contact features (buying committee, seniority, personas)
  • Historical features (customer status, previous deals)
  • Temporal features (trends, momentum, urgency)

[OK] Layer 1 (Rules): {df_rule_results['rule_matched'].sum()} matches ({df_rule_results['rule_matched'].sum()/len(df_features):.1%})
  Precision: {layer1_results['precision']:.1%} {'[PASS]' if layer1_results['meets_target'] else '[FAIL]'} (target: >=90%)

[OK] Layer 2 (Patterns): {len(discovered_patterns) if discovered_patterns else 0} patterns discovered
  Matches: {df_pattern_results['pattern_matched'].sum()} ({df_pattern_results['pattern_matched'].sum()/len(df_features):.1%})
  Precision: {layer2_results['precision']:.1%} {'[PASS]' if layer2_results['meets_target'] else '[FAIL]'} (target: >=70%)

[OK] Layer 3 (ML): {'Trained with 5-fold CV' if X is not None and len(X) >= 30 else 'Skipped (insufficient data)'}
  {f"CV Accuracy: {layer3_results['accuracy']:.1%} {'[PASS]' if layer3_results['meets_target'] else '[FAIL]'} (target: >=60%)" if layer3_results else "N/A"}

[OK] Layer 4 (Recommendations): {len(recommendations)} generated
  Average confidence: {overall_results['avg_confidence']:.1%}

[OK] Layer 5 (Explanations): {len(explanations)} generated (template-based)

[OK] Overall System Performance:
  Accuracy: {overall_results['accuracy']:.1%}
  Precision: {overall_results['precision']:.1%}
  Recall: {overall_results['recall']:.1%}
  F1-Score: {overall_results['f1_score']:.3f}

[FILES] Files Generated:
  - outputs/model_training_data_engineered.csv
  - outputs/trained_models/lgbm_model.pkl
  - outputs/discovered_patterns/patterns.json
  - outputs/recommendations/recommendations.json
  - outputs/recommendations/explanations.json
  - outputs/validation_reports/validation_report.json

{'[SUCCESS] ALL TARGETS MET - System ready for deployment' if validation_report['summary']['all_targets_met'] else '[WARN] Some targets not met - Review and refine'}

Next Steps:
1. Review validation report and sample recommendations
2. Test with main_score_pipeline.py for real-time scoring
3. Integrate with CRM/sales tools
4. Deploy to production
"""
    
    print(summary)
    
    print("\n" + "="*80)
    print("SUCCESS! Training pipeline complete.")
    print("="*80)


if __name__ == "__main__":
    main()

