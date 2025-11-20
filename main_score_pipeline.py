"""
Real-time Scoring Pipeline
Score new opportunities using trained hybrid system
"""

import os
import sys

# Ensure we're running from the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import warnings
import json
import os
from scipy import stats
warnings.filterwarnings('ignore')

# Import utilities
from utils.data_loader import load_and_preprocess_all

# Import feature engineering
from features.feature_engineering import (
    engineer_all_features,
    create_interaction_features,
    clean_and_prepare_dataset
)

# Import layers
from layers.layer1_rules_engine import create_rules_engine
from layers.layer2_pattern_discovery import create_pattern_discovery_engine
from layers.layer3_ml_ensemble import create_ml_ensemble
from layers.layer4_recommendation import create_recommendation_engine
from layers.layer5_llm_explainer import create_llm_explainer
from config.llm_config import get_llm_config


def detect_feature_drift(df_current, drift_baseline_path='outputs/model_training_data_engineered.csv', alert_threshold=0.05):
    """
    Detect feature drift by comparing current data distributions to training baseline

    Args:
        df_current: Current scoring data (DataFrame)
        drift_baseline_path: Path to baseline training data
        alert_threshold: Statistical significance threshold for drift alerts

    Returns:
        Dict with drift analysis results
    """
    print("\n" + "="*80)
    print("FEATURE DRIFT DETECTION")
    print("="*80)

    drift_results = {
        'drift_detected': False,
        'significant_drifts': [],
        'drift_summary': {},
        'recommendations': []
    }

    # Load baseline training data
    try:
        df_baseline = pd.read_csv(drift_baseline_path)
        print(f"[DRIFT] Loaded baseline data: {len(df_baseline)} training samples")
    except FileNotFoundError:
        print(f"[DRIFT] WARNING: Baseline data not found at {drift_baseline_path}")
        print("[DRIFT] Skipping drift detection")
        return drift_results

    # Get numeric features (exclude IDs and targets)
    exclude_cols = ['opportunity_id', 'account_id', 'is_won']
    numeric_cols = [col for col in df_current.columns
                   if col not in exclude_cols and df_current[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"[DRIFT] Analyzing {len(numeric_cols)} numeric features for drift")

    significant_drifts = []

    for col in numeric_cols:
        if col not in df_baseline.columns:
            print(f"[DRIFT] WARNING: Feature '{col}' not found in baseline data")
            continue

        # Get non-null values for both datasets
        baseline_vals = df_baseline[col].dropna()
        current_vals = df_current[col].dropna()

        if len(baseline_vals) < 10 or len(current_vals) < 10:
            continue  # Skip features with insufficient data

        try:
            # Statistical tests for distribution differences
            # 1. Kolmogorov-Smirnov test (compares distributions)
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_vals, current_vals)

            # 2. Mann-Whitney U test (compares medians)
            mw_stat, mw_pvalue = stats.mannwhitneyu(baseline_vals, current_vals, alternative='two-sided')

            # 3. Compare basic statistics
            baseline_mean = baseline_vals.mean()
            current_mean = current_vals.mean()
            baseline_std = baseline_vals.std()
            current_std = current_vals.std()

            mean_diff_pct = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-10)
            std_diff_pct = abs(current_std - baseline_std) / (abs(baseline_std) + 1e-10)

            # Check for significant drift
            is_significant = (
                ks_pvalue < alert_threshold or  # Distribution changed
                mw_pvalue < alert_threshold or  # Median changed
                mean_diff_pct > 0.25 or         # Mean changed by >25%
                std_diff_pct > 0.25            # Std changed by >25%
            )

            if is_significant:
                drift_info = {
                    'feature': col,
                    'drift_type': 'distribution' if ks_pvalue < alert_threshold else 'statistic',
                    'ks_pvalue': float(ks_pvalue),
                    'mw_pvalue': float(mw_pvalue),
                    'baseline_mean': float(baseline_mean),
                    'current_mean': float(current_mean),
                    'mean_diff_pct': float(mean_diff_pct),
                    'baseline_std': float(baseline_std),
                    'current_std': float(current_std),
                    'std_diff_pct': float(std_diff_pct),
                    'baseline_samples': len(baseline_vals),
                    'current_samples': len(current_vals)
                }
                significant_drifts.append(drift_info)

        except Exception as e:
            print(f"[DRIFT] Error analyzing feature '{col}': {e}")
            continue

    # Summarize results
    drift_results['significant_drifts'] = significant_drifts
    drift_results['drift_detected'] = len(significant_drifts) > 0

    if drift_results['drift_detected']:
        print(f"\n[DRIFT] ⚠️  DRIFT DETECTED in {len(significant_drifts)} features!")
        print("[DRIFT] Top drifting features:")

        # Sort by most significant drift (lowest p-value)
        sorted_drifts = sorted(significant_drifts, key=lambda x: min(x['ks_pvalue'], x['mw_pvalue']))

        for i, drift in enumerate(sorted_drifts[:5], 1):  # Show top 5
            print(f"  {i}. {drift['feature']}: {drift['drift_type']} drift")
            print(f"     KS p-value: {drift['ks_pvalue']:.4f}, Mean diff: {drift['mean_diff_pct']:.1%}")

        # Generate recommendations
        drift_results['recommendations'] = [
            "Consider retraining model with recent data",
            "Monitor prediction accuracy for degradation",
            "Review data collection processes for changes",
            f"Focus on {len(significant_drifts)} drifting features in next model iteration"
        ]

        print(f"\n[DRIFT] Recommendations:")
        for rec in drift_results['recommendations']:
            print(f"  • {rec}")

    else:
        print(f"\n[DRIFT] ✅ No significant drift detected in {len(numeric_cols)} features")
        print(f"[DRIFT] All features within normal variation (p > {alert_threshold})")

    # Save drift results
    drift_output_path = 'outputs/drift_analysis.json'
    with open(drift_output_path, 'w') as f:
        json.dump(drift_results, f, indent=2, default=str)
    print(f"[DRIFT] Drift analysis saved to {drift_output_path}")

    return drift_results


def load_trained_models():
    """Load all trained models and engines"""
    print("="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)
    
    # Layer 1: Rules Engine (no training needed)
    rules_engine = create_rules_engine()
    print("[OK] Layer 1: Rules Engine loaded")
    
    # Layer 2: Pattern Discovery (load discovered patterns)
    pattern_engine = create_pattern_discovery_engine()
    try:
        pattern_engine.load_patterns()
        print("[OK] Layer 2: Pattern Discovery loaded")
    except FileNotFoundError:
        print("[WARNING] Layer 2: No patterns found - run training first")
    
    # Layer 3: ML Ensemble (load trained model)
    ml_ensemble = create_ml_ensemble()
    try:
        ml_ensemble.load_model()
        print("[OK] Layer 3: ML Ensemble loaded")
    except FileNotFoundError:
        print("[WARNING] Layer 3: No model found - run training first")
    
    # Layer 4: Recommendation Engine
    recommendation_engine = create_recommendation_engine(rules_engine, pattern_engine, ml_ensemble)
    print("[OK] Layer 4: Recommendation Engine loaded")
    
    # Layer 5: LLM Explainer (configured in config/llm_config.py)
    llm_config = get_llm_config()
    llm_explainer = create_llm_explainer(**llm_config)
    print("[OK] Layer 5: LLM Explainer loaded")
    
    return rules_engine, pattern_engine, ml_ensemble, recommendation_engine, llm_explainer


def score_opportunities(df_features, rules_engine, pattern_engine, ml_ensemble, recommendation_engine, llm_explainer):
    """
    Score opportunities and generate recommendations
    """
    print("\n" + "="*80)
    print("SCORING OPPORTUNITIES")
    print("="*80)
    
    print(f"\n[OK] Scoring {len(df_features)} opportunities...")
    
    # Generate recommendations
    recommendations = recommendation_engine.generate_recommendations_batch(
        df_features,
        df_historical=df_features  # Use for similarity search
    )
    
    # Generate explanations for top recommendations (not just high-confidence)
    # Sort by ensemble score (weighted prioritization) and take top 10 for LLM explanations
    sorted_recs = sorted(recommendations, key=lambda x: x.get('ensemble_score', x.get('confidence', 0)), reverse=True)
    top_recommendations = sorted_recs[:10]  # Top 10 by ensemble score

    if top_recommendations:
        # Get rate limit from LLM explainer instance
        requests_per_minute = getattr(llm_explainer, 'requests_per_minute', 10)
        print(f"\n[OK] Generating LLM explanations for top {len(top_recommendations)} recommendations...")
        print(f"[OK] Rate limiting: {requests_per_minute} requests per minute")
        explanations = llm_explainer.generate_explanations_batch(top_recommendations, requests_per_minute)
    else:
        explanations = []
    
    return recommendations, explanations


def display_llm_explanations(explanations):
    """Display LLM explanations in a nicely formatted way"""
    if not explanations:
        print("\n[INFO] No LLM explanations generated (no high-confidence recommendations)")
        return

    print("\n" + "="*100)
    print("LLM-GENERATED EXPLANATIONS")
    print("="*100)

    for i, explanation in enumerate(explanations, 1):
        opp_id = explanation.get('opportunity_id', 'Unknown')
        exp_text = explanation.get('explanation', '')

        # Clean up Unicode issues for Windows console display
        # Simply remove all non-ASCII characters to avoid encoding issues
        exp_text = ''.join(char for char in exp_text if ord(char) < 128)

        # Truncate if too long for console
        if len(exp_text) > 1500:
            exp_text = exp_text[:1500] + "...\n\n[Explanation truncated for console display]"

        print(f"\nEXPLANATION {i}: Opportunity {opp_id}")
        print("-" * 80)
        print(exp_text)
        print("-" * 80)

    print(f"\n[SUCCESS] Displayed {len(explanations)} LLM-generated explanations")


def display_top_recommendations(recommendations, top_n=10):
    """Display top N recommendations"""
    print("\n" + "="*80)
    print(f"TOP {top_n} RECOMMENDATIONS")
    print("="*80)
    
    # Sort by ensemble score (weighted combination of all sources) for better prioritization
    sorted_recs = sorted(recommendations, key=lambda x: x.get('ensemble_score', x.get('confidence', 0)), reverse=True)
    
    for i, rec in enumerate(sorted_recs[:top_n], 1):
        print(f"\n{i}. {rec.get('account', 'Unknown')} (ID: {rec.get('opportunity_id', 'Unknown')})")
        print(f"   Recommendation: {rec.get('recommendation_type', 'Unknown')}")
        print(f"   Primary Confidence: {rec.get('confidence', 0):.0%}")
        print(f"   Ensemble Score: {rec.get('ensemble_score', 0):.0%} (weighted prioritization)")
        print(f"   Source: {rec.get('primary_source', 'Unknown')}")
        
        # Show key signals
        if rec.get('explanation_structured'):
            key_signals = rec['explanation_structured'].get('key_signals', [])
            if key_signals:
                print(f"   Key Signals: {', '.join(key_signals[:3])}")
        
        # Show top action
        if rec.get('recommended_actions'):
            top_action = rec['recommended_actions'][0]
            print(f"   Top Action: {top_action.get('action', 'Unknown')} [{top_action.get('urgency', 'MEDIUM')}]")


def score_single_opportunity(opportunity_id, df_features, rules_engine, pattern_engine, ml_ensemble, recommendation_engine, llm_explainer):
    """
    Score a single opportunity and display detailed recommendation
    """
    print("\n" + "="*80)
    print(f"SCORING OPPORTUNITY: {opportunity_id}")
    print("="*80)
    
    # Find opportunity
    opp = df_features[df_features['opportunity_id'] == opportunity_id]
    
    if len(opp) == 0:
        print(f"[ERROR] Opportunity {opportunity_id} not found")
        return None
    
    opp_features = opp.iloc[0].to_dict()
    
    # Generate recommendation
    recommendation = recommendation_engine.generate_recommendation(
        opp_features,
        df_historical=df_features,
        feature_cols=ml_ensemble.feature_cols if ml_ensemble.feature_cols else []
    )
    
    # Generate explanation
    explanation = llm_explainer.generate_explanation(recommendation)
    
    # Display
    try:
        print("\n" + explanation)
    except UnicodeEncodeError:
        print("\n[Unicode encoding issue - explanation contains special characters]")
        print(f"Explanation length: {len(explanation)} characters")
    
    return recommendation


def main():
    """Main scoring pipeline"""
    
    print("="*80)
    print("PIPELINE BUILDER - REAL-TIME SCORING")
    print("Score opportunities using trained hybrid system")
    print("="*80)
    
    # Load trained models
    rules_engine, pattern_engine, ml_ensemble, recommendation_engine, llm_explainer = load_trained_models()
    
    # Load and prepare data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_dict = load_and_preprocess_all()
    
    # Engineer features
    print("\n" + "="*80)
    print("ENGINEERING FEATURES")
    print("="*80)
    
    df_features = engineer_all_features(data_dict)
    df_features = create_interaction_features(df_features)
    df_features = clean_and_prepare_dataset(df_features)
    
    print(f"\n[OK] Features ready: {len(df_features)} opportunities")

    # Check for feature drift before scoring
    drift_results = detect_feature_drift(df_features)

    # Score all opportunities
    recommendations, explanations = score_opportunities(
        df_features,
        rules_engine,
        pattern_engine,
        ml_ensemble,
        recommendation_engine,
        llm_explainer
    )
    
    # Display top recommendations
    display_top_recommendations(recommendations, top_n=10)

    # Display LLM explanations
    display_llm_explanations(explanations)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    recommendation_engine.save_recommendations(
        recommendations,
        output_path='outputs/recommendations/scoring_recommendations.json'
    )
    
    if explanations:
        llm_explainer.save_explanations(
            explanations,
            output_path='outputs/recommendations/scoring_explanations.json'
        )
    
    # Example: Score a specific opportunity
    if len(df_features) > 0:
        sample_opp_id = df_features.iloc[0]['opportunity_id']
        print("\n" + "="*80)
        print("EXAMPLE: DETAILED SCORING")
        print("="*80)
        
        score_single_opportunity(
            sample_opp_id,
            df_features,
            rules_engine,
            pattern_engine,
            ml_ensemble,
            recommendation_engine,
            llm_explainer
        )
    
    # Summary
    print("\n" + "="*80)
    print("SCORING COMPLETE - SUMMARY")
    print("="*80)
    
    high_conf = sum(1 for r in recommendations if r.get('confidence', 0) > 0.7)
    med_conf = sum(1 for r in recommendations if 0.5 < r.get('confidence', 0) <= 0.7)
    low_conf = sum(1 for r in recommendations if r.get('confidence', 0) <= 0.5)
    
    summary = f"""
[OK] Opportunities Scored: {len(recommendations)}

[OK] Confidence Distribution:
  • High (>70%): {high_conf} opportunities
  • Medium (50-70%): {med_conf} opportunities
  • Low (<50%): {low_conf} opportunities

[OK] Source Distribution:
"""
    
    source_counts = {}
    for r in recommendations:
        source = r.get('primary_source', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"  • {source}: {count} ({count/len(recommendations):.1%})\n"
    
    summary += f"""
📁 Results Saved:
  ├── outputs/recommendations/scoring_recommendations.json
  └── outputs/recommendations/scoring_explanations.json

Next Steps:
1. Review high-confidence recommendations
2. Integrate with CRM for automated task creation
3. Set up continuous monitoring for new opportunities
4. Retrain models quarterly with fresh data
"""
    
    print(summary)
    
    print("\n" + "="*80)
    print("SUCCESS! Scoring pipeline complete.")
    print("="*80)


if __name__ == "__main__":
    main()

