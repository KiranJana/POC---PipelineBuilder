"""Validation framework to measure each layer separately"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
from datetime import datetime


def validate_layer1_rules(df_features, df_rule_results):
    """
    Validate Layer 1: Universal Rules Engine
    Target: ≥90% precision
    """
    print("\n" + "="*80)
    print("VALIDATING LAYER 1: UNIVERSAL RULES ENGINE")
    print("="*80)
    
    # Merge features with results
    df_eval = df_features[['opportunity_id', 'is_won']].merge(df_rule_results, on='opportunity_id')
    
    # Get rule-matched opportunities
    rule_matched = df_eval[df_eval['rule_matched']]
    
    if len(rule_matched) == 0:
        print("[Layer 1] No rules matched - cannot validate")
        return None
    
    # For rules, predict "won" if action contains positive keywords
    rule_matched['predicted_won'] = rule_matched['rule_action'].str.contains(
        'Create|Opportunity|Expansion|Renewal|QBR',
        case=False,
        na=False
    ).astype(int)
    
    # Calculate metrics
    precision = precision_score(rule_matched['is_won'], rule_matched['predicted_won'], zero_division=0)
    recall = recall_score(rule_matched['is_won'], rule_matched['predicted_won'], zero_division=0)
    accuracy = accuracy_score(rule_matched['is_won'], rule_matched['predicted_won'])
    f1 = f1_score(rule_matched['is_won'], rule_matched['predicted_won'], zero_division=0)
    coverage = len(rule_matched) / len(df_eval)
    
    # Confusion matrix
    cm = confusion_matrix(rule_matched['is_won'], rule_matched['predicted_won'])
    
    # Per-rule performance
    per_rule_performance = {}
    for rule_id in rule_matched['rule_id'].unique():
        if pd.notna(rule_id):
            rule_data = rule_matched[rule_matched['rule_id'] == rule_id]
            rule_precision = precision_score(
                rule_data['is_won'],
                rule_data['predicted_won'],
                zero_division=0
            )
            per_rule_performance[rule_id] = {
                'precision': float(rule_precision),
                'matches': int(len(rule_data)),
                'wins': int(rule_data['is_won'].sum()),
                'predicted_wins': int(rule_data['predicted_won'].sum())
            }
    
    results = {
        'layer': 'Layer 1: Universal Rules',
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'coverage': float(coverage),
        'n_matches': int(len(rule_matched)),
        'confusion_matrix': cm.tolist(),
        'per_rule_performance': per_rule_performance,
        'meets_target': precision >= 0.90
    }
    
    print(f"\n[Layer 1] Performance:")
    print(f"  Precision: {precision:.1%} {'[PASS]' if precision >= 0.90 else '[FAIL]'} (target: >=90%)")
    print(f"  Recall: {recall:.1%}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Coverage: {coverage:.1%} ({len(rule_matched)}/{len(df_eval)} opportunities)")
    
    print(f"\n[Layer 1] Confusion Matrix:")
    print(f"              Predicted")
    print(f"           Lost    Won")
    print(f"Actual Lost  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Won   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Top performing rules
    sorted_rules = sorted(per_rule_performance.items(), key=lambda x: x[1]['precision'], reverse=True)
    print(f"\n[Layer 1] Top 5 Rules by Precision:")
    for rule_id, metrics in sorted_rules[:5]:
        print(f"  {rule_id}: {metrics['precision']:.1%} ({metrics['matches']} matches, {metrics['wins']} wins)")
    
    return results


def validate_layer2_patterns(df_features, df_pattern_results):
    """
    Validate Layer 2: Pattern Discovery
    Target: ≥70% precision
    """
    print("\n" + "="*80)
    print("VALIDATING LAYER 2: PATTERN DISCOVERY")
    print("="*80)
    
    # Merge features with results
    df_eval = df_features[['opportunity_id', 'is_won']].merge(df_pattern_results, on='opportunity_id')
    
    # Get pattern-matched opportunities
    pattern_matched = df_eval[df_eval['pattern_matched']]
    
    if len(pattern_matched) == 0:
        print("[Layer 2] No patterns matched - cannot validate")
        return None
    
    # Patterns predict "won" (they are discovered from won deals)
    pattern_matched['predicted_won'] = 1
    
    # Calculate metrics
    precision = precision_score(pattern_matched['is_won'], pattern_matched['predicted_won'], zero_division=0)
    recall = recall_score(pattern_matched['is_won'], pattern_matched['predicted_won'], zero_division=0)
    accuracy = accuracy_score(pattern_matched['is_won'], pattern_matched['predicted_won'])
    f1 = f1_score(pattern_matched['is_won'], pattern_matched['predicted_won'], zero_division=0)
    coverage = len(pattern_matched) / len(df_eval)
    
    # Confusion matrix
    cm = confusion_matrix(pattern_matched['is_won'], pattern_matched['predicted_won'])
    
    # Per-pattern performance
    per_pattern_performance = {}
    for pattern_id in pattern_matched['pattern_id'].unique():
        if pd.notna(pattern_id):
            pattern_data = pattern_matched[pattern_matched['pattern_id'] == pattern_id]
            pattern_precision = precision_score(
                pattern_data['is_won'],
                pattern_data['predicted_won'],
                zero_division=0
            )
            per_pattern_performance[pattern_id] = {
                'precision': float(pattern_precision),
                'matches': int(len(pattern_data)),
                'wins': int(pattern_data['is_won'].sum())
            }
    
    results = {
        'layer': 'Layer 2: Pattern Discovery',
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'coverage': float(coverage),
        'n_matches': int(len(pattern_matched)),
        'confusion_matrix': cm.tolist(),
        'per_pattern_performance': per_pattern_performance,
        'meets_target': precision >= 0.70
    }
    
    print(f"\n[Layer 2] Performance:")
    print(f"  Precision: {precision:.1%} {'[PASS]' if precision >= 0.70 else '[FAIL]'} (target: >=70%)")
    print(f"  Recall: {recall:.1%}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Coverage: {coverage:.1%} ({len(pattern_matched)}/{len(df_eval)} opportunities)")
    
    print(f"\n[Layer 2] Confusion Matrix:")
    print(f"              Predicted")
    print(f"           Lost    Won")
    print(f"Actual Lost  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Won   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    return results


def validate_layer3_ml(df_features, df_ml_results):
    """
    Validate Layer 3: ML Ensemble
    Target: ≥60% accuracy (realistic for complex cases)
    """
    print("\n" + "="*80)
    print("VALIDATING LAYER 3: ML ENSEMBLE")
    print("="*80)
    
    # Merge features with results
    df_eval = df_features[['opportunity_id', 'is_won']].merge(df_ml_results, on='opportunity_id')
    
    if len(df_eval) == 0:
        print("[Layer 3] No ML predictions - cannot validate")
        return None
    
    # Calculate metrics
    precision = precision_score(df_eval['is_won'], df_eval['ml_prediction'], zero_division=0)
    recall = recall_score(df_eval['is_won'], df_eval['ml_prediction'], zero_division=0)
    accuracy = accuracy_score(df_eval['is_won'], df_eval['ml_prediction'])
    f1 = f1_score(df_eval['is_won'], df_eval['ml_prediction'], zero_division=0)
    
    try:
        auc = roc_auc_score(df_eval['is_won'], df_eval['ml_probability'])
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(df_eval['is_won'], df_eval['ml_prediction'])
    
    results = {
        'layer': 'Layer 3: ML Ensemble',
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auc': float(auc),
        'n_predictions': int(len(df_eval)),
        'confusion_matrix': cm.tolist(),
        'meets_target': accuracy >= 0.60
    }
    
    print(f"\n[Layer 3] Performance:")
    print(f"  Accuracy: {accuracy:.1%} {'[PASS]' if accuracy >= 0.60 else '[FAIL]'} (target: >=60%)")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  Predictions: {len(df_eval)}")
    
    print(f"\n[Layer 3] Confusion Matrix:")
    print(f"              Predicted")
    print(f"           Lost    Won")
    print(f"Actual Lost  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Won   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    return results


def validate_overall_system(df_features, recommendations):
    """
    Validate overall system performance
    """
    print("\n" + "="*80)
    print("VALIDATING OVERALL SYSTEM")
    print("="*80)
    
    # Create DataFrame from recommendations
    df_recs = pd.DataFrame([
        {
            'opportunity_id': r['opportunity_id'],
            'predicted_outcome': r['predicted_outcome'],
            'confidence': r['confidence'],
            'primary_source': r['primary_source']
        }
        for r in recommendations if 'predicted_outcome' in r
    ])
    
    # Merge with actual outcomes
    df_eval = df_features[['opportunity_id', 'is_won']].merge(df_recs, on='opportunity_id')
    
    if len(df_eval) == 0:
        print("[Overall] No recommendations to validate")
        return None
    
    # Calculate metrics
    precision = precision_score(df_eval['is_won'], df_eval['predicted_outcome'], zero_division=0)
    recall = recall_score(df_eval['is_won'], df_eval['predicted_outcome'], zero_division=0)
    accuracy = accuracy_score(df_eval['is_won'], df_eval['predicted_outcome'])
    f1 = f1_score(df_eval['is_won'], df_eval['predicted_outcome'], zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(df_eval['is_won'], df_eval['predicted_outcome'])
    
    # Source breakdown
    source_breakdown = df_eval['primary_source'].value_counts().to_dict()
    
    # Average confidence
    avg_confidence = df_eval['confidence'].mean()
    
    results = {
        'layer': 'Overall System',
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'avg_confidence': float(avg_confidence),
        'n_predictions': int(len(df_eval)),
        'confusion_matrix': cm.tolist(),
        'source_breakdown': {k: int(v) for k, v in source_breakdown.items()}
    }
    
    print(f"\n[Overall] System Performance:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Avg Confidence: {avg_confidence:.1%}")
    
    print(f"\n[Overall] Confusion Matrix:")
    print(f"              Predicted")
    print(f"           Lost    Won")
    print(f"Actual Lost  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Won   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    print(f"\n[Overall] Source Breakdown:")
    for source, count in source_breakdown.items():
        print(f"  {source}: {count} ({count/len(df_eval):.1%})")
    
    return results


def generate_validation_report(layer1_results, layer2_results, layer3_results, overall_results, output_path='outputs/validation_reports/validation_report.json'):
    """
    Generate comprehensive validation report
    """
    print("\n" + "="*80)
    print("GENERATING VALIDATION REPORT")
    print("="*80)
    
    report = {
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'layer1_rules': layer1_results,
        'layer2_patterns': layer2_results,
        'layer3_ml': layer3_results,
        'overall_system': overall_results,
        'summary': {
            'layer1_meets_target': layer1_results['meets_target'] if layer1_results else False,
            'layer2_meets_target': layer2_results['meets_target'] if layer2_results else False,
            'layer3_meets_target': layer3_results['meets_target'] if layer3_results else False,
            'all_targets_met': (
                (layer1_results['meets_target'] if layer1_results else False) and
                (layer2_results['meets_target'] if layer2_results else False) and
                (layer3_results['meets_target'] if layer3_results else False)
            )
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Validation report saved: {output_path}")
    
    # Print summary
    print(f"\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if layer1_results:
        status = "[PASS]" if layer1_results['meets_target'] else "[FAIL]"
        print(f"\nLayer 1 (Rules): {status}")
        print(f"  Precision: {layer1_results['precision']:.1%} (target: >=90%)")
        print(f"  Coverage: {layer1_results['coverage']:.1%}")
    
    if layer2_results:
        status = "[PASS]" if layer2_results['meets_target'] else "[FAIL]"
        print(f"\nLayer 2 (Patterns): {status}")
        print(f"  Precision: {layer2_results['precision']:.1%} (target: >=70%)")
        print(f"  Coverage: {layer2_results['coverage']:.1%}")
    
    if layer3_results:
        status = "[PASS]" if layer3_results['meets_target'] else "[FAIL]"
        print(f"\nLayer 3 (ML): {status}")
        print(f"  Accuracy: {layer3_results['accuracy']:.1%} (target: >=60%)")
    
    if overall_results:
        print(f"\nOverall System:")
        print(f"  Accuracy: {overall_results['accuracy']:.1%}")
        print(f"  Precision: {overall_results['precision']:.1%}")
        print(f"  F1-Score: {overall_results['f1_score']:.3f}")
    
    if report['summary']['all_targets_met']:
        print(f"\n[SUCCESS] ALL TARGETS MET - System ready for deployment")
    else:
        print(f"\n[FAIL] Some targets not met - Review and refine")
    
    return report

