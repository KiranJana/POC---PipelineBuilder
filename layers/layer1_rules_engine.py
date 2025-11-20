"""
Layer 1: Universal Rules Engine
Applies 22 expert-defined rules with high confidence (≥90% precision target)
"""

import pandas as pd
import numpy as np
from config.universal_rules import check_all_rules, get_all_matching_rules, ALL_RULES


class UniversalRulesEngine:
    """
    Layer 1: Universal Rules Engine
    Applies expert-defined patterns with explainable outputs
    """
    
    def __init__(self):
        self.rules = ALL_RULES
        self.n_rules = len(self.rules)
        print(f"[Layer 1] Universal Rules Engine initialized with {self.n_rules} rules")
    
    def apply_rules(self, opp_features):
        """
        Apply all rules to a single opportunity
        Returns first matched rule or None
        """
        result = check_all_rules(opp_features)
        return result if result.get('matched', False) else None
    
    def apply_rules_batch(self, df_features):
        """
        Apply rules to a batch of opportunities
        Returns DataFrame with rule results
        """
        print(f"\n[Layer 1] Applying {self.n_rules} universal rules to {len(df_features)} opportunities...")
        
        results = []
        matched_count = 0
        
        for idx, row in df_features.iterrows():
            opp_features = row.to_dict()
            rule_result = self.apply_rules(opp_features)
            
            if rule_result:
                matched_count += 1
                results.append({
                    'opportunity_id': opp_features['opportunity_id'],
                    'rule_matched': True,
                    'rule_id': rule_result['rule_id'],
                    'rule_name': rule_result['rule_name'],
                    'rule_confidence': rule_result['confidence'],
                    'rule_action': rule_result['action'],
                    'rule_signals': rule_result['signals'],
                    'priority_actions': rule_result.get('priority_actions', [])
                })
            else:
                results.append({
                    'opportunity_id': opp_features['opportunity_id'],
                    'rule_matched': False,
                    'rule_id': None,
                    'rule_name': None,
                    'rule_confidence': 0.0,
                    'rule_action': None,
                    'rule_signals': {},
                    'priority_actions': []
                })
        
        df_results = pd.DataFrame(results)
        
        match_rate = matched_count / len(df_features)
        print(f"[Layer 1] Rules matched: {matched_count}/{len(df_features)} ({match_rate:.1%})")
        
        # Show rule distribution
        if matched_count > 0:
            rule_counts = df_results[df_results['rule_matched']]['rule_id'].value_counts()
            print(f"\n[Layer 1] Top 5 most frequently matched rules:")
            for rule_id, count in rule_counts.head(5).items():
                print(f"  {rule_id}: {count} matches")
        
        return df_results
    
    def validate_rules(self, df_features, df_results):
        """
        Validate rule performance
        Calculate precision, recall for each rule
        """
        print(f"\n[Layer 1] Validating rule performance...")
        
        # Merge features with results
        df_eval = df_features[['opportunity_id', 'is_won']].merge(df_results, on='opportunity_id')
        
        # Overall rule performance
        rule_matched = df_eval[df_eval['rule_matched']]
        
        if len(rule_matched) > 0:
            # For rules that predict "Create Opportunity" or similar positive actions
            rule_matched['predicted_won'] = rule_matched['rule_action'].str.contains(
                'Create|Opportunity|Expansion|Renewal', 
                case=False, 
                na=False
            ).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score, accuracy_score
            
            precision = precision_score(rule_matched['is_won'], rule_matched['predicted_won'], zero_division=0)
            recall = recall_score(rule_matched['is_won'], rule_matched['predicted_won'], zero_division=0)
            accuracy = accuracy_score(rule_matched['is_won'], rule_matched['predicted_won'])
            
            print(f"\n[Layer 1] Overall Rule Performance:")
            print(f"  Precision: {precision:.1%}")
            print(f"  Recall: {recall:.1%}")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Coverage: {len(rule_matched)/len(df_eval):.1%}")
            
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
                        'precision': rule_precision,
                        'matches': len(rule_data),
                        'wins': rule_data['is_won'].sum()
                    }
            
            # Show top performing rules
            print(f"\n[Layer 1] Top 5 highest precision rules:")
            sorted_rules = sorted(per_rule_performance.items(), key=lambda x: x[1]['precision'], reverse=True)
            for rule_id, metrics in sorted_rules[:5]:
                print(f"  {rule_id}: {metrics['precision']:.1%} precision ({metrics['matches']} matches, {metrics['wins']} wins)")
            
            return {
                'overall_precision': precision,
                'overall_recall': recall,
                'overall_accuracy': accuracy,
                'coverage': len(rule_matched)/len(df_eval),
                'per_rule_performance': per_rule_performance
            }
        else:
            print("[Layer 1] No rules matched - cannot calculate performance")
            return None
    
    def get_rule_explanation(self, rule_result):
        """
        Generate human-readable explanation for a rule match
        """
        if not rule_result or not rule_result.get('matched', False):
            return "No rule matched"
        
        explanation = f"""
Rule Matched: {rule_result['rule_name']} ({rule_result['rule_id']})
Confidence: {rule_result['confidence']:.0%}
Recommended Action: {rule_result['action']}

Signals Detected:
"""
        for signal, value in rule_result['signals'].items():
            explanation += f"  • {signal}: {value}\n"
        
        if rule_result.get('priority_actions'):
            explanation += "\nPriority Actions:\n"
            for action in rule_result['priority_actions']:
                urgency_flag = "🔥 " if action['urgency'] == 'HIGH' else ""
                explanation += f"  {action['priority']}. {urgency_flag}{action['action']} ({action['impact']})\n"
        
        return explanation


def create_rules_engine():
    """Factory function to create rules engine"""
    return UniversalRulesEngine()

