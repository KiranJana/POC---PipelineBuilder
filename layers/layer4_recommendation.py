"""
Layer 4: Intelligent Ensemble & Recommendation Engine
Combines all layers and generates actionable recommendations
"""

import pandas as pd
import numpy as np
from utils.similarity_search import find_similar_deals, get_similar_deal_statistics, SimilaritySearchCache
from features.buying_committee import get_persona_recommendations, calculate_persona_risk
from config.action_mappings import PATTERN_ACTIONS
from config.thresholds import (
    RECOMMENDATION_MIN_ML_CONFIDENCE,
    ENSEMBLE_RULE_WEIGHT, ENSEMBLE_PATTERN_WEIGHT, ENSEMBLE_ML_WEIGHT
)


class RecommendationEngine:
    """
    Layer 4: Intelligent Ensemble
    Combines rules, patterns, and ML predictions into unified recommendations
    """
    
    def __init__(self, rules_engine, pattern_engine, ml_ensemble):
        self.rules_engine = rules_engine
        self.pattern_engine = pattern_engine
        self.ml_ensemble = ml_ensemble

        print("[Layer 4] Recommendation Engine initialized")


    def generate_recommendation(self, opp_features, df_historical=None, feature_cols=None, similarity_cache=None):
        """
        Generate comprehensive recommendation for a single opportunity

        Args:
            opp_features: Dictionary of opportunity features
            df_historical: Historical deals for similarity search (optional, legacy)
            feature_cols: Feature columns for similarity search (optional, legacy)
            similarity_cache: Pre-initialized SimilaritySearchCache for batch processing (preferred)

        Returns:
            Comprehensive recommendation dictionary
        """
        # Layer 1: Check rules
        rule_result = self.rules_engine.apply_rules(opp_features)
        
        # Layer 2: Check patterns
        pattern_result = self.pattern_engine.apply_patterns(opp_features)
        
        # ML STRATEGY: Train on ALL data (Layer 3), apply only to messy middle (here)
        # Rationale: Training on all data makes ML robust to unseen patterns in production.
        # But we ONLY apply ML predictions when rules and patterns don't match.
        # This maintains hierarchy: Rules > Patterns > ML

        # Check if this opportunity is handled by rules or patterns
        handled_by_rules = rule_result is not None
        handled_by_patterns = pattern_result is not None

        # Only apply ML to messy middle (not covered by rules or patterns)
        if not handled_by_rules and not handled_by_patterns and self.ml_ensemble.calibrated_model is not None:
            X_single = pd.DataFrame([{col: opp_features.get(col, 0) for col in self.ml_ensemble.feature_cols}])
            ml_proba = self.ml_ensemble.predict_proba(X_single)[0, 1]
            ml_prediction = int(ml_proba > 0.5)
            ml_explanation = self.ml_ensemble.explain_prediction(X_single)
            ml_applied = True
        else:
            ml_proba = 0.5
            ml_prediction = 0
            ml_explanation = []
            ml_applied = False

        # DETERMINE RECOMMENDATION BASED ON ENSEMBLE HIERARCHY (WATERFALL)
        # Priority: Rules > Patterns > ML, but ML provides additional insights

        if handled_by_rules:
            # Use rule-based recommendation
            primary_source = 'RULE'
            confidence = rule_result['confidence']
            recommendation_type = rule_result['action']
            predicted_outcome = 1 if 'Create' in rule_result['action'] or 'Opportunity' in rule_result['action'] else 0
            priority_actions = rule_result.get('priority_actions', [])

        elif handled_by_patterns:
            # Use pattern-based recommendation as primary
            primary_source = 'PATTERN'
            confidence = pattern_result['confidence']
            recommendation_type = 'Create Opportunity'  # Default for patterns
            priority_actions = []
            predicted_outcome = 1  # Patterns predict wins

        elif ml_applied:
            # Use ML prediction (now applied to all opportunities)
            primary_source = 'ML'
            confidence = ml_proba
            recommendation_type = 'Create Opportunity' if ml_prediction == 1 else 'Monitor and Nurture'
            predicted_outcome = ml_prediction
            priority_actions = []

        else:
            # No signals from any layer - monitor only
            primary_source = 'NO_SIGNALS'
            confidence = 0.0
            recommendation_type = 'Insufficient Signals - Monitor Only'
            predicted_outcome = 0
            priority_actions = []

        # Apply confidence threshold for ML recommendations only
        if primary_source == 'ML' and confidence < RECOMMENDATION_MIN_ML_CONFIDENCE:
            # Low confidence ML prediction - downgrade to monitoring
            primary_source = 'LOW_CONFIDENCE'
            recommendation_type = 'Insufficient Signals - Monitor Only'
            predicted_outcome = 0
            priority_actions = [{
                'action': 'Continue monitoring for stronger signals',
                'urgency': 'LOW',
                'impact': 'Monitor',
                'priority': 10
            }]

        # CALCULATE ENSEMBLE SCORE FOR PRIORITIZATION (WEIGHTED COMBINATION)
        # Use weighted ensemble for ranking/prioritization while keeping waterfall for decisions

        # Get confidence scores from each layer (normalized 0-1)
        rule_conf = rule_result['confidence'] if handled_by_rules else 0.0
        pattern_conf = pattern_result['confidence'] if handled_by_patterns else 0.0
        ml_conf = ml_proba if ml_applied else 0.5  # Default to neutral for ML

        # Calculate weighted ensemble score for prioritization
        ensemble_score = (
            ENSEMBLE_RULE_WEIGHT * rule_conf +
            ENSEMBLE_PATTERN_WEIGHT * pattern_conf +
            ENSEMBLE_ML_WEIGHT * ml_conf
        )

        # Normalize to 0-1 scale (since weights sum to 1.0)
        ensemble_score = min(1.0, max(0.0, ensemble_score))

        print(f"[Layer 4] Ensemble prioritization: Rule={rule_conf:.1%}, Pattern={pattern_conf:.1%}, ML={ml_conf:.1%} -> Score={ensemble_score:.1%}")
        
        # Collect all recommended actions
        recommended_actions = []

        # Add primary source actions first (highest priority)
        if primary_source == 'RULE' and rule_result and 'priority_actions' in rule_result:
            recommended_actions.extend(rule_result['priority_actions'])
        elif primary_source == 'PATTERN':
            recommended_actions.extend(priority_actions)
        elif primary_source == 'ML':
            # ML doesn't have specific actions, add generic ones
            pass

        # Add buying committee recommendations
        if 'persona_coverage' in opp_features:
            committee_analysis = {
                'persona_coverage': opp_features.get('persona_coverage', 0),
                'missing_personas': [],  # Would need full analysis
                'has_decision_maker': opp_features.get('has_decision_maker', 0),
                'has_champion': opp_features.get('has_champion', 0)
            }
            persona_actions = get_persona_recommendations(committee_analysis)
            recommended_actions.extend(persona_actions)
        
        # Sort and prioritize actions
        recommended_actions = sorted(recommended_actions, key=lambda x: (
            0 if x['urgency'] == 'CRITICAL' else 1 if x['urgency'] == 'HIGH' else 2,
            x.get('priority', 99)
        ))[:5]  # Top 5 actions
        
        # Find similar deals (optimized with cache or fallback to legacy method)
        similar_deals = []
        if similarity_cache is not None:
            # OPTIMIZED PATH: Use pre-initialized cache (batch processing)
            try:
                similar_deals = similarity_cache.find_similar(opp_features, top_k=5)
            except Exception as e:
                print(f"[Layer 4] Warning: Could not find similar deals (cached): {e}")
        elif df_historical is not None and feature_cols is not None:
            # LEGACY PATH: Create new search for each opportunity (slower)
            try:
                similar_deals = find_similar_deals(
                    opp_features,
                    df_historical,
                    feature_cols,
                    top_k=5,
                    outcome_filter='Won'
                )
            except Exception as e:
                print(f"[Layer 4] Warning: Could not find similar deals: {e}")
        
        # Build comprehensive recommendation
        recommendation = {
            'opportunity_id': opp_features.get('opportunity_id', 'Unknown'),
            'account': opp_features.get('company_name', 'Unknown'),
            'recommendation_type': recommendation_type,
            'predicted_outcome': predicted_outcome,
            'confidence': float(confidence),
            'confidence_calibrated': True,
            'ensemble_score': float(ensemble_score),  # Weighted score for prioritization
            'primary_source': primary_source,
            
            # Layer results
            'matched_rules': [rule_result] if rule_result else [],
            'discovered_patterns': [pattern_result] if pattern_result else [],
            'ml_prediction': {
                'probability': float(ml_proba),
                'prediction': int(ml_prediction),
                'top_features': ml_explanation[:5] if ml_explanation else []
            },
            
            # Actions
            'recommended_actions': recommended_actions,
            
            # Context
            'similar_deals': similar_deals,
            
            # Temporal analysis
            'temporal_analysis': {
                'engagement_trend': opp_features.get('engagement_trend', 'UNKNOWN'),
                'urgency': opp_features.get('urgency', 'MEDIUM'),
                'is_stalled': bool(opp_features.get('is_stalled', 0))
            },
            
            # Buying committee
            'buying_committee_analysis': {
                'persona_coverage': float(opp_features.get('persona_coverage', 0)),
                'has_c_level': bool(opp_features.get('has_c_level', 0)),
                'has_decision_maker': bool(opp_features.get('has_decision_maker', 0)),
                'risk': calculate_persona_risk({
                    'missing_personas': [],  # Would need full analysis
                    'persona_coverage': opp_features.get('persona_coverage', 0)
                })
            },
            
            # Explanation
            'explanation_structured': self._generate_structured_explanation(
                rule_result, pattern_result, ml_proba, similar_deals
            )
        }
        
        return recommendation
    
    def _generate_structured_explanation(self, rule_result, pattern_result, ml_proba, similar_deals):
        """Generate structured explanation"""
        why_confident = []
        key_signals = []
        
        if rule_result:
            why_confident.append(f"Universal Rule {rule_result['rule_id']} matched with {rule_result['confidence']:.0%} confidence")
            for signal, value in rule_result['signals'].items():
                key_signals.append(f"{signal}: {value}")
        
        if pattern_result:
            why_confident.append(f"Discovered pattern shows {pattern_result['confidence']:.0%} win rate ({pattern_result['support']} historical deals)")
            key_signals.append(f"Pattern: {pattern_result['pattern']}")
        
        if ml_proba > 0.5:
            why_confident.append(f"ML model predicts {ml_proba:.0%} probability")
        
        # Similar deals context
        if similar_deals:
            stats = get_similar_deal_statistics(similar_deals)
            historical_context = f"{len(similar_deals)} similar deals won with avg ${stats['avg_amount']:,.0f} in {stats['avg_close_time']:.0f} days"
        else:
            historical_context = "No similar historical deals found"
        
        return {
            'why_confident': why_confident,
            'key_signals': key_signals,
            'historical_context': historical_context
        }

    def _get_intelligent_pattern_recommendation(self, pattern_result):
        """Generate intelligent recommendation based on pattern characteristics"""
        pattern_name = pattern_result['pattern'].lower()

        if 'expansion' in pattern_name:
            return "Pursue Expansion Opportunity"
        elif 'renewal' in pattern_name:
            return "Focus on Renewal Retention"
        elif 'new_logo' in pattern_name:
            return "Accelerate New Logo Acquisition"
        elif 'competitor' in pattern_name:
            return "Address Competitive Threat"
        elif 'executive' in pattern_name or 'c_level' in pattern_name:
            return "Engage Executive Leadership"
        elif 'demo' in pattern_name:
            return "Advance to Demo Stage"
        elif 'pricing' in pattern_name:
            return "Provide Competitive Pricing"
        else:
            return f"Capitalize on {pattern_result['pattern']} Opportunity"

    def _get_default_pattern_actions(self, pattern_name):
        """Get default actions for unmapped patterns"""
        pattern_lower = pattern_name.lower()

        if 'expansion' in pattern_lower:
            return [
                {'action': 'Schedule expansion discovery call', 'priority': 1, 'urgency': 'HIGH', 'impact': '+25% win rate'},
                {'action': 'Prepare expansion value proposition', 'priority': 2, 'urgency': 'MEDIUM', 'impact': '+20% win rate'}
            ]
        elif 'competitor' in pattern_lower:
            return [
                {'action': 'Send competitive differentiation guide', 'priority': 1, 'urgency': 'HIGH', 'impact': '+30% win rate'},
                {'action': 'Schedule competitive comparison demo', 'priority': 2, 'urgency': 'HIGH', 'impact': '+25% win rate'}
            ]
        else:
            return [
                {'action': 'Review pattern signals with sales team', 'priority': 1, 'urgency': 'MEDIUM', 'impact': '+15% win rate'},
                {'action': 'Schedule qualified follow-up', 'priority': 2, 'urgency': 'MEDIUM', 'impact': '+10% win rate'}
            ]

    def generate_recommendations_batch(self, df_features, df_historical=None):
        """
        Generate recommendations for a batch of opportunities
        OPTIMIZED: Uses cached similarity search to avoid O(n²) memory usage
        """
        print(f"\n[Layer 4] Generating recommendations for {len(df_features)} opportunities...")

        # Get feature columns for similarity search
        feature_cols = self.ml_ensemble.feature_cols if self.ml_ensemble.feature_cols else []

        # Initialize similarity search cache ONCE for all opportunities
        similarity_cache = None
        if df_historical is not None and feature_cols:
            similarity_cache = SimilaritySearchCache(df_historical, feature_cols, outcome_filter='Won')
            print(f"[Layer 4] Similarity search cache initialized for batch processing")

        recommendations = []
        for idx, row in df_features.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx+1}/{len(df_features)}...")

            opp_features = row.to_dict()

            try:
                recommendation = self.generate_recommendation(
                    opp_features,
                    df_historical=df_historical,
                    feature_cols=feature_cols,
                    similarity_cache=similarity_cache  # Pass cached searcher
                )
                recommendations.append(recommendation)
            except KeyError as e:
                print(f"[Layer 4] Missing required feature for opportunity {opp_features.get('opportunity_id')}: {e}")
                recommendations.append({
                    'opportunity_id': opp_features.get('opportunity_id', 'Unknown'),
                    'recommendation_type': 'ERROR',
                    'confidence': 0.0,
                    'error': f'Missing feature: {str(e)}',
                    'error_type': 'KeyError'
                })
            except ValueError as e:
                print(f"[Layer 4] Invalid value for opportunity {opp_features.get('opportunity_id')}: {e}")
                recommendations.append({
                    'opportunity_id': opp_features.get('opportunity_id', 'Unknown'),
                    'recommendation_type': 'ERROR',
                    'confidence': 0.0,
                    'error': f'Invalid value: {str(e)}',
                    'error_type': 'ValueError'
                })
            except Exception as e:
                print(f"[Layer 4] Unexpected error processing opportunity {opp_features.get('opportunity_id')}: {e}")
                import traceback
                traceback.print_exc()
                recommendations.append({
                    'opportunity_id': opp_features.get('opportunity_id', 'Unknown'),
                    'recommendation_type': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        print(f"[Layer 4] Generated {len(recommendations)} recommendations")
        
        # Summary statistics
        successful = [r for r in recommendations if r.get('recommendation_type') != 'ERROR']
        if successful:
            avg_confidence = np.mean([r['confidence'] for r in successful])
            source_breakdown = pd.Series([r['primary_source'] for r in successful]).value_counts()
            
            print(f"\n[Layer 4] Recommendation Summary:")
            print(f"  Average confidence: {avg_confidence:.1%}")
            print(f"  Source breakdown:")
            for source, count in source_breakdown.items():
                print(f"    {source}: {count} ({count/len(successful):.1%})")
        
        return recommendations
    
    def save_recommendations(self, recommendations, output_path='outputs/recommendations/recommendations.json'):
        """Save recommendations to JSON"""
        import json
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        print(f"\n[Layer 4] Saved {len(recommendations)} recommendations to {output_path}")


def create_recommendation_engine(rules_engine, pattern_engine, ml_ensemble):
    """Factory function to create recommendation engine"""
    return RecommendationEngine(rules_engine, pattern_engine, ml_ensemble)

