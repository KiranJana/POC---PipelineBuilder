"""
Layer 2: Pattern Discovery Engine
Discovers hidden patterns using association rule mining
Target: >=70% precision for discovered patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
from mlxtend.frequent_patterns import fpgrowth, association_rules
import json

# Import shared discretization logic for consistency
from utils.discretization import discretize_features, get_discretized_feature_names
from config.thresholds import (
    PATTERN_MIN_SUPPORT, PATTERN_MIN_CONFIDENCE,
    PATTERN_MAX_RULES_STATISTICAL, PATTERN_MAX_RULES_ML,
    PATTERN_SAMPLE_SIZE_THRESHOLD,
    PATTERN_MIN_SINGLE_FEATURE_SUPPORT, PATTERN_MIN_WIN_RATE_SINGLE,
    PATTERN_MIN_LIFT_SINGLE, PATTERN_MAX_PVALUE_SINGLE,
    PATTERN_MIN_COMBO_SUPPORT, PATTERN_MIN_WIN_RATE_COMBO,
    PATTERN_MIN_LIFT_COMBO, PATTERN_MAX_PVALUE_COMBO,
    PATTERN_MIN_SYNERGY_BOOST, PATTERN_MIN_TRIPLE_SUPPORT,
    PATTERN_MIN_WIN_RATE_TRIPLE, PATTERN_MIN_LIFT_TRIPLE,
    PATTERN_MAX_PVALUE_TRIPLE
)


class PatternDiscoveryEngine:
    """
    Layer 2: Pattern Discovery
    Uses association rule mining to discover winning signal combinations
    """

    # Use centralized thresholds (no local definitions needed)
    
    def __init__(self, min_support=None, min_confidence=None, min_lift=1.2):
        """
        Initialize pattern discovery engine

        Args:
            min_support: Minimum support (fraction of deals that must have pattern)
            min_confidence: Minimum confidence (win rate for pattern)
            min_lift: Minimum lift (how much better than baseline)
        """
        # Use config values if not provided
        self.min_support = min_support if min_support is not None else PATTERN_MIN_SUPPORT
        self.min_confidence = min_confidence if min_confidence is not None else PATTERN_MIN_CONFIDENCE
        self.min_lift = min_lift
        self.discovered_patterns = []
        self.pattern_performance = {}
        
        print(f"[Layer 2] Pattern Discovery Engine initialized")
        print(f"  Min Support: {min_support} (pattern must appear in >={min_support*100:.0f}% of deals)")
        print(f"  Min Confidence: {min_confidence} (pattern must predict correctly >={min_confidence*100:.0f}%)")
        print(f"  Min Lift: {min_lift}")

        # Adaptive algorithm selection based on data scale
        self.adaptive_mode = True
        self.dataset_size = None  # Will be set during mining

    def _calculate_lift(self, win_rate, baseline_win_rate):
        """Calculate lift (improvement over baseline win rate)"""
        return win_rate / baseline_win_rate if baseline_win_rate > 0 else 1.0

    def _calculate_fisher_pvalue(self, wins_with_pattern, total_with_pattern, total_wins, total_deals):
        """
        Calculate Fisher's Exact Test p-value for pattern significance

        Args:
            wins_with_pattern: Number of wins with this pattern
            total_with_pattern: Total deals with this pattern
            total_wins: Total wins in dataset
            total_deals: Total deals in dataset

        Returns:
            p-value from Fisher's Exact Test (two-sided)
        """
        # Create contingency table:
        # [[wins_with_pattern, losses_with_pattern],
        #  [wins_without_pattern, losses_without_pattern]]
        losses_with_pattern = total_with_pattern - wins_with_pattern
        wins_without_pattern = total_wins - wins_with_pattern
        losses_without_pattern = (total_deals - total_with_pattern) - wins_without_pattern

        # Ensure non-negative counts (shouldn't happen but safety check)
        contingency_table = [
            [max(0, wins_with_pattern), max(0, losses_with_pattern)],
            [max(0, wins_without_pattern), max(0, losses_without_pattern)]
        ]

        try:
            # Use Fisher's exact test (two-sided)
            odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
            return p_value
        except ValueError:
            # Fallback if contingency table is invalid
            return 1.0
    
    # Removed: discretize_features method - now using shared utils.discretization.discretize_features
    # This ensures 100% consistency between training and inference
    
    def mine_patterns(self, df_features):
        """
        Mine association rules from feature data
        Adaptive: Uses statistical analysis for small datasets, Apriori for large ones
        """
        dataset_size = len(df_features)
        self.dataset_size = dataset_size

        print(f"\n[Layer 2] Mining patterns from {dataset_size} opportunities...")

        # Adaptive algorithm selection
        if self.adaptive_mode:
            if dataset_size < 1000:
                print(f"[Layer 2] Dataset size: {dataset_size} (< 1000) -> Using statistical analysis")
                return self._mine_patterns_statistical(df_features)
            else:
                print(f"[Layer 2] Dataset size: {dataset_size} (>= 1000) -> Using FP-Growth analysis")
                return self._mine_patterns_fpgrowth(df_features)
        else:
            # Legacy behavior - use statistical
            return self._mine_patterns_statistical(df_features)

    def _mine_patterns_statistical(self, df_features):
        """Statistical pattern discovery for smaller datasets - improved version"""
        print(f"[Layer 2] Using statistical pattern discovery (improved algorithm)...")

        # Discretize features using shared function
        df_discrete = discretize_features(df_features)
        discretized_cols = get_discretized_feature_names()
        print(f"[Layer 2] Discretized {len(discretized_cols)} features using shared logic")

        # Select discretized features (exclude continuous and IDs)
        feature_cols = [col for col in df_discrete.columns if col not in
                       ['opportunity_id', 'account_id', 'is_won', 'Won'] and
                       df_discrete[col].dtype in ['int64', 'int32', 'bool']]

        # Separate won and lost deals
        df_won = df_discrete[df_discrete['Won'] == 1]
        df_lost = df_discrete[df_discrete['Won'] == 0]

        print(f"[Layer 2] Won deals: {len(df_won)}, Lost deals: {len(df_lost)}")

        if len(df_won) < 20:  # Higher threshold for quality
            print("[Layer 2] WARNING: Insufficient won deals for pattern mining (< 20)")
            return []

        try:
            import itertools

            patterns = []
            baseline_win_rate = df_discrete['Won'].mean()
            print(f"[Layer 2] Baseline win rate: {baseline_win_rate:.1%}")

            # Strategy: Find meaningful patterns with strict criteria
            # Focus on combinations that significantly outperform individual features

            # 1. Find high-quality individual features first
            strong_features = {}
            for feature in feature_cols:
                feature_present = df_discrete[df_discrete[feature] == 1]
                feature_wins = feature_present['Won'].sum()
                feature_total = len(feature_present)

                if feature_total >= PATTERN_MIN_SINGLE_FEATURE_SUPPORT:
                    win_rate = feature_present['Won'].mean()
                    lift = self._calculate_lift(win_rate, baseline_win_rate)
                    p_value = self._calculate_fisher_pvalue(feature_wins, feature_total, len(df_won), len(df_discrete))

                    # Strict criteria: high win rate, significant lift, statistical significance, reasonable sample size
                    if (win_rate >= PATTERN_MIN_WIN_RATE_SINGLE and lift >= PATTERN_MIN_LIFT_SINGLE and
                        p_value <= PATTERN_MAX_PVALUE_SINGLE and feature_total >= 20):
                        strong_features[feature] = {
                            'win_rate': win_rate,
                            'support': feature_total,
                            'wins': int(feature_wins),
                            'lift': lift,
                            'p_value': p_value
                        }

            print(f"[Layer 2] Found {len(strong_features)} strong individual features")

            # 2. Find feature combinations that create synergy
            feature_list = list(strong_features.keys())
            combination_patterns = []

            # Try pairs first (most interpretable)
            for combo in itertools.combinations(feature_list, 2):
                feature1, feature2 = combo
                both_present = df_discrete[(df_discrete[feature1] == 1) & (df_discrete[feature2] == 1)]

                if len(both_present) >= PATTERN_MIN_COMBO_SUPPORT:
                    combo_win_rate = both_present['Won'].mean()
                    combo_wins = both_present['Won'].sum()
                    combo_lift = self._calculate_lift(combo_win_rate, baseline_win_rate)

                    # Check for synergy: combination should outperform both individual features
                    f1_rate = strong_features[feature1]['win_rate']
                    f2_rate = strong_features[feature2]['win_rate']

                    # Calculate statistical significance
                    combo_pvalue = self._calculate_fisher_pvalue(combo_wins, len(both_present), len(df_won), len(df_discrete))

                    # Synergy criteria: combination wins at least 10% more than better individual feature
                    synergy_boost = combo_win_rate - max(f1_rate, f2_rate)

                    if (combo_win_rate >= PATTERN_MIN_WIN_RATE_COMBO and combo_lift >= PATTERN_MIN_LIFT_COMBO and
                        synergy_boost >= PATTERN_MIN_SYNERGY_BOOST and combo_pvalue <= PATTERN_MAX_PVALUE_COMBO):
                        # Normalize antecedents by sorting alphabetically to avoid duplicates
                        antecedents = sorted([feature1, feature2])
                        pattern_desc = ' + '.join(antecedents)

                        combination_patterns.append({
                            'pattern': pattern_desc,
                            'antecedents': antecedents,
                            'confidence': float(combo_win_rate),
                            'support': int(len(both_present)),
                            'lift': float(combo_lift),
                            'wins': int(combo_wins),
                            'losses': int(len(both_present) - combo_wins),
                            'synergy_boost': synergy_boost,
                            'p_value': combo_pvalue
                        })

            # Try triples for even stronger patterns (but harder to interpret)
            if len(feature_list) >= 3:
                for combo in itertools.combinations(feature_list, 3):
                    feature1, feature2, feature3 = combo
                    all_present = df_discrete[(df_discrete[feature1] == 1) &
                                            (df_discrete[feature2] == 1) &
                                            (df_discrete[feature3] == 1)]

                    if len(all_present) >= PATTERN_MIN_TRIPLE_SUPPORT:
                        triple_win_rate = all_present['Won'].mean()
                        triple_wins = all_present['Won'].sum()
                        triple_lift = self._calculate_lift(triple_win_rate, baseline_win_rate)
                        triple_pvalue = self._calculate_fisher_pvalue(triple_wins, len(all_present), len(df_won), len(df_discrete))

                        if (triple_win_rate >= PATTERN_MIN_WIN_RATE_TRIPLE and triple_lift >= PATTERN_MIN_LIFT_TRIPLE and
                            triple_pvalue <= PATTERN_MAX_PVALUE_TRIPLE):
                            # Normalize antecedents by sorting alphabetically to avoid duplicates
                            antecedents = sorted([feature1, feature2, feature3])
                            pattern_desc = ' + '.join(antecedents)

                            combination_patterns.append({
                                'pattern': pattern_desc,
                                'antecedents': antecedents,
                                'confidence': float(triple_win_rate),
                                'support': int(len(all_present)),
                                'lift': float(triple_lift),
                                'wins': int(triple_wins),
                                'losses': int(len(all_present) - triple_wins),
                                'synergy_boost': 0.15,  # Assume higher synergy for triples
                                'p_value': triple_pvalue
                            })

            # 3. Create final patterns list - prioritize combinations over singles
            final_patterns = []

            # Add combination patterns first (higher value)
            for idx, pattern in enumerate(combination_patterns):
                final_patterns.append({
                    'pattern_id': f'COMBO_{idx:03d}',
                    'pattern': pattern['pattern'],
                    'antecedents': pattern['antecedents'],
                    'confidence': pattern['confidence'],
                    'support': pattern['support'],
                    'lift': pattern['lift'],
                    'wins': pattern['wins'],
                    'losses': pattern['losses']
                })

            # Add top individual features only if we don't have enough combinations
            if len(final_patterns) < 5:
                used_features = set()
                for pattern in final_patterns:
                    used_features.update(pattern['antecedents'])

                remaining_features = [(f, stats) for f, stats in strong_features.items()
                                    if f not in used_features]

                # Sort by win rate and add top remaining features
                remaining_features.sort(key=lambda x: x[1]['win_rate'], reverse=True)

                for idx, (feature, stats) in enumerate(remaining_features[:5]):
                    pattern_id = f'SINGLE_{idx:03d}'
                    final_patterns.append({
                        'pattern_id': pattern_id,
                        'pattern': feature,
                        'antecedents': [feature],
                        'confidence': float(stats['win_rate']),
                        'support': int(stats['support']),
                        'lift': float(stats['lift']),
                        'wins': stats['wins'],
                        'losses': int(stats['support'] - stats['wins'])
                    })

            # Sort by confidence, then support, then lift
            final_patterns = sorted(final_patterns,
                                  key=lambda x: (x['confidence'], x['support'], x['lift']),
                                  reverse=True)

            # Limit to top 10 most valuable patterns
            final_patterns = final_patterns[:10]

            self.discovered_patterns = final_patterns

            print(f"\n[Layer 2] Discovered {len(final_patterns)} high-quality patterns")
            print(f"[Layer 2] Pattern breakdown: {len([p for p in final_patterns if len(p['antecedents']) > 1])} combinations, {len([p for p in final_patterns if len(p['antecedents']) == 1])} singles")

            if final_patterns:
                print(f"[Layer 2] Top patterns by confidence:")
                for i, pattern in enumerate(final_patterns[:5], 1):
                    pattern_type = "COMBO" if len(pattern['antecedents']) > 1 else "SINGLE"
                    print(f"  {i}. [{pattern_type}] {pattern['pattern']}: {pattern['confidence']:.1%} win rate ({pattern['support']} deals, {pattern['lift']:.1f}x lift)")

            return final_patterns

        except Exception as e:
            print(f"[Layer 2] Error in statistical pattern mining: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _mine_patterns_fpgrowth(self, df_features):
        """FP-Growth pattern discovery for larger datasets"""
        print(f"[Layer 2] Using FP-Growth pattern discovery (comprehensive)...")

        # Discretize features using shared function
        df_discrete = discretize_features(df_features)
        print(f"[Layer 2] Discretized features using shared logic")

        # Select discretized features
        feature_cols = [col for col in df_discrete.columns if col not in
                       ['opportunity_id', 'account_id', 'is_won', 'Won'] and
                       df_discrete[col].dtype in ['int64', 'int32', 'bool']]

        # Separate won and lost deals
        df_won = df_discrete[df_discrete['Won'] == 1]

        if len(df_won) < 20:
            print("[Layer 2] WARNING: Insufficient won deals for FP-Growth, switching to statistical")
            return self._mine_patterns_statistical(df_features)

        # Create one-hot encoded DataFrame for FP-Growth (using sparse format for memory efficiency)
        df_encoded = df_won[feature_cols].copy()
        df_encoded = df_encoded.astype(bool)
        # Convert to sparse format to reduce memory usage (only stores True values)
        df_encoded = df_encoded.astype(pd.SparseDtype("bool", False))

        # Pre-filter columns: remove features with support < min_support (they can't contribute to patterns)
        feature_supports = df_encoded.sum() / len(df_encoded)  # Calculate support for each feature
        filtered_cols = feature_supports[feature_supports >= self.min_support].index.tolist()
        df_encoded = df_encoded[filtered_cols]

        print(f"[Layer 2] Pre-filtered {len(feature_cols) - len(filtered_cols)} rare features (< {self.min_support:.1%} support)")
        print(f"[Layer 2] Running FP-Growth on {len(df_encoded)} won deals with {len(filtered_cols)} features...")

        try:
            # Mine frequent itemsets with safety limits (using FP-Growth for better memory efficiency)
            frequent_itemsets = fpgrowth(df_encoded, min_support=self.min_support, use_colnames=True, max_len=3)

            if len(frequent_itemsets) == 0:
                print("[Layer 2] No frequent itemsets found, lowering min_support")
                frequent_itemsets = fpgrowth(df_encoded, min_support=self.min_support/2, use_colnames=True, max_len=3)

            print(f"[Layer 2] Found {len(frequent_itemsets)} frequent itemsets")

            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)

                if len(rules) > 0:
                    # Filter by lift and limit results
                    rules = rules[rules['lift'] >= self.min_lift].head(20)  # Top 20 rules

                    print(f"[Layer 2] Generated {len(rules)} association rules")

                    # Convert rules to pattern format
                    patterns = []
                    for idx, rule in rules.iterrows():
                        antecedents = sorted(list(rule['antecedents']))  # Sort alphabetically to normalize
                        pattern_desc = ' + '.join(antecedents)

                        # Calculate actual performance on full dataset
                        mask = df_discrete[antecedents].all(axis=1)
                        matched_deals = df_discrete[mask]

                        if len(matched_deals) >= 15:  # Higher threshold for FP-Growth patterns
                            win_rate = matched_deals['Won'].mean()
                            wins_count = matched_deals['Won'].sum()
                            p_value = self._calculate_fisher_pvalue(wins_count, len(matched_deals), len(df_won), len(df_discrete))

                            # Filter by statistical significance (same threshold as statistical method)
                            if p_value <= PATTERN_MAX_PVALUE_COMBO:
                                patterns.append({
                                    'pattern_id': f'APR_{idx:03d}',
                                    'pattern': pattern_desc,
                                    'antecedents': antecedents,
                                    'confidence': float(win_rate),
                                    'support': int(len(matched_deals)),
                                    'lift': float(rule['lift']),
                                    'wins': int(wins_count),
                                    'losses': int(len(matched_deals) - wins_count),
                                    'p_value': p_value
                                })

                    # Sort by confidence and return top patterns
                    patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:15]

                    print(f"\n[Layer 2] Discovered {len(patterns)} high-quality patterns using FP-Growth")
                    print(f"[Layer 2] Top 5 patterns by confidence:")
                    for i, pattern in enumerate(patterns[:5], 1):
                        print(f"  {i}. {pattern['pattern']}: {pattern['confidence']:.1%} win rate ({pattern['support']} deals)")

                    return patterns
                else:
                    print("[Layer 2] No association rules met criteria")
                    return []
            else:
                print("[Layer 2] No frequent itemsets found")
                return []

        except Exception as e:
            print(f"[Layer 2] FP-Growth failed ({e}), falling back to statistical analysis")
            return self._mine_patterns_statistical(df_features)

    def check_pattern_match(self, opp_features, pattern):
        """
        Check if an opportunity matches a discovered pattern
        Uses shared discretization logic for 100% consistency with training
        """
        # Convert opportunity to DataFrame and discretize
        df_opp = pd.DataFrame([opp_features])
        df_discrete = discretize_features(df_opp)

        # Check if all antecedents match (all must be = 1)
        for antecedent in pattern['antecedents']:
            if antecedent not in df_discrete.columns:
                return False
            if df_discrete[antecedent].iloc[0] != 1:
                return False

        return True
    
    def apply_patterns(self, opp_features):
        """
        Apply discovered patterns to a single opportunity
        Returns best matching pattern or None
        """
        if not self.discovered_patterns:
            return None
        
        # Find all matching patterns
        matching_patterns = []
        for pattern in self.discovered_patterns:
            if self.check_pattern_match(opp_features, pattern):
                matching_patterns.append(pattern)
        
        # Return highest confidence pattern
        if matching_patterns:
            best_pattern = max(matching_patterns, key=lambda x: x['confidence'])
            return {
                'matched': True,
                'pattern_id': best_pattern['pattern_id'],
                'pattern': best_pattern['pattern'],
                'confidence': best_pattern['confidence'],
                'support': best_pattern['support'],
                'win_rate': best_pattern['confidence'],
                'historical_performance': {
                    'wins': best_pattern['wins'],
                    'losses': best_pattern['losses'],
                    'sample_size': best_pattern['support']
                }
            }
        
        return None
    
    def apply_patterns_batch(self, df_features):
        """
        Apply patterns to a batch of opportunities
        """
        print(f"\n[Layer 2] Applying {len(self.discovered_patterns)} patterns to {len(df_features)} opportunities...")
        
        if not self.discovered_patterns:
            print("[Layer 2] No patterns to apply")
            return pd.DataFrame({
                'opportunity_id': df_features['opportunity_id'],
                'pattern_matched': False,
                'pattern_id': None,
                'pattern_desc': None,
                'pattern_confidence': 0.0,
                'pattern_support': 0
            })
        
        # Initialize results
        pattern_matched = []
        pattern_ids = []
        pattern_descs = []
        pattern_confidences = []
        pattern_supports = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 100
        matched_count = 0
        
        for chunk_start in range(0, len(df_features), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df_features))
            df_chunk = df_features.iloc[chunk_start:chunk_end]
            
            for idx, row in df_chunk.iterrows():
                opp_features = row.to_dict()
                pattern_result = self.apply_patterns(opp_features)
                
                if pattern_result:
                    matched_count += 1
                    pattern_matched.append(True)
                    pattern_ids.append(pattern_result['pattern_id'])
                    pattern_descs.append(pattern_result['pattern'])
                    pattern_confidences.append(pattern_result['confidence'])
                    pattern_supports.append(pattern_result['support'])
                else:
                    pattern_matched.append(False)
                    pattern_ids.append(None)
                    pattern_descs.append(None)
                    pattern_confidences.append(0.0)
                    pattern_supports.append(0)
        
        df_results = pd.DataFrame({
            'opportunity_id': df_features['opportunity_id'].values,
            'pattern_matched': pattern_matched,
            'pattern_id': pattern_ids,
            'pattern_desc': pattern_descs,
            'pattern_confidence': pattern_confidences,
            'pattern_support': pattern_supports
        })
        
        match_rate = matched_count / len(df_features)
        print(f"[Layer 2] Patterns matched: {matched_count}/{len(df_features)} ({match_rate:.1%})")
        
        return df_results
    
    def save_patterns(self, output_path='outputs/discovered_patterns/patterns.json'):
        """Save discovered patterns to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.discovered_patterns, f, indent=2)
        print(f"\n[Layer 2] Saved {len(self.discovered_patterns)} patterns to {output_path}")
    
    def load_patterns(self, input_path='outputs/discovered_patterns/patterns.json'):
        """Load discovered patterns from JSON"""
        with open(input_path, 'r') as f:
            self.discovered_patterns = json.load(f)
        print(f"\n[Layer 2] Loaded {len(self.discovered_patterns)} patterns from {input_path}")


def create_pattern_discovery_engine(min_support=0.02, min_confidence=0.65):
    """Factory function to create pattern discovery engine"""
    return PatternDiscoveryEngine(min_support=min_support, min_confidence=min_confidence)

