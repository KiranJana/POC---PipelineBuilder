"""Similarity search for RAG - find similar historical deals"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class SimilaritySearchCache:
    """
    Cached similarity search to avoid re-normalizing historical data for each opportunity
    Optimizes memory usage for batch processing
    """

    def __init__(self, df_historical, feature_cols, outcome_filter='Won'):
        """
        Initialize with historical data and normalize once

        Args:
            df_historical: DataFrame of historical opportunities
            feature_cols: List of feature columns to use
            outcome_filter: Filter by outcome ('Won', 'Lost', or None for all)
        """
        # Filter by outcome
        if outcome_filter == 'Won':
            self.df_hist = df_historical[df_historical['is_won'] == 1].copy()
        elif outcome_filter == 'Lost':
            self.df_hist = df_historical[df_historical['is_won'] == 0].copy()
        else:
            self.df_hist = df_historical.copy()

        self.feature_cols = feature_cols

        if len(self.df_hist) == 0:
            print(f"[Similarity Cache] No historical deals match outcome filter: {outcome_filter}")
            self.hist_matrix = None
            self.scaler = None
            self.hist_normalized = None
            return

        # Prepare historical matrix once
        self.hist_matrix = self.df_hist[feature_cols].fillna(0).values
        self.scaler = StandardScaler()
        self.hist_normalized = self.scaler.fit_transform(self.hist_matrix)

        print(f"[Similarity Cache] Initialized with {len(self.df_hist)} historical deals")

    def find_similar(self, opp_features, top_k=5):
        """
        Find similar deals for a single opportunity using cached normalized data

        Args:
            opp_features: Features for current opportunity (dict or Series)
            top_k: Number of similar deals to return

        Returns:
            List of similar deals with metadata
        """
        if self.hist_normalized is None:
            return []

        # Extract opportunity vector
        if isinstance(opp_features, dict):
            opp_vector = np.array([opp_features.get(col, 0) for col in self.feature_cols])
        else:
            opp_vector = np.array([opp_features[col] for col in self.feature_cols])

        # Normalize using pre-fitted scaler
        opp_normalized = self.scaler.transform(opp_vector.reshape(1, -1))

        # Compute cosine similarity using pre-normalized historical data
        similarities = cosine_similarity(opp_normalized, self.hist_normalized)[0]

        # Get top K most similar
        top_indices = similarities.argsort()[-min(top_k * 2, len(similarities)):][::-1]

        # Filter out deals with very low similarity (< 0.1)
        valid_indices = [idx for idx in top_indices if similarities[idx] >= 0.1][:top_k]

        if len(valid_indices) == 0:
            valid_indices = top_indices[:top_k]

        # Build similar deals list
        similar_deals = []
        for idx in valid_indices:
            deal = self.df_hist.iloc[idx]

            deal_info = {
                'opportunity_id': deal.get('opportunity_id', f'HIST_{idx}'),
                'account': deal.get('company_name', 'Unknown'),
                'outcome': 'WON' if deal['is_won'] == 1 else 'LOST',
                'amount': float(deal.get('amount', 0)),
                'close_time_days': int(deal.get('deal_duration_days', 0)) if pd.notna(deal.get('deal_duration_days')) else 0,
                'similarity_score': float(similarities[idx]),
                'matching_signals': [],  # Can be computed if needed
                'confidence': 'HIGH' if similarities[idx] > 0.7 else 'MEDIUM' if similarities[idx] > 0.4 else 'LOW'
            }

            similar_deals.append(deal_info)

        similar_deals.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_deals[:top_k]


def find_similar_deals(opp_features, df_historical, feature_cols, top_k=5, outcome_filter='Won'):
    """
    Find most similar historical deals using improved similarity calculation

    Args:
        opp_features: Features for current opportunity (dict or Series)
        df_historical: DataFrame of historical opportunities with features
        feature_cols: List of feature columns to use for similarity
        top_k: Number of similar deals to return
        outcome_filter: Filter by outcome ('Won', 'Lost', or None for all)

    Returns:
        List of similar deals with metadata
    """
    # Filter by outcome if specified
    if outcome_filter == 'Won':
        df_hist = df_historical[df_historical['is_won'] == 1].copy()
    elif outcome_filter == 'Lost':
        df_hist = df_historical[df_historical['is_won'] == 0].copy()
    else:
        df_hist = df_historical.copy()

    if len(df_hist) == 0:
        print(f"[Similarity] No historical deals match outcome filter: {outcome_filter}")
        return []

    # Extract and prepare features
    if isinstance(opp_features, dict):
        opp_vector = pd.Series([opp_features.get(col, 0) for col in feature_cols], index=feature_cols)
    else:
        opp_vector = pd.Series([opp_features[col] for col in feature_cols], index=feature_cols)

    hist_matrix = df_hist[feature_cols].fillna(0)

    # Filter out features that are zero for both current opportunity and most historical deals
    # This prevents similarity from being dominated by irrelevant features
    valid_features = []
    for col in feature_cols:
        opp_val = opp_vector[col]
        hist_vals = hist_matrix[col]

        # Include feature if:
        # 1. Current opportunity has non-zero value, OR
        # 2. At least 10% of historical deals have non-zero values
        if opp_val != 0 or (hist_vals != 0).sum() >= len(hist_matrix) * 0.1:
            valid_features.append(col)

    if len(valid_features) == 0:
        print("[Similarity] No valid features found for similarity calculation")
        return []

    # Use only valid features
    opp_vector_filtered = opp_vector[valid_features]
    hist_matrix_filtered = hist_matrix[valid_features]

    # Normalize using historical data distribution (not just current opportunity)
    scaler = StandardScaler()
    hist_normalized = scaler.fit_transform(hist_matrix_filtered.values)
    opp_normalized = scaler.transform(opp_vector_filtered.values.reshape(1, -1))

    # Compute cosine similarity
    similarities = cosine_similarity(opp_normalized, hist_normalized)[0]

    # Get top K most similar (but filter out near-zero similarities)
    top_indices = similarities.argsort()[-min(top_k * 2, len(similarities)):][::-1]

    # Filter out deals with very low similarity scores (< 0.1)
    valid_indices = [idx for idx in top_indices if similarities[idx] >= 0.1][:top_k]

    if len(valid_indices) == 0:
        # If no good matches, return the best available ones anyway
        valid_indices = top_indices[:top_k]

    similar_deals = []
    for idx in valid_indices:
        deal = df_hist.iloc[idx]

        # Calculate matching signals more intelligently
        matching_signals = []
        for col in valid_features[:15]:  # Check more features for signals
            opp_val = opp_vector[col]
            hist_val = hist_matrix.loc[deal.name, col]

            # Consider a match if both have meaningful values
            if opp_val > 0 and hist_val > 0:
                # For binary/categorical features, exact match
                if opp_val in [0, 1] and hist_val in [0, 1]:
                    if opp_val == hist_val == 1:
                        matching_signals.append(col)
                # For continuous features, relative similarity
                else:
                    if abs(opp_val - hist_val) / max(abs(opp_val), abs(hist_val), 1) < 0.3:
                        matching_signals.append(col)
            # Also consider if both have similar "missing" patterns
            elif opp_val == 0 and hist_val == 0 and opp_val != 0:
                continue  # Skip zero matches unless they indicate something meaningful

        # Create deal metadata
        deal_info = {
            'opportunity_id': deal.get('opportunity_id', f'HIST_{idx}'),
            'account': deal.get('company_name', 'Unknown'),
            'outcome': 'WON' if deal['is_won'] == 1 else 'LOST',
            'amount': float(deal.get('amount', 0)),
            'close_time_days': int(deal.get('deal_duration_days', 0)) if pd.notna(deal.get('deal_duration_days')) else 0,
            'similarity_score': float(similarities[idx]),
            'matching_signals': matching_signals[:5],  # Top 5 matching signals
            'confidence': 'HIGH' if similarities[idx] > 0.7 else 'MEDIUM' if similarities[idx] > 0.4 else 'LOW'
        }

        similar_deals.append(deal_info)

    # Sort by similarity score
    similar_deals.sort(key=lambda x: x['similarity_score'], reverse=True)

    return similar_deals[:top_k]


def get_similar_deal_statistics(similar_deals):
    """
    Calculate aggregate statistics from similar deals
    """
    if not similar_deals:
        return {
            'avg_amount': 0,
            'avg_close_time': 0,
            'win_rate': 0,
            'count': 0
        }
    
    amounts = [d['amount'] for d in similar_deals if d['amount'] > 0]
    close_times = [d['close_time_days'] for d in similar_deals if d['close_time_days'] > 0]
    wins = sum(1 for d in similar_deals if d['outcome'] == 'WON')
    
    return {
        'avg_amount': np.mean(amounts) if amounts else 0,
        'avg_close_time': np.mean(close_times) if close_times else 0,
        'win_rate': wins / len(similar_deals) if similar_deals else 0,
        'count': len(similar_deals)
    }

