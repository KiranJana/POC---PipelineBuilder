"""
Shared Discretization Logic
Used by both training (pattern discovery) and inference (pattern matching)
to ensure 100% consistency
"""

import pandas as pd
from config.thresholds import (
    INTENT_HIGH_THRESHOLD, INTENT_MEDIUM_THRESHOLD,
    ACTIVITY_HIGH_THRESHOLD, ACTIVITY_MEDIUM_THRESHOLD,
    EMAIL_HIGH_THRESHOLD, EMAIL_MEDIUM_THRESHOLD,
    VELOCITY_HIGH_THRESHOLD,
    PAGE_VISIT_HIGH_THRESHOLD
)


def discretize_features(df):
    """
    Discretize continuous features into categorical bins for pattern mining

    CRITICAL: This function must be used identically in both:
    1. Training (Layer 2 pattern discovery)
    2. Inference (Layer 2 pattern matching)

    Any changes here must be synchronized across both paths.

    Args:
        df: DataFrame with continuous features

    Returns:
        DataFrame with discretized binary features (0/1)
    """
    df_discrete = df.copy()

    # ===================================================================
    # INTENT SIGNALS
    # ===================================================================
    df_discrete['Intent_High'] = (df['max_intent_score_during'] >= INTENT_HIGH_THRESHOLD).astype(int)
    df_discrete['Intent_Medium'] = (
        (df['max_intent_score_during'] >= INTENT_MEDIUM_THRESHOLD) &
        (df['max_intent_score_during'] < INTENT_HIGH_THRESHOLD)
    ).astype(int)

    # ===================================================================
    # ACTIVITY LEVELS
    # ===================================================================
    df_discrete['Activity_High'] = (df['total_activity_count_during'] >= ACTIVITY_HIGH_THRESHOLD).astype(int)
    df_discrete['Activity_Medium'] = (
        (df['total_activity_count_during'] >= ACTIVITY_MEDIUM_THRESHOLD) &
        (df['total_activity_count_during'] < ACTIVITY_HIGH_THRESHOLD)
    ).astype(int)

    # ===================================================================
    # EMAIL ENGAGEMENT
    # ===================================================================
    df_discrete['Email_High'] = (df['email_count_during'] >= EMAIL_HIGH_THRESHOLD).astype(int)
    df_discrete['Email_Medium'] = (
        (df['email_count_during'] >= EMAIL_MEDIUM_THRESHOLD) &
        (df['email_count_during'] < EMAIL_HIGH_THRESHOLD)
    ).astype(int)

    # ===================================================================
    # ENGAGEMENT VELOCITY
    # ===================================================================
    df_discrete['Velocity_High'] = (df['activity_velocity'] >= VELOCITY_HIGH_THRESHOLD).astype(int)

    # ===================================================================
    # EXECUTIVE ENGAGEMENT
    # ===================================================================
    df_discrete['C_Level_Engaged'] = (df['has_c_level'] == 1).astype(int)
    df_discrete['Decision_Maker_Engaged'] = (df['has_decision_maker'] == 1).astype(int)
    df_discrete['Exec_Involved'] = (df['has_executive_involvement'] == 1).astype(int)

    # ===================================================================
    # BUYING COMMITTEE
    # ===================================================================
    df_discrete['Multi_Threaded'] = (df['is_multi_threaded'] == 1).astype(int)

    # ===================================================================
    # PRODUCT INTEREST
    # ===================================================================
    df_discrete['Pricing_Interest'] = (df['pricing_page_visits'] >= 1).astype(int)
    df_discrete['Demo_Requested'] = (df['demo_request_count'] >= 1).astype(int)

    # ===================================================================
    # COMPETITIVE SIGNALS
    # ===================================================================
    df_discrete['Competitor_Research'] = (df['had_competitor_intent'] == 1).astype(int)

    # ===================================================================
    # ENGAGEMENT TRENDS
    # ===================================================================
    df_discrete['Engagement_Increasing'] = (df['engagement_trend'] == 'INCREASING').astype(int)
    df_discrete['Engagement_Decreasing'] = (df['engagement_trend'] == 'DECREASING').astype(int)

    # ===================================================================
    # DEAL TYPE
    # ===================================================================
    df_discrete['New_Logo'] = (df['is_new_logo'] == 1).astype(int)
    df_discrete['Expansion'] = (df['is_expansion'] == 1).astype(int)
    df_discrete['Renewal'] = (df['is_renewal'] == 1).astype(int)

    # ===================================================================
    # STALLED DEALS (Negative Signal)
    # ===================================================================
    df_discrete['Is_Stalled'] = (df['is_stalled'] == 1).astype(int)

    # ===================================================================
    # TARGET VARIABLE (if present)
    # ===================================================================
    if 'is_won' in df.columns:
        df_discrete['Won'] = df['is_won']

    return df_discrete


def get_discretized_feature_names():
    """
    Return list of all discretized feature names
    Useful for filtering and selecting features
    """
    return [
        'Intent_High', 'Intent_Medium',
        'Activity_High', 'Activity_Medium',
        'Email_High', 'Email_Medium',
        'Velocity_High',
        'C_Level_Engaged', 'Decision_Maker_Engaged', 'Exec_Involved',
        'Multi_Threaded',
        'Pricing_Interest', 'Demo_Requested',
        'Competitor_Research',
        'Engagement_Increasing', 'Engagement_Decreasing',
        'New_Logo', 'Expansion', 'Renewal',
        'Is_Stalled',
        'Won'  # Target variable
    ]


def check_pattern_match_single_opportunity(opp_features, pattern_conditions):
    """
    Check if a single opportunity matches a pattern

    Args:
        opp_features: Dict or Series with opportunity features
        pattern_conditions: Dict with feature conditions {feature: value}

    Returns:
        bool: True if all conditions match
    """
    # First discretize the opportunity
    df_opp = pd.DataFrame([opp_features])
    df_discrete = discretize_features(df_opp)

    # Check if all pattern conditions match
    for feature, required_value in pattern_conditions.items():
        if feature not in df_discrete.columns:
            return False
        if df_discrete[feature].iloc[0] != required_value:
            return False

    return True
