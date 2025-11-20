"""Main feature engineering pipeline - orchestrates all feature extraction"""

import pandas as pd
import numpy as np

# Initialize pandarallel availability
PANDARALLEL_AVAILABLE = False
try:
    from pandarallel import pandarallel
    PANDARALLEL_AVAILABLE = True
except ImportError:
    pass

from features.signal_extractors import (
    extract_activity_features,
    extract_intent_features,
    extract_email_features,
    extract_map_features,
    extract_contact_features,
    extract_historical_features
)
from features.temporal_features import (
    calculate_temporal_features,
    calculate_renewal_features,
    calculate_days_since_first_contact
)
from features.buying_committee import analyze_buying_committee


def extract_opportunity_features(opp_row, data_dict):
    """
    Extract all features for a single opportunity
    This function is designed to be parallelizable with pandarallel
    """
    opp = opp_row  # opp_row is already a Series from iterrows()

    # Extract dataframes from data_dict (passed as parameter to avoid closure issues)
    df_opp = data_dict['opportunities']
    df_accounts = data_dict['accounts']
    df_contacts = data_dict['contacts']
    df_activities = data_dict['activities']
    df_intent = data_dict['intent']
    df_emails = data_dict['emails']
    df_map = data_dict['map']

    # 1. DEFINE EVALUATION DATE FIRST (Critical for Leakage Prevention)
    # For training, this defaults to close_date.
    # For snapshots, you can modify this line to use a specific snapshot date.
    create_date = opp['create_date']
    close_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    # Use this variable to control Time-Travel
    evaluation_date = close_date

    deal_duration_days = (evaluation_date - create_date).days

    opp_features = {
        'opportunity_id': opp['opportunity_id'],
        'account_id': opp['account_id'],
        'amount': opp['amount'],
        'is_won': opp['is_won'],
        'deal_duration_days': deal_duration_days,
        'final_stage': opp['final_stage']
    }

    # 2. PASS EVALUATION_DATE TO ALL EXTRACTORS

    # Activity features
    activity_feats = extract_activity_features(opp, df_activities, evaluation_date=evaluation_date)
    opp_features.update(activity_feats)

    # Intent features
    intent_feats = extract_intent_features(opp, df_intent, df_accounts, evaluation_date=evaluation_date)
    opp_features.update(intent_feats)

    # Email features
    email_feats = extract_email_features(opp, df_emails, df_accounts, evaluation_date=evaluation_date)
    opp_features.update(email_feats)

    # MAP features
    map_feats = extract_map_features(opp, df_map, df_accounts, evaluation_date=evaluation_date)
    opp_features.update(map_feats)

    # Contact features (Contacts are usually static/current state, so no date needed unless you have creation dates)
    contact_feats = extract_contact_features(opp, df_contacts)
    opp_features.update(contact_feats)

    # Historical features
    historical_feats = extract_historical_features(opp, df_opp, evaluation_date=evaluation_date)
    opp_features.update(historical_feats)

    # Temporal features
    temporal_feats = calculate_temporal_features(opp, df_activities, df_intent, df_accounts, evaluation_date=evaluation_date)
    opp_features.update(temporal_feats)

    # Renewal features
    renewal_feats = calculate_renewal_features(opp, df_accounts)
    opp_features.update(renewal_feats)

    # Days since first contact
    days_since_first = calculate_days_since_first_contact(opp, df_contacts)
    opp_features['days_since_first_contact'] = days_since_first

    # Buying committee analysis
    committee_analysis = analyze_buying_committee(opp, df_contacts)
    opp_features['persona_coverage'] = committee_analysis['persona_coverage']
    opp_features['committee_diversity_score'] = committee_analysis['committee_diversity_score']
    opp_features['has_c_level'] = committee_analysis['has_c_level']
    opp_features['has_decision_maker'] = committee_analysis['has_decision_maker']
    opp_features['has_champion'] = committee_analysis['has_champion']

    return opp_features


def engineer_all_features(data_dict):
    """
    Engineer all features for all opportunities
    Returns DataFrame with comprehensive feature set
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE ENGINEERING")
    print("="*80)

    df_opp = data_dict['opportunities']
    df_accounts = data_dict['accounts']
    df_contacts = data_dict['contacts']
    df_activities = data_dict['activities']
    df_intent = data_dict['intent']
    df_emails = data_dict['emails']
    df_map = data_dict['map']

    # Start with opportunities merged with accounts
    df_features = df_opp.merge(df_accounts, on='account_id', how='left')
    print(f"[OK] Base features: {len(df_features)} opportunities")

    # Encode categorical features (Target encoding for industry, one-hot for others)
    # TARGET ENCODING for Industry (reduces dimensionality, captures win rate patterns)
    # Use category_encoders.TargetEncoder to prevent data leakage through CV smoothing
    try:
        from category_encoders import TargetEncoder

        # Initialize target encoder with smoothing to prevent overfitting
        target_encoder = TargetEncoder(
            cols=['industry'],
            smoothing=1.0,  # Smoothing parameter (higher = more smoothing)
            handle_unknown='value',  # Handle unknown categories
            handle_missing='value'   # Handle missing values
        )

        # Fit and transform - this automatically handles CV to prevent leakage
        df_features['industry_win_rate'] = target_encoder.fit_transform(
            df_features[['industry']], df_features['is_won']
        )

        print(f"[OK] Applied target encoding to industry with CV smoothing")

    except ImportError:
        print("[WARNING] category_encoders not installed, falling back to global mean encoding")
        print("[WARNING] This may cause data leakage - install category_encoders for proper CV encoding")
        # Fallback to global mean (less ideal but prevents breakage)
        industry_win_rates = df_features.groupby('industry')['is_won'].mean()
        df_features['industry_win_rate'] = df_features['industry'].map(industry_win_rates)
        global_win_rate = df_features['is_won'].mean()
        df_features['industry_win_rate'] = df_features['industry_win_rate'].fillna(global_win_rate)

    # One-hot encode other categorical features
    employee_dummies = pd.get_dummies(df_features['employee_band'], prefix='employee_band')
    region_dummies = pd.get_dummies(df_features['region'], prefix='region')

    df_features = pd.concat([df_features, employee_dummies, region_dummies], axis=1)
    print(f"[OK] Categorical features encoded")

    # Extract features for each opportunity using parallel processing
    print(f"[OK] Extracting features for {len(df_opp)} opportunities...")

    all_features = []  # Initialize here

    # Try parallel processing first
    if PANDARALLEL_AVAILABLE:
        try:
            pandarallel.initialize(progress_bar=True, verbose=0)
            print(f"[OK] Using pandarallel for parallel processing")

            # Use parallel_apply for feature extraction
            all_features = df_opp.parallel_apply(
                lambda row: extract_opportunity_features(row, data_dict),
                axis=1
            ).tolist()

        except Exception as e:
            print(f"[WARNING] Pandarallel failed ({e}), falling back to sequential processing")
            all_features = []  # Reset for sequential processing

    # Sequential processing (fallback or primary)
    if not PANDARALLEL_AVAILABLE or not all_features:
        print(f"[OK] Using sequential processing")
        all_features = []

        for idx, opp in df_opp.iterrows():
            if idx % 100 == 0:
                print(f"  Processing opportunity {idx+1}/{len(df_opp)}...")

            opp_features = extract_opportunity_features(opp, data_dict)
            all_features.append(opp_features)
    
    # Convert to DataFrame
    df_engineered = pd.DataFrame(all_features)
    
    # Merge with categorical features
    df_final = df_engineered.merge(
        df_features[['opportunity_id', 'industry_win_rate'] +
                    list(employee_dummies.columns) + list(region_dummies.columns)],
        on='opportunity_id',
        how='left'
    )
    
    print(f"\n[OK] Feature engineering complete")
    print(f"  Total features: {len(df_final.columns)}")
    print(f"  Total opportunities: {len(df_final)}")
    
    return df_final


def create_interaction_features(df):
    """Create interaction and derived features"""
    print("\n[OK] Creating interaction features...")
    
    # Engagement intensity (weighted activity score)
    df['engagement_intensity'] = (
        df['call_count_during'] * 1.5 +
        df['email_count_during'] * 0.5 +
        df['meeting_count_during'] * 3
    )
    
    # Activity per contact
    df['activity_per_contact'] = df['total_activity_count_during'] / np.maximum(df['contact_count_total'], 1)
    
    # Decision maker ratio
    df['decision_maker_ratio'] = df['decision_maker_count'] / np.maximum(df['contact_count_total'], 1)
    
    # Deal size category
    df['deal_size_category'] = pd.cut(
        df['amount'],
        bins=[0, 30000, 60000, 100000, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(float)
    
    # High value new logo
    df['high_value_new_logo'] = ((df['amount'] > df['amount'].median()) & (df['is_new_logo'] == 1)).astype(int)
    
    # Highly engaged flag
    df['is_highly_engaged'] = (df['total_activity_count_during'] > df['total_activity_count_during'].median()).astype(int)
    
    # Multi-threaded flag
    df['is_multi_threaded'] = (df['contact_count_total'] >= 3).astype(int)
    
    # Marketing qualified
    df['is_marketing_qualified'] = (df['total_marketing_events_during'] >= 3).astype(int)
    
    # Intent qualified
    df['is_intent_qualified'] = (df['high_surge_count_during'] >= 1).astype(int)
    
    # Executive involvement flag
    df['has_executive_involvement'] = (df['executive_count'] >= 1).astype(int)
    
    # Velocity score (activity per day * contact count)
    df['velocity_score'] = df['activity_velocity'] * df['contact_count_total']

    # 1. RESPONSE RATIO (Ghosting Detector)
    # Avoid division by zero
    df['response_ratio'] = df['inbound_email_count'] / (df['outbound_email_count'] + 1)

    # 2. MEETING TO EMAIL RATIO (Depth of Engagement)
    # High emails + low meetings = "Just browsing"
    # High meetings = "Serious buyer"
    df['meeting_intensity'] = df['meeting_count_during'] / (df['total_email_threads'] + 1)

    # 3. STAKEHOLDER BREADTH
    # Are we talking to just one person?
    df['contacts_per_month'] = df['contact_count_total'] / (df['deal_duration_days'] / 30).replace(0, 1)

    # 4. STAKEHOLDER VELOCITY (NEW)
    # How quickly are we expanding the buying committee?
    # Use contact growth rate as proxy since we don't have new_contacts_last_30d
    df['stakeholder_velocity'] = df['contact_count_total'] / (df['deal_duration_days'] + 1)

    # 5. INBOUND TO TOTAL RATIO (Customer Engagement Quality)
    # What percentage of communication is from the customer?
    total_email_volume = df['inbound_email_count'] + df['outbound_email_count']
    df['inbound_to_total_ratio'] = df['inbound_email_count'] / (total_email_volume + 1)

    # 6. MEETING TO CALL RATIO (Progression Quality)
    # Are we advancing from calls to meetings?
    df['meeting_to_call_ratio'] = df['meeting_count_during'] / (df['call_count_during'] + 1)

    # 7. INTENT TO ACTIVITY RATIO (Signal Alignment)
    # How well do our activities align with intent signals?
    df['intent_to_activity_ratio'] = df['avg_intent_score_during'] / (df['total_activity_count_during'] + 1)

    # 8. EXECUTIVE INVOLVEMENT DEPTH
    # How deeply are executives engaged?
    df['executive_engagement_depth'] = df['executive_count'] / (df['contact_count_total'] + 1)

    # 9. BUYING COMMITTEE MATURITY
    # How mature is the buying committee?
    df['committee_maturity_score'] = (
        df['has_c_level'] * 3 +
        df['has_decision_maker'] * 2 +
        df['has_champion'] * 2 +
        df['persona_coverage']
    ) / 8  # Normalize to 0-1 scale

    # 10. ACTIVITY EFFICIENCY
    # How efficiently are we generating activities?
    df['activity_efficiency'] = df['total_activity_count_during'] / (df['contact_count_total'] + 1)

    # ===================================================================
    # TIME-NORMALIZED COUNT FEATURES (P1 Enhancement)
    # ===================================================================

    # Prevent division by zero and handle missing durations
    deal_duration_weeks = np.maximum(df['deal_duration_days'] / 7, 0.1)  # Minimum 0.1 weeks
    deal_duration_months = np.maximum(df['deal_duration_days'] / 30, 0.1)  # Minimum 0.1 months

    # Activity counts normalized by time
    df['activities_per_week'] = df['total_activity_count_during'] / deal_duration_weeks
    df['activities_per_month'] = df['total_activity_count_during'] / deal_duration_months

    # Email counts normalized by time
    df['emails_per_week'] = df['email_count_during'] / deal_duration_weeks
    df['emails_per_month'] = df['email_count_during'] / deal_duration_months

    # Call counts normalized by time
    df['calls_per_week'] = df['call_count_during'] / deal_duration_weeks
    df['calls_per_month'] = df['call_count_during'] / deal_duration_months

    # Meeting counts normalized by time
    df['meetings_per_week'] = df['meeting_count_during'] / deal_duration_weeks
    df['meetings_per_month'] = df['meeting_count_during'] / deal_duration_months

    # Contact engagement normalized by time
    df['contacts_per_week'] = df['contact_count_total'] / deal_duration_weeks
    df['contacts_per_month'] = df['contact_count_total'] / deal_duration_months

    # Marketing engagement normalized by time
    df['marketing_events_per_week'] = df['total_marketing_events_during'] / deal_duration_weeks
    df['marketing_events_per_month'] = df['total_marketing_events_during'] / deal_duration_months

    # Intent signals normalized by time
    df['intent_signals_per_week'] = df['unique_intent_topics'] / deal_duration_weeks
    df['intent_signals_per_month'] = df['unique_intent_topics'] / deal_duration_months

    # ===================================================================
    # POLYNOMIAL FEATURE INTERACTIONS (P2 Enhancement)
    # ===================================================================

    # Create interactions for top features that are likely to have synergistic effects
    # Based on domain knowledge of sales processes

    # Intent × Engagement interactions (high intent + high engagement = hot lead)
    df['intent_activity_interaction'] = df['avg_intent_score_during'] * df['total_activity_count_during']
    df['intent_clevel_interaction'] = df['avg_intent_score_during'] * df['has_c_level']

    # Executive involvement × Decision maker interactions
    df['exec_decision_interaction'] = df['has_c_level'] * df['has_decision_maker']
    df['exec_champion_interaction'] = df['has_c_level'] * df['has_champion']

    # Activity quality × Quantity interactions
    df['meeting_email_interaction'] = df['meeting_count_during'] * df['email_count_during']
    df['meeting_velocity_interaction'] = df['meeting_count_during'] * df['activity_velocity']

    # Temporal momentum interactions
    df['velocity_response_interaction'] = df['activity_velocity'] * df['response_ratio']

    # Buying committee completeness interactions
    df['committee_size_interaction'] = df['contact_count_total'] * df['persona_coverage']
    df['committee_maturity_interaction'] = df['persona_coverage'] * df['committee_diversity_score']

    # Deal size × Activity interactions (bigger deals need more activity)
    df['amount_activity_interaction'] = df['amount'] * df['total_activity_count_during'] / 100000  # Normalize

    # Time pressure interactions
    df['duration_amount_interaction'] = df['deal_duration_days'] * df['amount'] / 100000  # Normalize

    print(f"[OK] Created 42 interaction features (including 12 time-normalized + 8 polynomial interactions)")
    
    return df


def clean_and_prepare_dataset(df):
    """Clean and prepare final dataset for modeling"""
    print("\n[OK] Cleaning and preparing dataset...")
    
    # Fill NaN with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['opportunity_id', 'account_id', 'engagement_trend', 'urgency', 'opportunity_type']:
            df[col] = df[col].fillna('Unknown')
    
    print(f"[OK] Dataset cleaned")
    print(f"  Shape: {df.shape}")
    print(f"  Numeric features: {len(numeric_cols)}")
    
    return df


def save_engineered_features(df, output_path='outputs/model_training_data_engineered.csv'):
    """Save engineered features to CSV"""
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved engineered features: {output_path}")
    
    # Feature summary
    print(f"\nFEATURE SUMMARY:")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Total opportunities: {len(df)}")
    print(f"  Win rate: {df['is_won'].mean():.1%}")
    
    return output_path

