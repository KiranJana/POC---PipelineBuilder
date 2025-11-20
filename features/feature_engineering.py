"""Main feature engineering pipeline - orchestrates all feature extraction"""

import pandas as pd
import numpy as np
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
    
    # One-hot encode categorical features
    industry_dummies = pd.get_dummies(df_features['industry'], prefix='industry')
    employee_dummies = pd.get_dummies(df_features['employee_band'], prefix='employee_band')
    region_dummies = pd.get_dummies(df_features['region'], prefix='region')
    
    df_features = pd.concat([df_features, industry_dummies, employee_dummies, region_dummies], axis=1)
    print(f"[OK] Categorical features encoded")
    
    # Extract features for each opportunity
    all_features = []
    
    for idx, opp in df_opp.iterrows():
        if idx % 100 == 0:
            print(f"  Processing opportunity {idx+1}/{len(df_opp)}...")
        
        opp_features = {
            'opportunity_id': opp['opportunity_id'],
            'account_id': opp['account_id'],
            'amount': opp['amount'],
            'is_won': opp['is_won']
        }
        
        # Activity features
        activity_feats = extract_activity_features(opp, df_activities)
        opp_features.update(activity_feats)
        
        # Intent features
        intent_feats = extract_intent_features(opp, df_intent, df_accounts)
        opp_features.update(intent_feats)
        
        # Email features
        email_feats = extract_email_features(opp, df_emails, df_accounts)
        opp_features.update(email_feats)
        
        # MAP features
        map_feats = extract_map_features(opp, df_map, df_accounts)
        opp_features.update(map_feats)
        
        # Contact features
        contact_feats = extract_contact_features(opp, df_contacts)
        opp_features.update(contact_feats)
        
        # Historical features
        historical_feats = extract_historical_features(opp, df_opp)
        opp_features.update(historical_feats)
        
        # Temporal features
        temporal_feats = calculate_temporal_features(opp, df_activities, df_intent, df_accounts)
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
        
        all_features.append(opp_features)
    
    # Convert to DataFrame
    df_engineered = pd.DataFrame(all_features)
    
    # Merge with categorical features
    df_final = df_engineered.merge(
        df_features[['opportunity_id'] + list(industry_dummies.columns) + 
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
    
    print(f"[OK] Created 11 interaction features")
    
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

