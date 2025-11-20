"""Data loading utilities for multi-source CSV ingestion"""

import pandas as pd
import numpy as np
from datetime import datetime


def validate_data(data_dict):
    """
    Validate loaded data for basic sanity checks
    Since data is clean from sales expert, this focuses on critical issues only
    """
    print("\n" + "="*80)
    print("VALIDATING DATA")
    print("="*80)

    errors = []
    warnings = []

    # 1. Check opportunities
    df_opp = data_dict['opportunities']

    # Required columns
    required_opp_cols = ['opportunity_id', 'account_id', 'create_date', 'final_stage']
    missing_cols = [col for col in required_opp_cols if col not in df_opp.columns]
    if missing_cols:
        errors.append(f"Opportunities missing required columns: {missing_cols}")

    # Unique IDs
    if df_opp['opportunity_id'].duplicated().any():
        dup_count = df_opp['opportunity_id'].duplicated().sum()
        errors.append(f"Opportunities has {dup_count} duplicate opportunity_ids")

    # Amount validation
    if 'amount' in df_opp.columns:
        negative_amounts = (df_opp['amount'] < 0).sum()
        if negative_amounts > 0:
            warnings.append(f"Found {negative_amounts} opportunities with negative amounts")

    # 2. Check accounts
    df_accounts = data_dict['accounts']
    if df_accounts['account_id'].duplicated().any():
        dup_count = df_accounts['account_id'].duplicated().sum()
        errors.append(f"Accounts has {dup_count} duplicate account_ids")

    # 3. Check activities
    df_activities = data_dict['activities']
    required_activity_cols = ['activity_id', 'opportunity_id', 'activity_dt']
    missing_cols = [col for col in required_activity_cols if col not in df_activities.columns]
    if missing_cols:
        errors.append(f"Activities missing required columns: {missing_cols}")

    # 4. Check referential integrity
    opp_account_ids = set(df_opp['account_id'].dropna())
    account_ids = set(df_accounts['account_id'])
    orphaned_opps = opp_account_ids - account_ids
    if orphaned_opps:
        warnings.append(f"Found {len(orphaned_opps)} opportunities with account_ids not in accounts table")

    activity_opp_ids = set(df_activities['opportunity_id'].dropna())
    opp_ids = set(df_opp['opportunity_id'])
    orphaned_activities = activity_opp_ids - opp_ids
    if orphaned_activities:
        warnings.append(f"Found {len(orphaned_activities)} activities with opportunity_ids not in opportunities table")

    # 5. Data quality metrics
    opp_missing_pct = (df_opp.isnull().sum() / len(df_opp) * 100).to_dict()
    high_missing = {col: pct for col, pct in opp_missing_pct.items() if pct > 50}
    if high_missing:
        warnings.append(f"Opportunities columns with >50% missing data: {high_missing}")

    # Print results
    if errors:
        print("\n[CRITICAL ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError(f"Data validation failed with {len(errors)} critical errors")

    if warnings:
        print("\n[WARNINGS]:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n[OK] Data validation passed")
    print(f"   Opportunities: {len(df_opp)} records")
    print(f"   Accounts: {len(df_accounts)} records")
    print(f"   Activities: {len(df_activities)} records")
    print(f"   Missing data in opportunities: {df_opp.isnull().sum().sum()} total nulls")

    return data_dict


def load_all_data():
    """Load all CSV data sources"""
    print("="*80)
    print("LOADING DATA FROM CSV FILES")
    print("="*80)
    
    # Load opportunities (target dataset)
    df_opp = pd.read_csv('input_data/crm_opportunities.csv')
    print(f"[OK] Opportunities: {len(df_opp)} records")
    
    # Load accounts
    df_accounts = pd.read_csv('input_data/crm_accounts.csv')
    print(f"[OK] Accounts: {len(df_accounts)} records")
    
    # Load contacts
    df_contacts = pd.read_csv('input_data/crm_contacts.csv')
    print(f"[OK] Contacts: {len(df_contacts)} records")
    
    # Load activities
    df_activities = pd.read_csv('input_data/crm_activities.csv')
    print(f"[OK] Activities: {len(df_activities)} records")
    
    # Load intent signals
    df_intent = pd.read_csv('input_data/intent_signals.csv')
    print(f"[OK] Intent Signals: {len(df_intent)} records")
    
    # Load email threads
    df_emails = pd.read_csv('input_data/corp_email_threads.csv')
    print(f"[OK] Email Threads: {len(df_emails)} records")
    
    # Load MAP events
    df_map = pd.read_csv('input_data/map_events.csv')
    print(f"[OK] MAP Events: {len(df_map)} records")
    
    # Load ZoomInfo
    df_zoominfo = pd.read_csv('input_data/zoominfo_people.csv')
    print(f"[OK] ZoomInfo People: {len(df_zoominfo)} records")
    
    return {
        'opportunities': df_opp,
        'accounts': df_accounts,
        'contacts': df_contacts,
        'activities': df_activities,
        'intent': df_intent,
        'emails': df_emails,
        'map': df_map,
        'zoominfo': df_zoominfo
    }


def preprocess_dates(data_dict):
    """Parse all date columns"""
    print("\n" + "="*80)
    print("PREPROCESSING DATES")
    print("="*80)
    
    # Parse opportunity dates
    data_dict['opportunities']['create_date'] = pd.to_datetime(data_dict['opportunities']['create_date'])
    data_dict['opportunities']['close_date'] = pd.to_datetime(data_dict['opportunities']['close_date'], errors='coerce')
    
    # Calculate deal duration
    data_dict['opportunities']['deal_duration_days'] = (
        data_dict['opportunities']['close_date'] - data_dict['opportunities']['create_date']
    ).dt.days
    
    # Parse activity dates
    data_dict['activities']['activity_dt'] = pd.to_datetime(data_dict['activities']['activity_dt'])
    
    # Parse intent signal dates
    data_dict['intent']['signal_dt'] = pd.to_datetime(data_dict['intent']['signal_dt'])
    
    # Parse email dates
    data_dict['emails']['sent_dt'] = pd.to_datetime(data_dict['emails']['sent_dt'])
    
    # Parse MAP event dates
    data_dict['map']['event_dt'] = pd.to_datetime(data_dict['map']['event_dt'])
    
    # Parse account renewal dates
    data_dict['accounts']['renewal_date'] = pd.to_datetime(data_dict['accounts']['renewal_date'], errors='coerce')
    
    print("[OK] All dates parsed")
    
    return data_dict


def create_target_variable(df_opp):
    """Create target variable (is_won)"""
    df_opp['is_won'] = (df_opp['final_stage'] == 'Closed Won').astype(int)
    print(f"[OK] Target created: {df_opp['is_won'].sum()} won, {(~df_opp['is_won'].astype(bool)).sum()} lost")
    return df_opp


def extract_page_type(asset_url):
    """Extract page type from MAP event URLs"""
    if pd.isna(asset_url):
        return 'other'
    
    url_lower = str(asset_url).lower()
    
    if 'pricing' in url_lower or '/product' in url_lower:
        return 'pricing_page'
    elif 'integration' in url_lower:
        return 'integration_page'
    elif 'demo' in url_lower:
        return 'demo_page'
    elif 'case-study' in url_lower or 'customer' in url_lower:
        return 'case_study_page'
    elif 'competitor' in url_lower or 'comparison' in url_lower:
        return 'competitive_page'
    else:
        return 'other'


def enrich_map_events(df_map):
    """Add page type categorization to MAP events"""
    df_map['page_type'] = df_map['asset_url'].apply(extract_page_type)
    return df_map


def is_c_level(seniority):
    """Check if contact is C-level"""
    if pd.isna(seniority):
        return False
    c_level_titles = ['Executive', 'VP', 'C-Level', 'Chief', 'President']
    return seniority in c_level_titles


def categorize_seniority(seniority):
    """Categorize seniority into tiers"""
    if pd.isna(seniority):
        return 'Unknown'
    if is_c_level(seniority):
        return 'C-Level'
    elif seniority == 'Director':
        return 'Director'
    elif seniority == 'Manager':
        return 'Manager'
    else:
        return 'IC'


def enrich_contacts(df_contacts, df_zoominfo):
    """Enrich contacts with ZoomInfo data and categorization"""
    # Merge with ZoomInfo
    df_contacts_enriched = df_contacts.merge(
        df_zoominfo[['email', 'is_executive_bool']],
        on='email',
        how='left'
    )
    
    # Add categorization
    df_contacts_enriched['is_c_level'] = df_contacts_enriched['seniority'].apply(is_c_level)
    df_contacts_enriched['seniority_tier'] = df_contacts_enriched['seniority'].apply(categorize_seniority)
    
    return df_contacts_enriched


def load_and_preprocess_all():
    """Main function to load and preprocess all data"""
    # Load data
    data_dict = load_all_data()

    # Validate data (basic sanity checks)
    data_dict = validate_data(data_dict)

    # Preprocess dates
    data_dict = preprocess_dates(data_dict)

    # Create target variable
    data_dict['opportunities'] = create_target_variable(data_dict['opportunities'])

    # Enrich MAP events
    data_dict['map'] = enrich_map_events(data_dict['map'])

    # Enrich contacts
    data_dict['contacts'] = enrich_contacts(data_dict['contacts'], data_dict['zoominfo'])

    print("\n[OK] Data loading and preprocessing complete")

    return data_dict

