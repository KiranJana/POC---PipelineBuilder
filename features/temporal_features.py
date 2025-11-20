"""Temporal pattern detection - engagement trends, velocity, momentum"""

import pandas as pd
import numpy as np


def count_activities_in_window(activities, days=7, offset=0):
    """Count activities in a time window"""
    if len(activities) == 0:
        return 0
    
    max_date = activities['activity_dt'].max()
    start_date = max_date - pd.Timedelta(days=days+offset)
    end_date = max_date - pd.Timedelta(days=offset)
    
    windowed = activities[
        (activities['activity_dt'] >= start_date) &
        (activities['activity_dt'] <= end_date)
    ]
    
    return len(windowed)


def avg_intent_in_window(intent_signals, days=7, offset=0):
    """Average intent score in a time window"""
    if len(intent_signals) == 0:
        return 0
    
    max_date = intent_signals['signal_dt'].max()
    start_date = max_date - pd.Timedelta(days=days+offset)
    end_date = max_date - pd.Timedelta(days=offset)
    
    windowed = intent_signals[
        (intent_signals['signal_dt'] >= start_date) &
        (intent_signals['signal_dt'] <= end_date)
    ]
    
    if len(windowed) == 0:
        return 0
    
    return windowed['intent_score'].mean()


def calculate_temporal_features(opp, df_activities, df_intent, df_accounts):
    """Calculate temporal features: trends, velocity, momentum"""
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    create_date = opp['create_date']
    close_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()
    
    # Get activities for this opportunity
    opp_activities = df_activities[
        (df_activities['opportunity_id'] == opp_id) &
        (df_activities['activity_dt'] >= create_date) &
        (df_activities['activity_dt'] <= close_date)
    ]
    
    # Get intent signals for this account
    account_domain = df_accounts[df_accounts['account_id'] == account_id]['domain'].values
    if len(account_domain) > 0 and pd.notna(account_domain[0]):
        opp_intent = df_intent[
            (df_intent['company_domain'] == account_domain[0]) &
            (df_intent['signal_dt'] >= create_date) &
            (df_intent['signal_dt'] <= close_date)
        ]
    else:
        opp_intent = pd.DataFrame()
    
    # Engagement velocity (last 7 days vs previous 7 days)
    if len(opp_activities) > 0:
        recent_activity = count_activities_in_window(opp_activities, days=7, offset=0)
        previous_activity = count_activities_in_window(opp_activities, days=7, offset=7)
        
        if previous_activity > 0:
            velocity_change = (recent_activity - previous_activity) / previous_activity
        else:
            velocity_change = 0 if recent_activity == 0 else 1.0
    else:
        velocity_change = 0
        recent_activity = 0
        previous_activity = 0
    
    # Intent momentum (last 7 days vs previous 7 days)
    if len(opp_intent) > 0:
        recent_intent = avg_intent_in_window(opp_intent, days=7, offset=0)
        previous_intent = avg_intent_in_window(opp_intent, days=7, offset=7)
        intent_momentum = recent_intent - previous_intent
    else:
        intent_momentum = 0
        recent_intent = 0
        previous_intent = 0
    
    # Determine engagement trend
    if velocity_change > 0.2:
        engagement_trend = 'INCREASING'
    elif velocity_change < -0.2:
        engagement_trend = 'DECREASING'
    else:
        engagement_trend = 'STABLE'
    
    # Determine urgency
    if velocity_change < -0.3 or intent_momentum < -10:
        urgency = 'HIGH'
    elif velocity_change < -0.1 or intent_momentum < -5:
        urgency = 'MEDIUM'
    else:
        urgency = 'LOW'
    
    # Stalled flag (no activity in last 30 days)
    if len(opp_activities) > 0:
        days_since_last = (close_date - opp_activities['activity_dt'].max()).days
        is_stalled = 1 if days_since_last > 30 else 0
    else:
        is_stalled = 1
        days_since_last = (close_date - create_date).days
    
    return {
        'engagement_trend': engagement_trend,
        'velocity_change': velocity_change,
        'intent_momentum': intent_momentum,
        'urgency': urgency,
        'is_stalled': is_stalled,
        'recent_activity_count': recent_activity,
        'previous_activity_count': previous_activity,
        'recent_intent_score': recent_intent,
        'previous_intent_score': previous_intent,
    }


def calculate_renewal_features(opp, df_accounts):
    """Calculate renewal-specific features"""
    account_id = opp['account_id']
    create_date = opp['create_date']
    
    # Get account renewal date
    account = df_accounts[df_accounts['account_id'] == account_id]
    
    if len(account) > 0:
        renewal_date = account['renewal_date'].values[0]
        is_customer = account['is_customer_bool'].values[0]
        
        if pd.notna(renewal_date):
            renewal_date = pd.to_datetime(renewal_date)
            days_to_renewal = (renewal_date - create_date).days
            is_renewal = 1 if days_to_renewal <= 90 and days_to_renewal >= -30 else 0
        else:
            days_to_renewal = 9999
            is_renewal = 0
    else:
        days_to_renewal = 9999
        is_renewal = 0
        is_customer = False
    
    # Determine opportunity type
    is_new_logo = opp.get('is_new_logo', 'FALSE')
    if is_new_logo == 'TRUE' or is_new_logo == True or is_new_logo == 1:
        opp_type = 'new_logo'
        is_new_logo_flag = 1
    elif is_renewal:
        opp_type = 'renewal'
        is_new_logo_flag = 0
    elif is_customer:
        opp_type = 'expansion'
        is_new_logo_flag = 0
    else:
        opp_type = 'new_logo'
        is_new_logo_flag = 1
    
    return {
        'is_renewal': is_renewal,
        'is_expansion': 1 if opp_type == 'expansion' else 0,
        'is_new_logo': is_new_logo_flag,
        'days_to_renewal': days_to_renewal,
        'opportunity_type': opp_type,
    }


def calculate_days_since_first_contact(opp, df_contacts):
    """Calculate days since first contact was created"""
    account_id = opp['account_id']
    create_date = opp['create_date']
    
    # Get contacts for this account
    account_contacts = df_contacts[df_contacts['account_id'] == account_id]
    
    if len(account_contacts) > 0 and 'last_activity_at' in account_contacts.columns:
        # Find earliest contact activity
        earliest_activity = pd.to_datetime(account_contacts['last_activity_at']).min()
        if pd.notna(earliest_activity):
            days_since_first = (create_date - earliest_activity).days
            return max(0, days_since_first)  # Ensure non-negative
    
    return 0

