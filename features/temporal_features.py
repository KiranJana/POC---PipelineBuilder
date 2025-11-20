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


def calculate_response_rate_in_window(activities, days=7, offset=0):
    """Calculate response rate (inbound/outbound emails) in a time window"""
    if len(activities) == 0:
        return 0

    max_date = activities['activity_dt'].max()
    start_date = max_date - pd.Timedelta(days=days+offset)
    end_date = max_date - pd.Timedelta(days=offset)

    windowed = activities[
        (activities['activity_dt'] >= start_date) &
        (activities['activity_dt'] <= end_date)
    ]

    if len(windowed) == 0:
        return 0

    # Count emails by direction (assuming direction column exists)
    if 'direction' in windowed.columns:
        inbound = len(windowed[windowed['direction'] == 'Inbound'])
        outbound = len(windowed[windowed['direction'] == 'Outbound'])
        return inbound / (outbound + 1) if outbound > 0 else (1.0 if inbound > 0 else 0.0)
    else:
        return 0


def calculate_meeting_progression_in_window(activities, days=7, offset=0):
    """Calculate meeting-to-call ratio in a time window"""
    if len(activities) == 0:
        return 0

    max_date = activities['activity_dt'].max()
    start_date = max_date - pd.Timedelta(days=days+offset)
    end_date = max_date - pd.Timedelta(days=offset)

    windowed = activities[
        (activities['activity_dt'] >= start_date) &
        (activities['activity_dt'] <= end_date)
    ]

    if len(windowed) == 0:
        return 0

    # Count meetings and calls
    meetings = len(windowed[windowed['activity_type'] == 'Meeting'])
    calls = len(windowed[windowed['activity_type'] == 'Call'])

    return meetings / (calls + 1) if calls > 0 else (1.0 if meetings > 0 else 0.0)


def calculate_quality_velocity_in_window(activities, quality_metric_func, days=7, offset=0):
    """Calculate velocity (change) for any quality metric over time windows"""
    if len(activities) == 0:
        return 0

    current = quality_metric_func(activities, days=days, offset=0)
    previous = quality_metric_func(activities, days=days, offset=days)

    if previous > 0:
        change = (current - previous) / previous
    else:
        change = 1.0 if current > 0 else 0.0

    return change


def calculate_temporal_features(opp, df_activities, df_intent, df_accounts, evaluation_date=None):
    """
    Calculate temporal features: trends, velocity, momentum

    Args:
        opp: opportunity row
        df_activities: activities dataframe
        df_intent: intent signals dataframe
        df_accounts: accounts dataframe
        evaluation_date: point-in-time for evaluation (defaults to close_date for backward compatibility)
    """
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    create_date = opp['create_date']
    close_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    # Use evaluation_date if provided, otherwise use close_date (backward compatibility)
    # This allows point-in-time evaluation without future leakage
    eval_date = evaluation_date if evaluation_date is not None else close_date
    
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
    
    # MULTI-WINDOW VELOCITY (7, 14, 30 days)
    temporal_feats = {}

    for window in [7, 14, 30]:
        if len(opp_activities) > 0:
            current = count_activities_in_window(opp_activities, days=window, offset=0)
            previous = count_activities_in_window(opp_activities, days=window, offset=window)

            if previous > 0:
                change = (current - previous) / previous
            else:
                change = 1.0 if current > 0 else 0.0
        else:
            current = 0
            previous = 0
            change = 0.0

        temporal_feats[f'velocity_change_{window}d'] = change
        temporal_feats[f'recent_activity_{window}d'] = current
        temporal_feats[f'previous_activity_{window}d'] = previous

    # Use 7-day velocity for backward compatibility
    velocity_change = temporal_feats['velocity_change_7d']
    recent_activity = temporal_feats['recent_activity_7d']
    previous_activity = temporal_feats['previous_activity_7d']

    # MULTI-WINDOW QUALITY METRICS (Response Rates, Meeting Progression)
    for window in [7, 14, 30]:
        # Response rate velocity
        response_velocity = calculate_quality_velocity_in_window(
            opp_activities, calculate_response_rate_in_window, days=window
        )
        temporal_feats[f'response_rate_velocity_{window}d'] = response_velocity

        # Meeting progression velocity
        meeting_velocity = calculate_quality_velocity_in_window(
            opp_activities, calculate_meeting_progression_in_window, days=window
        )
        temporal_feats[f'meeting_progression_velocity_{window}d'] = meeting_velocity

        # Current quality metrics (for reference)
        current_response_rate = calculate_response_rate_in_window(opp_activities, days=window, offset=0)
        current_meeting_ratio = calculate_meeting_progression_in_window(opp_activities, days=window, offset=0)

        temporal_feats[f'current_response_rate_{window}d'] = current_response_rate
        temporal_feats[f'current_meeting_ratio_{window}d'] = current_meeting_ratio

    # STALENESS METRICS (Crucial for "At Risk" prediction)
    # Use evaluation date to prevent future leakage
    if len(opp_activities) > 0:
        last_act_date = opp_activities['activity_dt'].max()
        days_since_last_act = (eval_date - last_act_date).days

        # Is the deal "ghosting" us?
        # Check if direction column exists (may not be present in all datasets)
        if 'direction' in opp_activities.columns:
            last_inbound = opp_activities[opp_activities['direction'] == 'Inbound']['activity_dt'].max()
            if pd.notna(last_inbound):
                days_since_inbound = (eval_date - last_inbound).days
            else:
                days_since_inbound = 999
        else:
            # If no direction column, assume all activities are inbound
            days_since_inbound = days_since_last_act
    else:
        days_since_last_act = (eval_date - create_date).days
        days_since_inbound = 999

    temporal_feats['days_since_last_activity'] = days_since_last_act
    temporal_feats['days_since_last_inbound'] = days_since_inbound

    # MULTI-WINDOW INTENT MOMENTUM
    for window in [7, 14, 30]:
        if len(opp_intent) > 0:
            current_intent = avg_intent_in_window(opp_intent, days=window, offset=0)
            previous_intent = avg_intent_in_window(opp_intent, days=window, offset=window)
            intent_momentum = current_intent - previous_intent
        else:
            current_intent = 0
            previous_intent = 0
            intent_momentum = 0

        temporal_feats[f'intent_momentum_{window}d'] = intent_momentum
        temporal_feats[f'current_intent_{window}d'] = current_intent
        temporal_feats[f'previous_intent_{window}d'] = previous_intent

    # Use 7-day intent momentum for backward compatibility
    intent_momentum = temporal_feats['intent_momentum_7d']
    recent_intent = temporal_feats['current_intent_7d']
    previous_intent = temporal_feats['previous_intent_7d']
    
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
        days_since_last = (eval_date - opp_activities['activity_dt'].max()).days
        is_stalled = 1 if days_since_last > 30 else 0
    else:
        is_stalled = 1
        days_since_last = (eval_date - create_date).days
    
    # Add backward compatibility features
    temporal_feats.update({
        'engagement_trend': engagement_trend,
        'velocity_change': velocity_change,
        'intent_momentum': intent_momentum,
        'urgency': urgency,
        'is_stalled': is_stalled,
        'recent_activity_count': recent_activity,
        'previous_activity_count': previous_activity,
        'recent_intent_score': recent_intent,
        'previous_intent_score': previous_intent,
    })

    return temporal_feats


def generate_point_in_time_snapshots(opp, df_activities, df_intent, df_accounts, snapshot_days=[30, 60, 90]):
    """
    Generate point-in-time snapshots for training to prevent temporal leakage.

    For a deal that runs from create_date to close_date, create snapshots at:
    - create_date + 30 days
    - create_date + 60 days
    - create_date + 90 days

    This simulates what the model would have seen at different points during the deal lifecycle.

    Args:
        opp: opportunity row
        df_activities: activities dataframe
        df_intent: intent signals dataframe
        df_accounts: accounts dataframe
        snapshot_days: list of days after create_date to take snapshots

    Returns:
        List of (snapshot_date, features_dict) tuples
    """
    create_date = opp['create_date']
    close_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    snapshots = []

    for days_after_create in snapshot_days:
        snapshot_date = create_date + pd.Timedelta(days=days_after_create)

        # Only create snapshot if it's before the deal closed (we can't predict after close)
        if snapshot_date < close_date:
            # Filter activities up to this snapshot date
            snapshot_activities = df_activities[
                (df_activities['opportunity_id'] == opp['opportunity_id']) &
                (df_activities['activity_dt'] <= snapshot_date)
            ]

            # Filter intent signals up to this snapshot date
            account_domain = df_accounts[df_accounts['account_id'] == opp['account_id']]['domain'].values
            if len(account_domain) > 0 and pd.notna(account_domain[0]):
                snapshot_intent = df_intent[
                    (df_intent['company_domain'] == account_domain[0]) &
                    (df_intent['signal_dt'] <= snapshot_date)
                ]
            else:
                snapshot_intent = pd.DataFrame()

            # Calculate features as they would appear at this snapshot date
            features = calculate_temporal_features(
                opp, snapshot_activities, snapshot_intent, df_accounts,
                evaluation_date=snapshot_date
            )

            # Add snapshot metadata
            features['snapshot_date'] = snapshot_date
            features['days_into_deal'] = days_after_create
            features['is_final_snapshot'] = False  # Will be set to True for the last snapshot

            snapshots.append((snapshot_date, features))

    # Add final snapshot at close date (for comparison)
    if snapshots:  # Only if we have intermediate snapshots
        final_features = calculate_temporal_features(
            opp, df_activities, df_intent, df_accounts,
            evaluation_date=close_date
        )
        final_features['snapshot_date'] = close_date
        final_features['days_into_deal'] = (close_date - create_date).days
        final_features['is_final_snapshot'] = True

        snapshots.append((close_date, final_features))

    return snapshots


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

