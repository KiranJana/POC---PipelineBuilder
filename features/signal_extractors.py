"""Signal extraction from various data sources - LEAKAGE FIXED"""

import pandas as pd
import numpy as np
from config.competitor_keywords import detect_competitor_intent, get_competitor_name


def extract_activity_features(opp, df_activities, evaluation_date=None):
    """Extract activity features for an opportunity up to evaluation_date"""
    opp_id = opp['opportunity_id']
    create_date = opp['create_date']

    # FIX: Use passed evaluation_date or fallback to close_date/now
    if evaluation_date is None:
        evaluation_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    # Filter strictly before evaluation_date
    opp_activities = df_activities[
        (df_activities['opportunity_id'] == opp_id) &
        (df_activities['activity_dt'] >= create_date) &
        (df_activities['activity_dt'] <= evaluation_date)
    ]
    
    # Count by type
    call_count = len(opp_activities[opp_activities['activity_type'] == 'Call'])
    email_count = len(opp_activities[opp_activities['activity_type'] == 'Email'])
    meeting_count = len(opp_activities[opp_activities['activity_type'] == 'Meeting'])
    total_activity = len(opp_activities)
    
    # Campaign engagement
    has_campaign = opp_activities['campaign_id'].notna().sum()
    
    # Activity velocity (activities per day)
    # FIX: Use evaluation_date for duration calculation
    duration = max((evaluation_date - create_date).days, 1)
    activity_velocity = total_activity / duration

    # Days to first activity
    if len(opp_activities) > 0:
        first_activity_date = opp_activities['activity_dt'].min()
        days_to_first_activity = (first_activity_date - create_date).days
    else:
        days_to_first_activity = duration

    # Days since last activity
    if len(opp_activities) > 0:
        last_activity_date = opp_activities['activity_dt'].max()
        days_since_last_activity = (evaluation_date - last_activity_date).days
    else:
        days_since_last_activity = duration
    
    return {
        'call_count_during': call_count,
        'email_count_during': email_count,
        'meeting_count_during': meeting_count,
        'total_activity_count_during': total_activity,
        'campaign_activity_count': has_campaign,
        'activity_velocity': activity_velocity,
        'days_to_first_activity': days_to_first_activity,
        'days_since_last_activity': days_since_last_activity,
    }


def extract_intent_features(opp, df_intent, df_accounts, evaluation_date=None):
    """Extract intent signal features for an opportunity up to evaluation_date"""
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    create_date = opp['create_date']

    if evaluation_date is None:
        evaluation_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    # Get account domain
    account_domain = df_accounts[df_accounts['account_id'] == account_id]['domain'].values
    if len(account_domain) == 0:
        domain = None
    else:
        domain = account_domain[0]

    # Get intent signals DURING opportunity window
    if pd.notna(domain):
        opp_intent = df_intent[
            (df_intent['company_domain'] == domain) &
            (df_intent['signal_dt'] >= create_date) &
            (df_intent['signal_dt'] <= evaluation_date)
        ]
    else:
        opp_intent = pd.DataFrame()
    
    # Intent metrics
    if len(opp_intent) > 0:
        avg_intent_score = opp_intent['intent_score'].mean()
        max_intent_score = opp_intent['intent_score'].max()
        high_surge_count = len(opp_intent[opp_intent['surge_level'] == 'high'])
        medium_surge_count = len(opp_intent[opp_intent['surge_level'] == 'med'])
        unique_topics = opp_intent['keyword_topic'].nunique()
        
        # Check for specific competitive topics
        had_competitor_intent = int(opp_intent['keyword_topic'].apply(detect_competitor_intent).any())
        competitor_name = opp_intent['keyword_topic'].apply(get_competitor_name).dropna().iloc[0] if had_competitor_intent else None
        had_pricing_intent = int(opp_intent['keyword_topic'].str.contains('pricing|cost', case=False, na=False).any())
    else:
        avg_intent_score = 0
        max_intent_score = 0
        high_surge_count = 0
        medium_surge_count = 0
        unique_topics = 0
        had_competitor_intent = 0
        competitor_name = None
        had_pricing_intent = 0
    
    return {
        'avg_intent_score_during': avg_intent_score,
        'max_intent_score_during': max_intent_score,
        'high_surge_count_during': high_surge_count,
        'medium_surge_count_during': medium_surge_count,
        'unique_intent_topics': unique_topics,
        'had_competitor_intent': had_competitor_intent,
        'competitor_name': competitor_name,
        'had_pricing_intent': had_pricing_intent,
    }


def extract_email_features(opp, df_emails, df_accounts, evaluation_date=None):
    """Extract email features up to evaluation_date"""
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    create_date = opp['create_date']

    if evaluation_date is None:
        evaluation_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    # Get account domain
    account_domain = df_accounts[df_accounts['account_id'] == account_id]['domain'].values
    if len(account_domain) == 0:
        domain = None
    else:
        domain = account_domain[0]

    # Get emails DURING window
    if pd.notna(domain):
        opp_emails = df_emails[
            (df_emails['to_domain'] == domain) &
            (df_emails['sent_dt'] >= create_date) &
            (df_emails['sent_dt'] <= evaluation_date)
        ]
    else:
        opp_emails = pd.DataFrame()
    
    # Email metrics
    if len(opp_emails) > 0:
        total_emails = len(opp_emails)
        inbound_emails = len(opp_emails[opp_emails['direction'] == 'inbound'])
        outbound_emails = len(opp_emails[opp_emails['direction'] == 'outbound'])
        exec_involvement = opp_emails['has_exec_in_cc_bool'].sum()
        unique_threads = opp_emails['thread_id'].nunique()
        
        # Email response rate
        if outbound_emails > 0:
            email_response_rate = inbound_emails / outbound_emails
        else:
            email_response_rate = 0
    else:
        total_emails = 0
        inbound_emails = 0
        outbound_emails = 0
        exec_involvement = 0
        unique_threads = 0
        email_response_rate = 0
    
    return {
        'total_email_threads': total_emails,
        'inbound_email_count': inbound_emails,
        'outbound_email_count': outbound_emails,
        'exec_involvement_count': exec_involvement,
        'unique_email_threads': unique_threads,
        'email_response_rate': email_response_rate,
    }


def extract_map_features(opp, df_map, df_accounts, evaluation_date=None):
    """Extract MAP features up to evaluation_date"""
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    create_date = opp['create_date']

    if evaluation_date is None:
        evaluation_date = opp['close_date'] if pd.notna(opp['close_date']) else pd.Timestamp.now()

    # Get account domain
    account_domain = df_accounts[df_accounts['account_id'] == account_id]['domain'].values
    if len(account_domain) == 0:
        domain = None
    else:
        domain = account_domain[0]

    # Get MAP events DURING window
    if pd.notna(domain):
        opp_map = df_map[
            (df_map['visitor_domain'] == domain) &
            (df_map['event_dt'] >= create_date) &
            (df_map['event_dt'] <= evaluation_date)
        ]
    else:
        opp_map = pd.DataFrame()
    
    # MAP metrics
    if len(opp_map) > 0:
        total_marketing_events = len(opp_map)
        total_score_delta = opp_map['score_delta'].sum()
        avg_score_delta = opp_map['score_delta'].mean()
        unique_visitors = opp_map['visitor_email'].nunique()
        
        # Event type breakdown
        webinar_count = len(opp_map[opp_map['event_type'].str.contains('webinar', case=False, na=False)])
        content_download_count = len(opp_map[opp_map['event_type'] == 'content_download'])
        demo_request_count = len(opp_map[opp_map['event_type'] == 'demo_request'])
        
        # Page type breakdown (using our categorization)
        pricing_page_visits = len(opp_map[opp_map['page_type'] == 'pricing_page'])
        integration_page_visits = len(opp_map[opp_map['page_type'] == 'integration_page'])
        case_study_views = len(opp_map[opp_map['page_type'] == 'case_study_page'])
        
        # Total engagement time
        total_duration = opp_map['duration_seconds'].sum()
    else:
        total_marketing_events = 0
        total_score_delta = 0
        avg_score_delta = 0
        unique_visitors = 0
        webinar_count = 0
        content_download_count = 0
        demo_request_count = 0
        pricing_page_visits = 0
        integration_page_visits = 0
        case_study_views = 0
        total_duration = 0
    
    return {
        'total_marketing_events_during': total_marketing_events,
        'total_score_delta_during': total_score_delta,
        'avg_score_delta_during': avg_score_delta,
        'unique_marketing_visitors': unique_visitors,
        'webinar_attended_count': webinar_count,
        'content_download_count': content_download_count,
        'demo_request_count': demo_request_count,
        'pricing_page_visits': pricing_page_visits,
        'integration_page_visits': integration_page_visits,
        'case_study_views': case_study_views,
        'total_engagement_duration': total_duration,
    }


def extract_contact_features(opp, df_contacts):
    """Extract contact and buying committee features for an opportunity"""
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    
    # Get all contacts for this account
    account_contacts = df_contacts[df_contacts['account_id'] == account_id]
    
    # Contact counts
    total_contacts = len(account_contacts)
    
    # Seniority breakdown
    executive_count = len(account_contacts[account_contacts['seniority'] == 'Executive'])
    vp_count = len(account_contacts[account_contacts['seniority'] == 'VP'])
    director_count = len(account_contacts[account_contacts['seniority'] == 'Director'])
    manager_count = len(account_contacts[account_contacts['seniority'] == 'Manager'])
    
    # Role flags
    decision_maker_count = account_contacts['role_flag'].str.contains('Decision_Maker', na=False).sum()
    influencer_count = account_contacts['role_flag'].str.contains('Influencer', na=False).sum()
    champion_count = account_contacts['role_flag'].str.contains('Champion', na=False).sum()
    
    # ZoomInfo enrichment
    zoominfo_exec_count = account_contacts['is_executive_bool'].sum() if 'is_executive_bool' in account_contacts.columns else 0
    
    # Multi-threading score
    multi_threading_score = (
        decision_maker_count * 3 +
        executive_count * 2 +
        vp_count * 1.5 +
        champion_count * 2
    )
    
    return {
        'contact_count_total': total_contacts,
        'executive_count': executive_count,
        'vp_count': vp_count,
        'director_count': director_count,
        'manager_count': manager_count,
        'decision_maker_count': decision_maker_count,
        'influencer_count': influencer_count,
        'champion_count': champion_count,
        'zoominfo_exec_count': int(zoominfo_exec_count),
        'multi_threading_score': multi_threading_score,
    }


def extract_historical_features(opp, df_opp, evaluation_date=None):
    """
    Extract historical customer features (no leakage)

    Args:
        opp: opportunity row
        df_opp: all opportunities dataframe
        evaluation_date: point-in-time for evaluation (defaults to create_date for backward compatibility)
    """
    opp_id = opp['opportunity_id']
    account_id = opp['account_id']
    create_date = opp['create_date']

    # Use evaluation_date if provided, otherwise use create_date
    # This prevents using future information about deal outcomes
    eval_date = evaluation_date if evaluation_date is not None else create_date

    # Get ALL PREVIOUS opportunities for this account (closed BEFORE evaluation date)
    # This ensures we only use information available at the time of evaluation
    previous_opps = df_opp[
        (df_opp['account_id'] == account_id) &
        (df_opp['close_date'] < eval_date) &
        (df_opp['opportunity_id'] != opp_id)
    ]
    
    # Historical metrics
    previous_deal_count = len(previous_opps)
    previous_wins_count = (previous_opps['final_stage'] == 'Closed Won').sum()
    previous_losses_count = (previous_opps['final_stage'] == 'Closed Lost').sum()
    
    if previous_deal_count > 0:
        historical_win_rate = previous_wins_count / previous_deal_count
        total_historical_revenue = previous_opps[previous_opps['final_stage'] == 'Closed Won']['amount'].sum()
        avg_historical_deal_size = previous_opps['amount'].mean()
    else:
        historical_win_rate = 0
        total_historical_revenue = 0
        avg_historical_deal_size = 0
    
    # Was customer before
    was_customer_before = 1 if previous_wins_count > 0 else 0
    
    # Days since last win
    if previous_wins_count > 0:
        last_win_date = previous_opps[previous_opps['final_stage'] == 'Closed Won']['close_date'].max()
        days_since_last_win = (create_date - last_win_date).days
    else:
        days_since_last_win = 9999
    
    # Days since last loss
    if previous_losses_count > 0:
        last_loss_date = previous_opps[previous_opps['final_stage'] == 'Closed Lost']['close_date'].max()
        days_since_last_loss = (create_date - last_loss_date).days
    else:
        days_since_last_loss = 9999
    
    return {
        'was_customer_before': was_customer_before,
        'previous_deal_count': previous_deal_count,
        'previous_wins_count': previous_wins_count,
        'previous_losses_count': previous_losses_count,
        'historical_win_rate': historical_win_rate,
        'total_historical_revenue': total_historical_revenue,
        'avg_historical_deal_size': avg_historical_deal_size,
        'days_since_last_win': days_since_last_win,
        'days_since_last_loss': days_since_last_loss,
    }

