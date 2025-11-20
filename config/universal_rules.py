"""
Universal Rules Engine - 22 Expert-Defined Patterns
Based on sales expert analysis from Universal_Rules.xlsx
"""

from config.competitor_keywords import detect_competitor_intent, get_competitor_name
from config.action_mappings import get_actions_for_pattern
from config.thresholds import (
    INTENT_HIGH_THRESHOLD, DAYS_LONG_TERM, DAYS_TO_RENEWAL_URGENT,
    DECISION_MAKER_MIN, C_LEVEL_MIN, EMAIL_MEDIUM_THRESHOLD, CALL_MIN_THRESHOLD,
    COMMITTEE_SIZE_MIN
)


def pattern_1_new_logo_decision_maker(opp_features):
    """
    Pattern #1: New Logo + Decision Maker + 3+ Emails + Webinar + Pricing Page
    Action: Create High-Priority Opportunity
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('decision_maker_count', 0) >= DECISION_MAKER_MIN,
        opp_features.get('email_count_during', 0) >= CALL_MIN_THRESHOLD,  # 3+ emails
        opp_features.get('webinar_attended_count', 0) >= 1,
        opp_features.get('pricing_page_visits', 0) >= 1
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_1',
            'rule_name': 'New Logo + Decision Maker + Multi-Touch',
            'confidence': 0.92,
            'signals': {
                'is_new_logo': True,
                'decision_makers': int(opp_features.get('decision_maker_count', 0)),
                'email_threads': int(opp_features.get('email_count_during', 0)),
                'webinar_attended': True,
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0))
            },
            **get_actions_for_pattern('Pattern_1')
        }
    return {'matched': False}


def pattern_2_returning_customer_pricing(opp_features):
    """
    Pattern #2: Returning Customer + 50+ Days + Pricing Page
    Action: Create Expansion Opportunity
    """
    conditions = [
        opp_features.get('was_customer_before', 0) == 1,
        opp_features.get('deal_duration_days', 0) >= DAYS_LONG_TERM,
        opp_features.get('pricing_page_visits', 0) >= 1
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_2',
            'rule_name': 'Returning Customer + Pricing Interest',
            'confidence': 0.88,
            'signals': {
                'was_customer': True,
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0)),
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0))
            },
            **get_actions_for_pattern('Pattern_2')
        }
    return {'matched': False}


def pattern_3_high_intent_competitor(opp_features):
    """
    Pattern #3: High Intent Score + Competitor Research + 50+ Page Views
    Action: Promote Competitive Comparison
    """
    conditions = [
        opp_features.get('max_intent_score_during', 0) >= INTENT_HIGH_THRESHOLD,
        opp_features.get('had_competitor_intent', 0) == 1,
        opp_features.get('total_marketing_events_during', 0) >= DAYS_LONG_TERM
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_3',
            'rule_name': 'High Intent + Competitive Research',
            'confidence': 0.90,
            'signals': {
                'intent_score': float(opp_features.get('max_intent_score_during', 0)),
                'competitor_research': True,
                'page_views': int(opp_features.get('total_marketing_events_during', 0))
            },
            **get_actions_for_pattern('Pattern_3')
        }
    return {'matched': False}


def pattern_4_renewal_c_level_committee(opp_features):
    """
    Pattern #4: Renewal Account + C-Level + Buying Committee 3+ + 30 Days to Renewal
    Action: Create Renewal Opportunity - QBR
    NOTE: This is a POSITIVE renewal signal (engagement + committee), not churn detection
    """
    conditions = [
        opp_features.get('is_renewal', 0) == 1,
        opp_features.get('executive_count', 0) + opp_features.get('vp_count', 0) >= C_LEVEL_MIN,
        opp_features.get('contact_count_total', 0) >= COMMITTEE_SIZE_MIN,
        opp_features.get('days_to_renewal', 999) <= DAYS_TO_RENEWAL_URGENT
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_4',
            'rule_name': 'Renewal + C-Level + Committee',
            'confidence': 0.93,
            'signals': {
                'is_renewal': True,
                'c_level_engaged': int(opp_features.get('executive_count', 0) + opp_features.get('vp_count', 0)),
                'committee_size': int(opp_features.get('contact_count_total', 0)),
                'days_to_renewal': int(opp_features.get('days_to_renewal', 999))
            },
            **get_actions_for_pattern('Pattern_4')
        }
    return {'matched': False}


def pattern_5_new_opp_exec_demo_pricing(opp_features):
    """
    Pattern #5: New Opp + Exec Involvement + Product Demo + Pricing Page
    Action: Exec Sponsor Engagement
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('executive_count', 0) >= 1,
        opp_features.get('demo_request_count', 0) >= 1,
        opp_features.get('pricing_page_visits', 0) >= 1
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_5',
            'rule_name': 'New Opp + Exec + Demo + Pricing',
            'confidence': 0.91,
            'signals': {
                'is_new_logo': True,
                'exec_count': int(opp_features.get('executive_count', 0)),
                'demo_requests': int(opp_features.get('demo_request_count', 0)),
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0))
            },
            **get_actions_for_pattern('Pattern_5')
        }
    return {'matched': False}


def pattern_6_renewal_negative_sentiment(opp_features):
    """
    Pattern #6: Renewal + 30+ Days + Negative Sentiment (declining engagement)
    Action: At-Risk Renewal - Immediate Action
    """
    conditions = [
        opp_features.get('is_renewal', 0) == 1,
        opp_features.get('days_to_renewal', 999) <= DAYS_TO_RENEWAL_URGENT,
        opp_features.get('engagement_trend', '') == 'DECREASING'
    ]

    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_6',
            'rule_name': 'At-Risk Renewal - Negative Sentiment',
            'confidence': 0.95,
            'signals': {
                'is_renewal': True,
                'days_to_renewal': int(opp_features.get('days_to_renewal', 999)),
                'engagement_trend': 'DECREASING'
            },
            **get_actions_for_pattern('Pattern_6')
        }
    return {'matched': False}


def pattern_7_expansion_demo_pricing(opp_features):
    """
    Pattern #7: Expansion Opp + Demo + Pricing Page + 30 Days Contact Age
    Action: Expansion Opportunity - Demo
    """
    conditions = [
        opp_features.get('is_expansion', 0) == 1,
        opp_features.get('demo_request_count', 0) >= 1,
        opp_features.get('pricing_page_visits', 0) >= 1,
        opp_features.get('days_since_first_contact', 999) >= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_7',
            'rule_name': 'Expansion + Demo + Pricing',
            'confidence': 0.87,
            'signals': {
                'is_expansion': True,
                'demo_requests': int(opp_features.get('demo_request_count', 0)),
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0)),
                'contact_age_days': int(opp_features.get('days_since_first_contact', 999))
            },
            **get_actions_for_pattern('Pattern_7')
        }
    return {'matched': False}


def pattern_8_new_opp_stalled(opp_features):
    """
    Pattern #8: New Opp + Stalled + Reminder + 14+ Days
    Action: Re-engage Stalled Opportunity
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('is_stalled', 0) == 1,
        opp_features.get('days_since_last_activity', 0) >= 14
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_8',
            'rule_name': 'Stalled New Opportunity',
            'confidence': 0.85,
            'signals': {
                'is_new_logo': True,
                'is_stalled': True,
                'days_since_activity': int(opp_features.get('days_since_last_activity', 0))
            },
            **get_actions_for_pattern('Pattern_8')
        }
    return {'matched': False}


def pattern_9_new_opp_declining(opp_features):
    """
    Pattern #9: New Opp + Engagement Declining + 30+ Days
    Action: At-Risk Opportunity - Re-engage
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('engagement_trend', '') == 'DECREASING',
        opp_features.get('deal_duration_days', 0) >= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_9',
            'rule_name': 'New Opp - Declining Engagement',
            'confidence': 0.88,
            'signals': {
                'is_new_logo': True,
                'engagement_trend': 'DECREASING',
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_9')
        }
    return {'matched': False}


def pattern_10_new_opp_high_surge(opp_features):
    """
    Pattern #10: New Opp + High Surge + 2+ Emails + 14 Days
    Action: High-Intent New Opportunity
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('high_surge_count_during', 0) >= 1,
        opp_features.get('email_count_during', 0) >= 2,
        opp_features.get('deal_duration_days', 0) <= 14
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_10',
            'rule_name': 'High-Surge New Opportunity',
            'confidence': 0.89,
            'signals': {
                'is_new_logo': True,
                'high_surge_count': int(opp_features.get('high_surge_count_during', 0)),
                'email_count': int(opp_features.get('email_count_during', 0)),
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_10')
        }
    return {'matched': False}


def pattern_11_new_opp_competitor_long(opp_features):
    """
    Pattern #11: New Opp + Competitor Page + 2+ Months
    Action: Competitive Evaluation - Long Cycle
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('had_competitor_intent', 0) == 1,
        opp_features.get('deal_duration_days', 0) >= 60
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_11',
            'rule_name': 'Long Competitive Evaluation',
            'confidence': 0.82,
            'signals': {
                'is_new_logo': True,
                'competitor_research': True,
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_11')
        }
    return {'matched': False}


def pattern_12_expansion_case_study(opp_features):
    """
    Pattern #12: Expansion + Case Study + 30+ Days
    Action: Expansion with Case Study
    """
    conditions = [
        opp_features.get('is_expansion', 0) == 1,
        opp_features.get('case_study_views', 0) >= 1,
        opp_features.get('deal_duration_days', 0) >= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_12',
            'rule_name': 'Expansion + Case Study Interest',
            'confidence': 0.84,
            'signals': {
                'is_expansion': True,
                'case_study_views': int(opp_features.get('case_study_views', 0)),
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_12')
        }
    return {'matched': False}


def pattern_13_renewal_declining_60days(opp_features):
    """
    Pattern #13: Renewal + Declining + 60+ Days
    Action: At-Risk Renewal - Declining Engagement
    """
    conditions = [
        opp_features.get('is_renewal', 0) == 1,
        opp_features.get('engagement_trend', '') == 'DECREASING',
        opp_features.get('days_to_renewal', 999) <= 60
    ]

    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_13',
            'rule_name': 'At-Risk Renewal - Declining',
            'confidence': 0.91,
            'signals': {
                'is_renewal': True,
                'engagement_trend': 'DECREASING',
                'days_to_renewal': int(opp_features.get('days_to_renewal', 999))
            },
            **get_actions_for_pattern('Pattern_13')
        }
    return {'matched': False}


def pattern_14_new_opp_integration(opp_features):
    """
    Pattern #14: New Opp + Integration + 30+ Days
    Action: Integration Interest
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('integration_page_visits', 0) >= 1,
        opp_features.get('deal_duration_days', 0) >= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_14',
            'rule_name': 'New Opp + Integration Interest',
            'confidence': 0.83,
            'signals': {
                'is_new_logo': True,
                'integration_visits': int(opp_features.get('integration_page_visits', 0)),
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_14')
        }
    return {'matched': False}


def pattern_15_renewal_pricing(opp_features):
    """
    Pattern #15: Renewal + Pricing Page + 30+ Days
    Action: Renewal with Pricing Interest
    """
    conditions = [
        opp_features.get('is_renewal', 0) == 1,
        opp_features.get('pricing_page_visits', 0) >= 1,
        opp_features.get('days_to_renewal', 999) <= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_15',
            'rule_name': 'Renewal + Pricing Interest',
            'confidence': 0.90,
            'signals': {
                'is_renewal': True,
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0)),
                'days_to_renewal': int(opp_features.get('days_to_renewal', 999))
            },
            **get_actions_for_pattern('Pattern_15')
        }
    return {'matched': False}


def pattern_16_new_opp_pricing(opp_features):
    """
    Pattern #16: New Opp + Pricing Page + 30+ Days
    Action: Pricing Interest - New Opportunity
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('pricing_page_visits', 0) >= 1,
        opp_features.get('deal_duration_days', 0) >= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_16',
            'rule_name': 'New Opp + Pricing Interest',
            'confidence': 0.86,
            'signals': {
                'is_new_logo': True,
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0)),
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_16')
        }
    return {'matched': False}


def pattern_17_new_opp_surge_14days(opp_features):
    """
    Pattern #17: New Opp + Surge + 14 Days
    Action: High-Surge New Opportunity
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('high_surge_count_during', 0) + opp_features.get('medium_surge_count_during', 0) >= 1,
        opp_features.get('deal_duration_days', 0) <= 14
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_17',
            'rule_name': 'High-Surge Quick Strike',
            'confidence': 0.88,
            'signals': {
                'is_new_logo': True,
                'surge_signals': int(opp_features.get('high_surge_count_during', 0) + opp_features.get('medium_surge_count_during', 0)),
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_17')
        }
    return {'matched': False}


def pattern_18_expansion_competitor(opp_features):
    """
    Pattern #18: Expansion + Competitor + 30+ Days
    Action: Expansion with Competitive Threat
    """
    conditions = [
        opp_features.get('is_expansion', 0) == 1,
        opp_features.get('had_competitor_intent', 0) == 1,
        opp_features.get('deal_duration_days', 0) >= 30
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_18',
            'rule_name': 'Expansion + Competitive Threat',
            'confidence': 0.85,
            'signals': {
                'is_expansion': True,
                'competitor_research': True,
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_18')
        }
    return {'matched': False}


def pattern_19_new_opp_integration_or_pricing(opp_features):
    """
    Pattern #19: New Opp + Integration Page OR Pricing Page + 60+ Days
    Action: Integration or Pricing Interest
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        (opp_features.get('integration_page_visits', 0) >= 1 or opp_features.get('pricing_page_visits', 0) >= 1),
        opp_features.get('deal_duration_days', 0) >= 60
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_19',
            'rule_name': 'Long Cycle + Integration/Pricing',
            'confidence': 0.81,
            'signals': {
                'is_new_logo': True,
                'integration_visits': int(opp_features.get('integration_page_visits', 0)),
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0)),
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_19')
        }
    return {'matched': False}


def pattern_20_renewal_pricing_declining(opp_features):
    """
    Pattern #20: Renewal + Pricing Page + Declining + 30+ Days
    Action: At-Risk Renewal - Pricing and Declining
    """
    conditions = [
        opp_features.get('is_renewal', 0) == 1,
        opp_features.get('pricing_page_visits', 0) >= 1,
        opp_features.get('engagement_trend', '') == 'DECREASING',
        opp_features.get('days_to_renewal', 999) <= DAYS_TO_RENEWAL_URGENT
    ]

    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_20',
            'rule_name': 'At-Risk Renewal - Pricing + Declining',
            'confidence': 0.94,
            'signals': {
                'is_renewal': True,
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0)),
                'engagement_trend': 'DECREASING',
                'days_to_renewal': int(opp_features.get('days_to_renewal', 999))
            },
            **get_actions_for_pattern('Pattern_20')
        }
    return {'matched': False}


def pattern_21_new_opp_pricing_competitive(opp_features):
    """
    Pattern #21: New Opp + Pricing Page + Competitive Mode + 7+ Days
    Action: Competitive Mode - Pricing Interest
    """
    conditions = [
        opp_features.get('is_new_logo', 0) == 1,
        opp_features.get('pricing_page_visits', 0) >= 1,
        opp_features.get('had_competitor_intent', 0) == 1,
        opp_features.get('deal_duration_days', 0) >= 7
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_21',
            'rule_name': 'Competitive + Pricing Interest',
            'confidence': 0.87,
            'signals': {
                'is_new_logo': True,
                'pricing_visits': int(opp_features.get('pricing_page_visits', 0)),
                'competitor_research': True,
                'days_in_cycle': int(opp_features.get('deal_duration_days', 0))
            },
            **get_actions_for_pattern('Pattern_21')
        }
    return {'matched': False}


def pattern_22_renewal_competitive(opp_features):
    """
    Pattern #22: Renewal + Competitive + 7+ Days
    Action: Renewal with Competitive Threat
    """
    conditions = [
        opp_features.get('is_renewal', 0) == 1,
        opp_features.get('had_competitor_intent', 0) == 1,
        opp_features.get('days_to_renewal', 999) <= 7
    ]
    
    if all(conditions):
        return {
            'matched': True,
            'rule_id': 'Pattern_22',
            'rule_name': 'Urgent Renewal + Competitive Threat',
            'confidence': 0.96,
            'signals': {
                'is_renewal': True,
                'competitor_research': True,
                'days_to_renewal': int(opp_features.get('days_to_renewal', 999))
            },
            **get_actions_for_pattern('Pattern_22')
        }
    return {'matched': False}


# Master list of all rule functions
ALL_RULES = [
    pattern_1_new_logo_decision_maker,
    pattern_2_returning_customer_pricing,
    pattern_3_high_intent_competitor,
    pattern_4_renewal_c_level_committee,
    pattern_5_new_opp_exec_demo_pricing,
    pattern_6_renewal_negative_sentiment,
    pattern_7_expansion_demo_pricing,
    pattern_8_new_opp_stalled,
    pattern_9_new_opp_declining,
    pattern_10_new_opp_high_surge,
    pattern_11_new_opp_competitor_long,
    pattern_12_expansion_case_study,
    pattern_13_renewal_declining_60days,
    pattern_14_new_opp_integration,
    pattern_15_renewal_pricing,
    pattern_16_new_opp_pricing,
    pattern_17_new_opp_surge_14days,
    pattern_18_expansion_competitor,
    pattern_19_new_opp_integration_or_pricing,
    pattern_20_renewal_pricing_declining,
    pattern_21_new_opp_pricing_competitive,
    pattern_22_renewal_competitive
]


def check_all_rules(opp_features):
    """
    Check all universal rules against opportunity features
    Returns first matched rule or empty dict if no match
    """
    for rule_func in ALL_RULES:
        result = rule_func(opp_features)
        if result.get('matched', False):
            return result
    
    return {'matched': False, 'rule_id': None}


def get_all_matching_rules(opp_features):
    """
    Get all rules that match (not just first match)
    Useful for analysis and validation
    """
    matched_rules = []
    for rule_func in ALL_RULES:
        result = rule_func(opp_features)
        if result.get('matched', False):
            matched_rules.append(result)
    
    return matched_rules

