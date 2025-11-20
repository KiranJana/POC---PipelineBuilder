"""Buying committee analysis - persona coverage, multi-threading, gap analysis"""

import pandas as pd
import numpy as np


def analyze_buying_committee(opp, df_contacts):
    """Analyze buying committee composition and coverage"""
    account_id = opp['account_id']
    
    # Get all contacts for this account
    account_contacts = df_contacts[df_contacts['account_id'] == account_id]
    
    if len(account_contacts) == 0:
        return {
            'persona_coverage': 0.0,
            'multi_threading_score': 0.0,
            'missing_personas': ['C-Level', 'Director', 'Manager'],
            'personas_engaged': [],
            'has_c_level': 0,
            'has_director': 0,
            'has_manager': 0,
            'has_decision_maker': 0,
            'has_champion': 0,
            'committee_diversity_score': 0.0,
        }
    
    # Categorize contacts by persona
    personas = {
        'C-Level': account_contacts[account_contacts['seniority_tier'] == 'C-Level'],
        'Director': account_contacts[account_contacts['seniority'] == 'Director'],
        'Manager': account_contacts[account_contacts['seniority'] == 'Manager'],
        'IC': account_contacts[account_contacts['seniority_tier'] == 'IC']
    }
    
    # Persona coverage (ideal: C-Level + Director + Manager)
    ideal_personas = ['C-Level', 'Director', 'Manager']
    personas_present = [p for p in ideal_personas if len(personas[p]) > 0]
    coverage = len(personas_present) / len(ideal_personas)
    
    # Multi-threading score (weighted by seniority)
    weights = {'C-Level': 5, 'Director': 3, 'Manager': 2, 'IC': 1}
    mt_score = sum(len(personas[p]) * weights[p] for p in personas)
    
    # Missing personas
    missing = [p for p in ideal_personas if len(personas[p]) == 0]
    
    # Role flags
    has_decision_maker = int(account_contacts['role_flag'].str.contains('Decision_Maker', na=False).any())
    has_champion = int(account_contacts['role_flag'].str.contains('Champion', na=False).any())
    has_influencer = int(account_contacts['role_flag'].str.contains('Influencer', na=False).any())
    
    # Diversity score (how many different seniority levels)
    unique_seniorities = account_contacts['seniority_tier'].nunique()
    diversity_score = unique_seniorities / 4.0  # Max 4 levels (C-Level, Director, Manager, IC)
    
    return {
        'persona_coverage': coverage,
        'multi_threading_score': mt_score,
        'missing_personas': missing,
        'personas_engaged': personas_present,
        'has_c_level': 1 if len(personas['C-Level']) > 0 else 0,
        'has_director': 1 if len(personas['Director']) > 0 else 0,
        'has_manager': 1 if len(personas['Manager']) > 0 else 0,
        'has_decision_maker': has_decision_maker,
        'has_champion': has_champion,
        'has_influencer': has_influencer,
        'committee_diversity_score': diversity_score,
    }


def calculate_persona_risk(buying_committee_analysis):
    """Calculate risk based on missing personas"""
    missing = buying_committee_analysis['missing_personas']
    
    if 'C-Level' in missing and 'Director' in missing:
        return 'HIGH'
    elif 'C-Level' in missing or len(missing) >= 2:
        return 'MEDIUM'
    else:
        return 'LOW'


def get_persona_recommendations(buying_committee_analysis):
    """Get diverse recommendations for engaging missing personas"""
    missing = buying_committee_analysis['missing_personas']
    recommendations = []

    # C-Level engagement strategies
    if 'C-Level' in missing:
        c_level_actions = [
            {
                'action': 'Schedule executive briefing session with CEO/CTO/CFO',
                'priority': 1,
                'urgency': 'HIGH',
                'reason': 'Executive sponsorship critical for deal advancement',
                'impact': '+30% win rate'
            },
            {
                'action': 'Send executive-level value proposition and ROI analysis',
                'priority': 2,
                'urgency': 'HIGH',
                'reason': 'Executives need business justification and strategic value',
                'impact': '+25% win rate'
            },
            {
                'action': 'Arrange peer executive reference call',
                'priority': 3,
                'urgency': 'MEDIUM',
                'reason': 'Social proof at executive level builds credibility',
                'impact': '+20% win rate'
            }
        ]
        # Add one C-level action (rotate through options for diversity)
        import random
        recommendations.append(c_level_actions[0])  # Always start with briefing

    # Director-level engagement strategies
    if 'Director' in missing:
        director_actions = [
            {
                'action': 'Connect with department heads and directors',
                'priority': 2,
                'urgency': 'HIGH',
                'reason': 'Directors control budgets and drive implementation',
                'impact': '+25% win rate'
            },
            {
                'action': 'Schedule technical deep-dive with department leadership',
                'priority': 3,
                'urgency': 'MEDIUM',
                'reason': 'Directors need technical details and implementation plans',
                'impact': '+20% win rate'
            }
        ]
        recommendations.extend(director_actions[:1])  # Add one director action

    # Decision maker identification strategies
    if not buying_committee_analysis['has_decision_maker']:
        decision_maker_actions = [
            {
                'action': 'Map organizational structure and identify budget holders',
                'priority': 1,
                'urgency': 'CRITICAL',
                'reason': 'Cannot close without identifying who controls the budget',
                'impact': '+35% win rate'
            },
            {
                'action': 'Conduct stakeholder analysis workshop',
                'priority': 2,
                'urgency': 'HIGH',
                'reason': 'Understand decision-making process and key influencers',
                'impact': '+30% win rate'
            },
            {
                'action': 'Leverage LinkedIn and external research to identify decision makers',
                'priority': 3,
                'urgency': 'HIGH',
                'reason': 'Use available tools to build contact database',
                'impact': '+25% win rate'
            }
        ]
        recommendations.append(decision_maker_actions[0])  # Always add decision maker identification

    # Champion development strategies
    if not buying_committee_analysis['has_champion']:
        champion_actions = [
            {
                'action': 'Identify and nurture internal advocate or power user',
                'priority': 2,
                'urgency': 'HIGH',
                'reason': 'Internal champions drive adoption and defend against competitors',
                'impact': '+28% win rate'
            },
            {
                'action': 'Create champion enablement program with exclusive access',
                'priority': 3,
                'urgency': 'MEDIUM',
                'reason': 'Champions need tools and status to effectively advocate',
                'impact': '+25% win rate'
            },
            {
                'action': 'Schedule regular check-ins with potential champions',
                'priority': 4,
                'urgency': 'MEDIUM',
                'reason': 'Build relationship and gather feedback from advocates',
                'impact': '+20% win rate'
            }
        ]
        recommendations.append(champion_actions[0])  # Add champion development

    # Additional recommendations based on committee composition
    coverage = buying_committee_analysis['persona_coverage']

    if coverage < 0.5:  # Poor coverage
        recommendations.append({
            'action': 'Expand buying committee through targeted outreach campaign',
            'priority': 4,
            'urgency': 'HIGH',
            'reason': 'Limited committee coverage increases risk and slows decision making',
            'impact': '+15% win rate'
        })

    # Diversity recommendations
    diversity = buying_committee_analysis.get('committee_diversity_score', 0)
    if diversity < 0.5:  # Low diversity
        recommendations.append({
            'action': 'Engage cross-functional stakeholders (IT, Finance, Operations)',
            'priority': 3,
            'urgency': 'MEDIUM',
            'reason': 'Diverse perspectives ensure comprehensive evaluation',
            'impact': '+18% win rate'
        })

    return recommendations

