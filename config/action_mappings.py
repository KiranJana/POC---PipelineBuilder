"""Action mappings for patterns to recommended actions"""

# Map pattern types to recommended actions - expanded with more diverse actions
PATTERN_ACTIONS = {
    'Pattern_1': {
        'action': 'Create High-Priority Opportunity',
        'priority_actions': [
            {'action': 'Schedule executive demo within 48 hours', 'priority': 1, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'High intent with decision maker engagement'},
            {'action': 'Prepare customized ROI calculator', 'priority': 2, 'urgency': 'HIGH', 'impact': '+20% win rate', 'reason': 'Pricing interest signals strong buying intent'},
            {'action': 'Engage decision maker with executive brief', 'priority': 3, 'urgency': 'HIGH', 'impact': '+18% win rate', 'reason': 'C-level involvement requires executive alignment'},
            {'action': 'Follow up on webinar attendance with targeted questions', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+15% win rate', 'reason': 'Webinar attendance shows active evaluation'},
            {'action': 'Schedule technical deep-dive session', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+12% win rate', 'reason': 'Multi-touch engagement indicates technical evaluation'}
        ]
    },
    'Pattern_2': {
        'action': 'Create Expansion Opportunity',
        'priority_actions': [
            {'action': 'Schedule account expansion review meeting', 'priority': 1, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Returning customer showing expansion signals'},
            {'action': 'Analyze current product usage and gaps', 'priority': 2, 'urgency': 'HIGH', 'impact': '+20% win rate', 'reason': 'Identify specific expansion opportunities'},
            {'action': 'Prepare expansion value proposition', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+18% win rate', 'reason': 'Tailored messaging for expansion scenarios'},
            {'action': 'Connect with existing champion for expansion advocacy', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+15% win rate', 'reason': 'Leverage internal relationships'},
            {'action': 'Create success story based on existing implementation', 'priority': 5, 'urgency': 'LOW', 'impact': '+12% win rate', 'reason': 'Build credibility with proven results'}
        ]
    },
    'Pattern_3': {
        'action': 'Address Competitive Threat - High Priority',
        'priority_actions': [
            {'action': 'Send competitive battlecard immediately', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+35% win rate', 'reason': 'High intent with competitor research requires immediate response'},
            {'action': 'Schedule competitive displacement demo', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+30% win rate', 'reason': 'Direct competitive comparison needed'},
            {'action': 'Prepare executive-level competitive analysis', 'priority': 3, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'C-level decision making requires executive messaging'},
            {'action': 'Identify competitor they are evaluating', 'priority': 4, 'urgency': 'HIGH', 'impact': '+20% win rate', 'reason': 'Targeted messaging against specific competitor'},
            {'action': 'Create risk mitigation plan if lost to competitor', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+15% win rate', 'reason': 'Plan for potential loss scenarios'}
        ]
    },
    'Pattern_4': {
        'action': 'Create Renewal Opportunity - Strategic QBR',
        'priority_actions': [
            {'action': 'Schedule executive business review with C-suite', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+40% renewal rate', 'reason': 'C-level engagement within 30 days of renewal'},
            {'action': 'Prepare comprehensive value realization report', 'priority': 2, 'urgency': 'HIGH', 'impact': '+35% renewal rate', 'reason': 'Demonstrate clear ROI and business impact'},
            {'action': 'Discuss strategic expansion opportunities', 'priority': 3, 'urgency': 'HIGH', 'impact': '+30% renewal rate', 'reason': 'Renewal conversations should include growth'},
            {'action': 'Review success metrics and KPIs', 'priority': 4, 'urgency': 'HIGH', 'impact': '+25% renewal rate', 'reason': 'Quantify value delivered to date'},
            {'action': 'Identify and engage renewal champion', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+20% renewal rate', 'reason': 'Internal advocacy crucial for renewal'}
        ]
    },
    'Pattern_5': {
        'action': 'Secure Executive Sponsorship',
        'priority_actions': [
            {'action': 'Schedule executive sponsor alignment session', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+35% win rate', 'reason': 'Executive involvement requires executive-level engagement'},
            {'action': 'Create executive business case presentation', 'priority': 2, 'urgency': 'HIGH', 'impact': '+30% win rate', 'reason': 'C-level decisions need business-level justification'},
            {'action': 'Map executive decision-making process', 'priority': 3, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Understand executive timeline and criteria'},
            {'action': 'Prepare peer executive references', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Social proof at executive level'},
            {'action': 'Develop executive communication plan', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+18% win rate', 'reason': 'Consistent executive-level messaging'}
        ]
    },
    'Pattern_6': {
        'action': 'Save At-Risk Renewal - Emergency Intervention',
        'priority_actions': [
            {'action': 'Schedule urgent executive intervention call', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+45% save rate', 'reason': 'Declining engagement within 30 days requires immediate action'},
            {'action': 'Conduct emergency value audit and remediation', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+40% save rate', 'reason': 'Identify and fix any product or service issues'},
            {'action': 'Escalate to executive level on both sides', 'priority': 3, 'urgency': 'CRITICAL', 'impact': '+35% save rate', 'reason': 'Executive sponsorship needed to save renewal'},
            {'action': 'Create retention incentive proposal', 'priority': 4, 'urgency': 'HIGH', 'impact': '+30% save rate', 'reason': 'Financial incentives may be needed'},
            {'action': 'Develop detailed renewal success plan', 'priority': 5, 'urgency': 'HIGH', 'impact': '+25% save rate', 'reason': 'Clear roadmap for renewal success'}
        ]
    },
    'Pattern_7': {
        'action': 'Advance Expansion Opportunity',
        'priority_actions': [
            {'action': 'Schedule expansion-focused product demo', 'priority': 1, 'urgency': 'HIGH', 'impact': '+28% win rate', 'reason': 'Demo request + pricing interest signals buying readiness'},
            {'action': 'Prepare expansion use case analysis', 'priority': 2, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Understand specific expansion scenarios'},
            {'action': 'Create expansion ROI model', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+22% win rate', 'reason': 'Quantify expansion benefits'},
            {'action': 'Connect with technical stakeholders for integration planning', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Technical evaluation needed'},
            {'action': 'Develop expansion implementation timeline', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+18% win rate', 'reason': 'Clear timeline reduces buying friction'}
        ]
    },
    'Pattern_8': {
        'action': 'Re-engage Stalled New Opportunity',
        'priority_actions': [
            {'action': 'Send personalized re-engagement email campaign', 'priority': 1, 'urgency': 'HIGH', 'impact': '+20% re-engagement', 'reason': 'Stalled new opportunity needs nurturing'},
            {'action': 'Create new value proposition based on recent changes', 'priority': 2, 'urgency': 'HIGH', 'impact': '+18% re-engagement', 'reason': 'Refresh messaging with new content'},
            {'action': 'Schedule low-pressure check-in call', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+15% re-engagement', 'reason': 'Re-establish relationship without pressure'},
            {'action': 'Send industry-relevant content and insights', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+12% re-engagement', 'reason': 'Demonstrate thought leadership'},
            {'action': 'Offer educational webinar or workshop', 'priority': 5, 'urgency': 'LOW', 'impact': '+10% re-engagement', 'reason': 'Low-commitment way to re-engage'}
        ]
    },
    'Pattern_9': {
        'action': 'Re-engage Declining New Opportunity',
        'priority_actions': [
            {'action': 'Schedule intervention call to understand concerns', 'priority': 1, 'urgency': 'HIGH', 'impact': '+25% save rate', 'reason': 'Declining engagement requires immediate understanding'},
            {'action': 'Send targeted content addressing potential objections', 'priority': 2, 'urgency': 'HIGH', 'impact': '+20% save rate', 'reason': 'Address specific concerns with relevant content'},
            {'action': 'Offer alternative solutions or approaches', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+18% save rate', 'reason': 'Different approach may reignite interest'},
            {'action': 'Connect with customer success peer', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+15% save rate', 'reason': 'Peer-level relationship building'},
            {'action': 'Create customized evaluation plan', 'priority': 5, 'urgency': 'LOW', 'impact': '+12% save rate', 'reason': 'Structured approach may help decision making'}
        ]
    },
    'Pattern_10': {
        'action': 'Accelerate High-Intent New Opportunity',
        'priority_actions': [
            {'action': 'Immediate SDR outreach within 2 hours', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+40% conversion', 'reason': 'High surge signals require immediate response'},
            {'action': 'Send personalized product walkthrough', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+35% conversion', 'reason': 'High intent needs immediate product education'},
            {'action': 'Schedule demo within 24 hours', 'priority': 3, 'urgency': 'HIGH', 'impact': '+30% conversion', 'reason': 'Fast demo booking critical for conversion'},
            {'action': 'Prepare competitive intelligence brief', 'priority': 4, 'urgency': 'HIGH', 'impact': '+25% conversion', 'reason': 'High activity may indicate competitive evaluation'},
            {'action': 'Assign dedicated onboarding specialist', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+20% conversion', 'reason': 'Dedicated resource shows commitment'}
        ]
    },
    'Pattern_11': {
        'action': 'Address Long-Cycle Competitive Evaluation',
        'priority_actions': [
            {'action': 'Send comprehensive competitive differentiation framework', 'priority': 1, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Long cycle indicates thorough competitive evaluation'},
            {'action': 'Schedule competitive comparison workshop', 'priority': 2, 'urgency': 'HIGH', 'impact': '+22% win rate', 'reason': 'Educate on differentiation points'},
            {'action': 'Provide detailed feature comparison matrix', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Detailed analysis needed for long cycles'},
            {'action': 'Arrange peer executive reference calls', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+18% win rate', 'reason': 'Social proof from similar companies'},
            {'action': 'Create risk assessment if choosing competitor', 'priority': 5, 'urgency': 'LOW', 'impact': '+15% win rate', 'reason': 'Highlight switching costs and risks'}
        ]
    },
    'Pattern_12': {
        'action': 'Leverage Expansion with Case Study',
        'priority_actions': [
            {'action': 'Send relevant expansion success stories', 'priority': 1, 'urgency': 'HIGH', 'impact': '+22% win rate', 'reason': 'Case study interest shows proof point demand'},
            {'action': 'Schedule peer reference call with similar expansion', 'priority': 2, 'urgency': 'HIGH', 'impact': '+20% win rate', 'reason': 'Peer validation crucial for expansion'},
            {'action': 'Create customized expansion roadmap', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+18% win rate', 'reason': 'Show clear path to expansion success'},
            {'action': 'Prepare expansion ROI analysis', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+16% win rate', 'reason': 'Quantify expansion benefits'},
            {'action': 'Connect with expansion champion', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+14% win rate', 'reason': 'Identify and nurture internal advocate'}
        ]
    },
    'Pattern_13': {
        'action': 'Save Declining Renewal - Intensive Care',
        'priority_actions': [
            {'action': 'Schedule emergency account review with leadership', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+40% save rate', 'reason': 'Declining engagement within 60 days critical'},
            {'action': 'Conduct comprehensive health check and remediation', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+35% save rate', 'reason': 'Identify all issues causing dissatisfaction'},
            {'action': 'Create renewal value proposition with incentives', 'priority': 3, 'urgency': 'HIGH', 'impact': '+30% save rate', 'reason': 'Address value concerns with concrete benefits'},
            {'action': 'Escalate to executive sponsorship on both sides', 'priority': 4, 'urgency': 'HIGH', 'impact': '+25% save rate', 'reason': 'Executive intervention often needed'},
            {'action': 'Develop detailed retention and success plan', 'priority': 5, 'urgency': 'HIGH', 'impact': '+20% save rate', 'reason': 'Clear plan to restore confidence'}
        ]
    },
    'Pattern_14': {
        'action': 'Advance Integration-Focused Opportunity',
        'priority_actions': [
            {'action': 'Send detailed integration documentation and guides', 'priority': 1, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Integration interest requires technical detail'},
            {'action': 'Schedule technical architecture review session', 'priority': 2, 'urgency': 'HIGH', 'impact': '+22% win rate', 'reason': 'Technical stakeholders need architecture discussion'},
            {'action': 'Provide integration sandbox environment', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Hands-on testing reduces technical risk'},
            {'action': 'Connect with solutions engineering team', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+18% win rate', 'reason': 'Technical expertise needed for complex integrations'},
            {'action': 'Create integration timeline and milestones', 'priority': 5, 'urgency': 'LOW', 'impact': '+15% win rate', 'reason': 'Clear implementation plan reduces concerns'}
        ]
    },
    'Pattern_15': {
        'action': 'Address Renewal Pricing Concerns',
        'priority_actions': [
            {'action': 'Send renewal pricing options with analysis', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+35% renewal rate', 'reason': 'Pricing interest close to renewal date'},
            {'action': 'Schedule renewal negotiation session', 'priority': 2, 'urgency': 'HIGH', 'impact': '+30% renewal rate', 'reason': 'Direct discussion of pricing concerns'},
            {'action': 'Prepare value vs cost analysis', 'priority': 3, 'urgency': 'HIGH', 'impact': '+25% renewal rate', 'reason': 'Demonstrate ROI justification'},
            {'action': 'Discuss multi-year renewal incentives', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+20% renewal rate', 'reason': 'Longer commitments may have better pricing'},
            {'action': 'Review contract terms and optimization opportunities', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+15% renewal rate', 'reason': 'Find ways to optimize current contract'}
        ]
    },
    'Pattern_16': {
        'action': 'Address New Opportunity Pricing Interest',
        'priority_actions': [
            {'action': 'Send comprehensive pricing information and options', 'priority': 1, 'urgency': 'HIGH', 'impact': '+28% win rate', 'reason': 'Pricing interest after 30+ days indicates serious evaluation'},
            {'action': 'Schedule pricing discussion and negotiation', 'priority': 2, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Direct conversation about pricing concerns'},
            {'action': 'Prepare competitive pricing analysis', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+22% win rate', 'reason': 'Show how pricing compares to alternatives'},
            {'action': 'Create payment plan options', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Flexible payment terms may address budget concerns'},
            {'action': 'Discuss value-based pricing approach', 'priority': 5, 'urgency': 'LOW', 'impact': '+18% win rate', 'reason': 'Focus on value delivered rather than cost'}
        ]
    },
    'Pattern_17': {
        'action': 'Capitalize on High-Surge New Opportunity',
        'priority_actions': [
            {'action': 'Immediate response within 1 hour', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+45% conversion', 'reason': 'Surge signals require immediate attention'},
            {'action': 'Send personalized urgent proposal', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+40% conversion', 'reason': 'High intent needs immediate offer'},
            {'action': 'Schedule emergency demo within 4 hours', 'priority': 3, 'urgency': 'CRITICAL', 'impact': '+35% conversion', 'reason': 'Fast demo critical for conversion'},
            {'action': 'Prepare expedited contracting process', 'priority': 4, 'urgency': 'HIGH', 'impact': '+30% conversion', 'reason': 'Remove any speedbumps to purchase'},
            {'action': 'Assign executive sponsor for fast-track approval', 'priority': 5, 'urgency': 'HIGH', 'impact': '+25% conversion', 'reason': 'Executive involvement needed for speed'}
        ]
    },
    'Pattern_18': {
        'action': 'Counter Expansion Competitive Threat',
        'priority_actions': [
            {'action': 'Send urgent competitive comparison for expansion', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+32% win rate', 'reason': 'Competitive threat during expansion evaluation'},
            {'action': 'Schedule executive-level competitive discussion', 'priority': 2, 'urgency': 'HIGH', 'impact': '+28% win rate', 'reason': 'C-level involvement requires executive engagement'},
            {'action': 'Prepare expansion success stories vs competitors', 'priority': 3, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Proof points against specific competitive claims'},
            {'action': 'Create expansion risk analysis if choosing competitor', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+22% win rate', 'reason': 'Highlight risks of competitor expansion'},
            {'action': 'Offer expansion incentives and guarantees', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Financial incentives to choose our solution'}
        ]
    },
    'Pattern_19': {
        'action': 'Address Technical Evaluation Concerns',
        'priority_actions': [
            {'action': 'Send comprehensive integration documentation', 'priority': 1, 'urgency': 'HIGH', 'impact': '+20% win rate', 'reason': 'Long cycle with technical interest needs detail'},
            {'action': 'Send detailed pricing and packaging information', 'priority': 2, 'urgency': 'HIGH', 'impact': '+18% win rate', 'reason': 'Pricing evaluation needs comprehensive options'},
            {'action': 'Schedule technical discovery session', 'priority': 3, 'urgency': 'MEDIUM', 'impact': '+16% win rate', 'reason': 'Understand specific technical requirements'},
            {'action': 'Provide proof of concept environment', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+14% win rate', 'reason': 'Technical validation needed'},
            {'action': 'Connect with customer success technical team', 'priority': 5, 'urgency': 'LOW', 'impact': '+12% win rate', 'reason': 'Ongoing technical support and guidance'}
        ]
    },
    'Pattern_20': {
        'action': 'Save Critical Renewal - All Hands on Deck',
        'priority_actions': [
            {'action': 'Schedule emergency executive summit', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+45% save rate', 'reason': 'Pricing interest + declining engagement = critical situation'},
            {'action': 'Conduct immediate value audit and gap analysis', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+40% save rate', 'reason': 'Identify all value delivery issues'},
            {'action': 'Create retention proposal with significant incentives', 'priority': 3, 'urgency': 'CRITICAL', 'impact': '+35% save rate', 'reason': 'Financial concessions may be needed'},
            {'action': 'Escalate to CEO level if necessary', 'priority': 4, 'urgency': 'HIGH', 'impact': '+30% save rate', 'reason': 'Executive intervention required'},
            {'action': 'Develop comprehensive renewal success roadmap', 'priority': 5, 'urgency': 'HIGH', 'impact': '+25% save rate', 'reason': 'Clear plan to rebuild relationship'}
        ]
    },
    'Pattern_21': {
        'action': 'Win Competitive Pricing Battle',
        'priority_actions': [
            {'action': 'Send competitive pricing comparison matrix', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+35% win rate', 'reason': 'Competitive + pricing evaluation needs immediate response'},
            {'action': 'Schedule competitive demo highlighting advantages', 'priority': 2, 'urgency': 'HIGH', 'impact': '+30% win rate', 'reason': 'Direct comparison needed'},
            {'action': 'Prepare executive-level ROI analysis', 'priority': 3, 'urgency': 'HIGH', 'impact': '+25% win rate', 'reason': 'Business case needed to justify premium pricing'},
            {'action': 'Offer competitive replacement incentives', 'priority': 4, 'urgency': 'MEDIUM', 'impact': '+20% win rate', 'reason': 'Financial incentives for switching'},
            {'action': 'Create risk mitigation plan for switching costs', 'priority': 5, 'urgency': 'MEDIUM', 'impact': '+15% win rate', 'reason': 'Address concerns about transition costs'}
        ]
    },
    'Pattern_22': {
        'action': 'Save Renewal from Competitive Threat',
        'priority_actions': [
            {'action': 'Send emergency competitive differentiation package', 'priority': 1, 'urgency': 'CRITICAL', 'impact': '+50% save rate', 'reason': 'Competitive threat 7 days from renewal requires immediate action'},
            {'action': 'Schedule executive-level competitive defense call', 'priority': 2, 'urgency': 'CRITICAL', 'impact': '+45% save rate', 'reason': 'C-level competitive discussion needed'},
            {'action': 'Create renewal proposal with competitive advantages', 'priority': 3, 'urgency': 'CRITICAL', 'impact': '+40% save rate', 'reason': 'Emphasize why staying is better than switching'},
            {'action': 'Offer renewal incentives and guarantees', 'priority': 4, 'urgency': 'HIGH', 'impact': '+35% save rate', 'reason': 'Financial incentives to stay'},
            {'action': 'Develop detailed competitive win strategy', 'priority': 5, 'urgency': 'HIGH', 'impact': '+30% save rate', 'reason': 'Coordinated plan to address competitive threat'}
        ]
    }
}

def get_actions_for_pattern(pattern_id):
    """Get recommended actions for a pattern"""
    return PATTERN_ACTIONS.get(pattern_id, {
        'action': 'Review Opportunity',
        'priority_actions': [
            {'action': 'Review opportunity details', 'priority': 1, 'urgency': 'MEDIUM', 'impact': 'Unknown'}
        ]
    })

