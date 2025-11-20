"""Competitor keywords for detection in intent signals"""

COMPETITOR_KEYWORDS = [
    'zscaler',
    'netskope', 
    'crowdstrike',
    'palo alto',
    'fortinet',
    'cisco',
    'checkpoint',
    'mcafee',
    'symantec',
    'trend micro',
    'sophos',
    'bitdefender',
    'proofpoint',
    'mimecast',
    'okta',
    'ping identity'
]

def detect_competitor_intent(keyword_topic):
    """Check if keyword topic mentions a competitor"""
    if not keyword_topic:
        return False
    topic_lower = str(keyword_topic).lower()
    return any(competitor in topic_lower for competitor in COMPETITOR_KEYWORDS)

def get_competitor_name(keyword_topic):
    """Extract competitor name from keyword topic"""
    if not keyword_topic:
        return None
    topic_lower = str(keyword_topic).lower()
    for competitor in COMPETITOR_KEYWORDS:
        if competitor in topic_lower:
            return competitor.title()
    return None

