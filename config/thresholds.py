"""
Centralized Thresholds Configuration
All magic numbers and thresholds used across the pipeline.
"""

# ===================================================================
# INTENT SCORE THRESHOLDS
# ===================================================================
INTENT_HIGH_THRESHOLD = 75          # Intent score >= 75 is HIGH
INTENT_MEDIUM_THRESHOLD = 50        # Intent score >= 50 is MEDIUM (< 75)
INTENT_LOW_THRESHOLD = 0            # Intent score < 50 is LOW

INTENT_SURGE_HIGH_THRESHOLD = 80    # Surge detection threshold

# ===================================================================
# ACTIVITY COUNT THRESHOLDS
# ===================================================================
ACTIVITY_HIGH_THRESHOLD = 10        # >= 10 activities is HIGH
ACTIVITY_MEDIUM_THRESHOLD = 5       # >= 5 activities is MEDIUM (< 10)
ACTIVITY_LOW_THRESHOLD = 0          # < 5 activities is LOW

EMAIL_HIGH_THRESHOLD = 10           # >= 10 emails is HIGH
EMAIL_MEDIUM_THRESHOLD = 5          # >= 5 emails is MEDIUM
EMAIL_LOW_THRESHOLD = 0             # < 5 emails is LOW

CALL_HIGH_THRESHOLD = 5             # >= 5 calls is HIGH
CALL_MEDIUM_THRESHOLD = 3           # >= 3 calls is MEDIUM
CALL_LOW_THRESHOLD = 0              # < 3 calls is LOW
CALL_MIN_THRESHOLD = 3              # Minimum calls for rules (alternative name)

MEETING_HIGH_THRESHOLD = 3          # >= 3 meetings is HIGH
MEETING_MEDIUM_THRESHOLD = 1        # >= 1 meeting is MEDIUM
MEETING_LOW_THRESHOLD = 0           # < 1 meeting is LOW

# ===================================================================
# CONTACT/STAKEHOLDER THRESHOLDS
# ===================================================================
DECISION_MAKER_HIGH_THRESHOLD = 3   # >= 3 decision makers is HIGH
DECISION_MAKER_MIN = 1              # Minimum decision makers for rules

C_LEVEL_HIGH_THRESHOLD = 2          # >= 2 C-levels is HIGH
C_LEVEL_MIN = 1                     # Minimum C-levels for rules

CHAMPION_MIN = 1                    # Minimum champions

COMMITTEE_SIZE_MIN = 3              # Minimum buying committee size

# ===================================================================
# TEMPORAL THRESHOLDS (DAYS)
# ===================================================================
DAYS_RECENT = 7                     # Last 7 days is "recent"
DAYS_SHORT_TERM = 14                # Short-term window
DAYS_MEDIUM_TERM = 30               # Medium-term window
DAYS_LONG_TERM = 50                 # Long-term window

DAYS_TO_RENEWAL_URGENT = 30         # < 30 days to renewal is urgent
DAYS_TO_RENEWAL_DEFAULT = 999       # Default for non-renewal opportunities

DEAL_DURATION_STALLED = 60          # > 60 days with low activity = stalled

# ===================================================================
# PAGE VISIT THRESHOLDS
# ===================================================================
PAGE_VISIT_HIGH_THRESHOLD = 5       # >= 5 visits is HIGH
PAGE_VISIT_MEDIUM_THRESHOLD = 2     # >= 2 visits is MEDIUM
PAGE_VISIT_LOW_THRESHOLD = 0        # < 2 visits is LOW

# ===================================================================
# CONFIDENCE THRESHOLDS
# ===================================================================
CONFIDENCE_RULE_HIGH = 0.95         # High-confidence rule match
CONFIDENCE_RULE_MEDIUM = 0.90       # Medium-confidence rule match

CONFIDENCE_PATTERN_HIGH = 0.80      # High-confidence pattern
CONFIDENCE_PATTERN_MEDIUM = 0.70    # Medium-confidence pattern
CONFIDENCE_PATTERN_MIN = 0.65       # Minimum pattern confidence

CONFIDENCE_ML_THRESHOLD = 0.60      # Minimum ML prediction confidence

CONFIDENCE_RECOMMENDATION_HIGH = 0.80
CONFIDENCE_RECOMMENDATION_MEDIUM = 0.70
CONFIDENCE_RECOMMENDATION_LOW = 0.60

# ===================================================================
# ENSEMBLE WEIGHTS
# ===================================================================
ENSEMBLE_WEIGHTS = {
    'rule': 0.5,        # Rules have highest priority
    'pattern': 0.3,     # Patterns second priority
    'ml': 0.2          # ML fills in the messy middle
}

# ===================================================================
# PATTERN DISCOVERY THRESHOLDS
# ===================================================================
PATTERN_MIN_SUPPORT = 0.02          # 2% minimum support for patterns
PATTERN_MIN_CONFIDENCE = 0.65       # 65% minimum confidence

PATTERN_MAX_RULES_STATISTICAL = 15  # Max patterns for statistical mode
PATTERN_MAX_RULES_ML = 10           # Max patterns for ML mode

PATTERN_SAMPLE_SIZE_THRESHOLD = 1000  # Use Apriori if >= 1000 deals

# ===================================================================
# VELOCITY THRESHOLDS
# ===================================================================
VELOCITY_HIGH_THRESHOLD = 2.0       # >= 2 activities/day is HIGH
VELOCITY_MEDIUM_THRESHOLD = 0.5     # >= 0.5 activities/day is MEDIUM
VELOCITY_LOW_THRESHOLD = 0.0        # < 0.5 activities/day is LOW

# ===================================================================
# ENGAGEMENT TREND DETECTION
# ===================================================================
ENGAGEMENT_INCREASE_THRESHOLD = 0.2  # 20% increase = increasing trend
ENGAGEMENT_DECREASE_THRESHOLD = -0.2 # 20% decrease = decreasing trend

# ===================================================================
# AMOUNT THRESHOLDS (Deal Size)
# ===================================================================
AMOUNT_LARGE_DEAL = 100000          # >= $100k is large deal
AMOUNT_MEDIUM_DEAL = 50000          # >= $50k is medium deal
AMOUNT_SMALL_DEAL = 0               # < $50k is small deal

# ===================================================================
# RESPONSE RATE THRESHOLDS
# ===================================================================
RESPONSE_RATE_HIGH = 0.5            # >= 50% response rate is HIGH
RESPONSE_RATE_MEDIUM = 0.25         # >= 25% response rate is MEDIUM
RESPONSE_RATE_LOW = 0.0             # < 25% response rate is LOW

# ===================================================================
# BUYING COMMITTEE THRESHOLDS
# ===================================================================
COMMITTEE_COMPLETE_THRESHOLD = 0.75  # 75% persona coverage = complete
COMMITTEE_PARTIAL_THRESHOLD = 0.5    # 50% persona coverage = partial

COMMITTEE_DIVERSITY_HIGH = 0.7      # >= 0.7 diversity score
COMMITTEE_DIVERSITY_MEDIUM = 0.4    # >= 0.4 diversity score

# ===================================================================
# SIMILARITY SEARCH THRESHOLDS
# ===================================================================
SIMILARITY_MIN_SCORE = 0.7          # Minimum cosine similarity
SIMILARITY_TOP_K = 3                # Return top 3 similar deals

# ===================================================================
# LLM CONFIGURATION
# ===================================================================
LLM_RATE_LIMIT_REQUESTS_PER_MINUTE = 10
LLM_RATE_LIMIT_DELAY = 6.0          # Seconds between requests

# ===================================================================
# VALIDATION THRESHOLDS
# ===================================================================
VALIDATION_MIN_PRECISION = 0.60     # Minimum acceptable precision
VALIDATION_MIN_RECALL = 0.50        # Minimum acceptable recall

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_intent_category(intent_score):
    """Categorize intent score into HIGH/MEDIUM/LOW"""
    if intent_score >= INTENT_HIGH_THRESHOLD:
        return 'HIGH'
    elif intent_score >= INTENT_MEDIUM_THRESHOLD:
        return 'MEDIUM'
    else:
        return 'LOW'

def get_activity_category(activity_count):
    """Categorize activity count into HIGH/MEDIUM/LOW"""
    if activity_count >= ACTIVITY_HIGH_THRESHOLD:
        return 'HIGH'
    elif activity_count >= ACTIVITY_MEDIUM_THRESHOLD:
        return 'MEDIUM'
    else:
        return 'LOW'

def get_velocity_category(velocity):
    """Categorize velocity into HIGH/MEDIUM/LOW"""
    if velocity >= VELOCITY_HIGH_THRESHOLD:
        return 'HIGH'
    elif velocity >= VELOCITY_MEDIUM_THRESHOLD:
        return 'MEDIUM'
    else:
        return 'LOW'
