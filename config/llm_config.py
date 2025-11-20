"""LLM Configuration for Layer 5 Explainer"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Try to find .env file in the same directory as this config file
config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# LLM Provider Configuration
LLM_ENABLED = True  # Set to True to use LLM explanations, False for template-based only
LLM_PROVIDER = 'gemini'  # Options: 'gemini', 'openai', 'anthropic'

# API Keys (loaded from environment variables)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Model Configuration
GEMINI_MODEL = "gemini-2.0-flash"  # Fast and efficient
OPENAI_MODEL = "gpt-4"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"

# Generation Parameters
TEMPERATURE = 0.7  # Creativity level (0.0 = deterministic, 1.0 = creative)
MAX_TOKENS = 400  # Maximum length of generated explanation

# Rate Limiting (requests per minute)
REQUESTS_PER_MINUTE = {
    'gemini': 10,     # Gemini 2.5 Flash free tier: 10 RPM, 250 RPD
    'openai': 60,     # OpenAI typical limit
    'anthropic': 50   # Anthropic typical limit
}


def get_llm_config():
    """Get LLM configuration for Layer 5"""
    if LLM_PROVIDER == 'gemini':
        api_key = GEMINI_API_KEY
    elif LLM_PROVIDER == 'openai':
        api_key = OPENAI_API_KEY
    elif LLM_PROVIDER == 'anthropic':
        api_key = ANTHROPIC_API_KEY
    else:
        api_key = None

    # Get rate limit for current provider
    requests_per_minute = REQUESTS_PER_MINUTE.get(LLM_PROVIDER, 10)

    # Get model name based on provider
    if LLM_PROVIDER == 'gemini':
        model_name = GEMINI_MODEL
    elif LLM_PROVIDER == 'openai':
        model_name = OPENAI_MODEL
    elif LLM_PROVIDER == 'anthropic':
        model_name = ANTHROPIC_MODEL
    else:
        model_name = None

    return {
        'use_llm': LLM_ENABLED,
        'llm_provider': LLM_PROVIDER,
        'llm_api_key': api_key,
        'model_name': model_name,
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS,
        'requests_per_minute': requests_per_minute
    }

