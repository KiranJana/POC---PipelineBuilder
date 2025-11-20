"""
Layer 5: LLM Explanation Generator
Converts structured recommendations into natural language explanations
Includes template fallback if LLM unavailable
"""

import json
import os


class LLMExplainer:
    """
    Layer 5: LLM Explanation Generator
    Converts structured JSON recommendations into natural language
    """
    
    def __init__(self, use_llm=False, llm_api_key=None, llm_provider='gemini', model_name=None, temperature=0.7, max_tokens=400, **kwargs):
        """
        Initialize LLM explainer

        Args:
            use_llm: Whether to use LLM (True) or templates (False)
            llm_api_key: API key for LLM provider
            llm_provider: 'openai', 'anthropic', or 'gemini'
            model_name: Specific model name to use (optional, uses provider defaults if not specified)
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum length of generated explanation
            **kwargs: Additional parameters (ignored for compatibility)
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_client = None
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if use_llm:
            # Validate API key is available
            api_key = None
            if llm_provider == 'openai':
                api_key = llm_api_key or os.getenv('OPENAI_API_KEY')
            elif llm_provider == 'anthropic':
                api_key = llm_api_key or os.getenv('ANTHROPIC_API_KEY')
            elif llm_provider == 'gemini':
                api_key = llm_api_key or os.getenv('GEMINI_API_KEY')

            if not api_key:
                print(f"[Layer 5] Warning: No API key found for {llm_provider}. Set {llm_provider.upper()}_API_KEY environment variable or pass llm_api_key parameter.")
                print("[Layer 5] Falling back to template-based explanations")
                self.use_llm = False
                return

            try:
                if llm_provider == 'openai':
                    import openai
                    self.llm_client = openai.OpenAI(api_key=api_key)
                    self.model_name = model_name or "gpt-4"
                    print("[Layer 5] LLM Explainer initialized with OpenAI GPT-4")
                elif llm_provider == 'anthropic':
                    import anthropic
                    self.llm_client = anthropic.Anthropic(api_key=api_key)
                    self.model_name = model_name or "claude-3-sonnet-20240229"
                    print("[Layer 5] LLM Explainer initialized with Anthropic Claude")
                elif llm_provider == 'gemini':
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    # Use model from config, with fallback to stable model
                    model_name = model_name or getattr(self, 'model_name', 'gemini-1.5-flash')
                    self.llm_client = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    print(f"[Layer 5] LLM Explainer initialized with Google Gemini: {model_name}")
            except Exception as e:
                print(f"[Layer 5] Warning: Could not initialize LLM: {e}")
                print("[Layer 5] Falling back to template-based explanations")
                self.use_llm = False
        else:
            print("[Layer 5] LLM Explainer initialized with template-based explanations")
    
    def generate_explanation(self, recommendation):
        """
        Generate natural language explanation for a recommendation
        """
        if self.use_llm and self.llm_client:
            try:
                return self._generate_llm_explanation(recommendation)
            except Exception as e:
                print(f"[Layer 5] LLM generation failed: {e}, falling back to template")
                return self._generate_template_explanation(recommendation)
        else:
            return self._generate_template_explanation(recommendation)
    
    def _generate_llm_explanation(self, recommendation):
        """Generate explanation using LLM"""
        prompt = self._build_llm_prompt(recommendation)
        
        if self.llm_provider == 'openai':
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a sales AI assistant helping sales reps prioritize opportunities. Generate clear, actionable explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == 'anthropic':
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=400,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        
        elif self.llm_provider == 'gemini':
            response = self.llm_client.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 400,
                }
            )
            return response.text
    
    def _build_llm_prompt(self, recommendation):
        """Build prompt for LLM"""
        prompt = f"""You are a sales AI assistant. Generate a clear, actionable explanation for this recommendation:

**Recommendation:** {recommendation['recommendation_type']}
**Confidence:** {recommendation['confidence']:.0%}
**Account:** {recommendation.get('account', 'Unknown')}

**Matched Rules:**
{json.dumps(recommendation.get('matched_rules', []), indent=2)}

**Discovered Patterns:**
{json.dumps(recommendation.get('discovered_patterns', []), indent=2)}

**ML Prediction:**
{json.dumps(recommendation.get('ml_prediction', {}), indent=2)}

**Similar Historical Deals:**
{json.dumps(recommendation.get('similar_deals', [])[:3], indent=2)}

**Recommended Actions:**
{json.dumps(recommendation.get('recommended_actions', []), indent=2)}

Generate a conversational summary (200-300 words) that:
1. Explains WHY we're confident (which signals/patterns matched)
2. Lists WHAT TO DO (prioritized actions with urgency)
3. Provides CONTEXT (similar historical deals and outcomes)
4. Sets EXPECTATIONS (expected deal size, close timeline)

Keep it concise, friendly, and actionable. Use emojis sparingly for emphasis."""

        return prompt
    
    def _generate_template_explanation(self, recommendation):
        """Generate explanation using templates (fallback)"""
        
        # Header (use plain text for Windows compatibility)
        explanation = f"""TARGET: {recommendation['recommendation_type']} for {recommendation.get('account', 'Unknown')}

Confidence: {recommendation['confidence']:.0%}

{'='*55}

WHY I'M CONFIDENT:
"""
        
        # Add why confident
        for reason in recommendation['explanation_structured']['why_confident']:
            explanation += f"\n* {reason}"
        
        # Add key signals
        if recommendation['explanation_structured']['key_signals']:
            explanation += "\n\nKEY SIGNALS:"
            for signal in recommendation['explanation_structured']['key_signals'][:5]:
                explanation += f"\n- {signal}"
        
        explanation += "\n\n" + "="*55 + "\n"
        
        # Add recommended actions
        if recommendation.get('recommended_actions'):
            explanation += "\nWHAT TO DO:\n"
            for i, action in enumerate(recommendation['recommended_actions'][:5], 1):
                urgency_flag = "URGENT: " if action['urgency'] in ['CRITICAL', 'HIGH'] else ""
                explanation += f"\n{i}. {urgency_flag}{action['action']} [{action['urgency']}]"
                explanation += f"\n   -> {action.get('reason', 'Recommended based on signals')}"
                if 'impact' in action:
                    explanation += f" ({action['impact']})"
        
        explanation += "\n\n" + "="*55 + "\n"
        
        # Add similar deals context
        if recommendation.get('similar_deals'):
            explanation += "\nSIMILAR DEALS THAT WON:\n"
            for i, deal in enumerate(recommendation['similar_deals'][:3], 1):
                explanation += f"\n- {deal.get('account', 'Unknown')}: ${deal.get('amount', 0):,.0f} in {deal.get('close_time_days', 0)} days ({deal.get('similarity_score', 0):.0%} similar)"
            
            # Calculate averages
            amounts = [d['amount'] for d in recommendation['similar_deals'] if d.get('amount', 0) > 0]
            times = [d['close_time_days'] for d in recommendation['similar_deals'] if d.get('close_time_days', 0) > 0]
            
            if amounts and times:
                avg_amount = sum(amounts) / len(amounts)
                avg_time = sum(times) / len(times)
                explanation += f"\n\nExpected outcome: ${avg_amount:,.0f} deal, {avg_time:.0f} day close cycle"
        else:
            explanation += f"\n{recommendation['explanation_structured']['historical_context']}"
        
        explanation += "\n\n" + "="*55 + "\n"
        
        # Add temporal analysis
        if recommendation.get('temporal_analysis'):
            temporal = recommendation['temporal_analysis']
            explanation += f"\nTEMPORAL ANALYSIS:"
            explanation += f"\n- Engagement trend: {temporal.get('engagement_trend', 'UNKNOWN')}"
            explanation += f"\n- Urgency: {temporal.get('urgency', 'MEDIUM')}"
            if temporal.get('is_stalled'):
                explanation += "\n- WARNING: Deal is stalled - immediate action required"
        
        # Add buying committee analysis
        if recommendation.get('buying_committee_analysis'):
            committee = recommendation['buying_committee_analysis']
            explanation += f"\n\nBUYING COMMITTEE:"
            explanation += f"\n- Persona coverage: {committee.get('persona_coverage', 0):.0%}"
            explanation += f"\n- C-level engaged: {'Yes' if committee.get('has_c_level') else 'No'}"
            explanation += f"\n- Decision maker engaged: {'Yes' if committee.get('has_decision_maker') else 'No'}"
            explanation += f"\n- Risk: {committee.get('risk', 'UNKNOWN')}"
        
        return explanation
    
    def generate_explanations_batch(self, recommendations, requests_per_minute=None):
        """
        Generate explanations for a batch of recommendations with rate limiting
        """
        # Use parameter if provided, otherwise use instance variable
        if requests_per_minute is None:
            requests_per_minute = getattr(self, 'requests_per_minute', 10)

        print(f"\n[Layer 5] Generating explanations for {len(recommendations)} recommendations...")
        print(f"[Layer 5] Rate limiting: {requests_per_minute} requests per minute")

        import time

        # Calculate minimum delay between requests (in seconds)
        min_delay = 60.0 / requests_per_minute if requests_per_minute > 0 else 0

        explanations = []
        for i, rec in enumerate(recommendations):
            if i % 10 == 0:  # More frequent progress updates
                print(f"  Processing {i+1}/{len(recommendations)}...")

            start_time = time.time()

            try:
                explanation = self.generate_explanation(rec)
                explanations.append({
                    'opportunity_id': rec.get('opportunity_id'),
                    'explanation': explanation
                })
            except Exception as e:
                print(f"[Layer 5] Error generating explanation for {rec.get('opportunity_id')}: {e}")
                explanations.append({
                    'opportunity_id': rec.get('opportunity_id'),
                    'explanation': f"Error generating explanation: {e}"
                })

            # Rate limiting: ensure minimum delay between requests
            if i < len(recommendations) - 1:  # Don't delay after last request
                elapsed = time.time() - start_time
                if elapsed < min_delay:
                    sleep_time = min_delay - elapsed
                    print(f"[Layer 5] Rate limiting: sleeping {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)

        print(f"[Layer 5] Generated {len(explanations)} explanations with rate limiting")
        return explanations
    
    def save_explanations(self, explanations, output_path='outputs/recommendations/explanations.json'):
        """Save explanations to JSON"""
        with open(output_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        print(f"\n[Layer 5] Saved {len(explanations)} explanations to {output_path}")


def create_llm_explainer(use_llm=False, llm_api_key=None, llm_provider='gemini', model_name=None, temperature=0.7, max_tokens=400, requests_per_minute=10):
    """Factory function to create LLM explainer"""
    explainer = LLMExplainer(use_llm=use_llm, llm_api_key=llm_api_key, llm_provider=llm_provider, model_name=model_name)
    # Store rate limit for batch processing
    explainer.requests_per_minute = requests_per_minute
    return explainer

