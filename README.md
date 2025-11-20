# Pipeline Builder - Enhanced Hybrid POC

**5-Layer AI System for Sales Opportunity Prediction & Recommendations**

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn lightgbm shap mlxtend joblib google-generativeai
```

**Note:** Google Gemini 2.0 Flash is already configured! See `GEMINI_SETUP.md` for details.

### 2. Train the System
```bash
python main_train_pipeline.py
```

### 3. Score Opportunities
```bash
python main_score_pipeline.py
```

### 4. Run Tests
```bash
python test_system.py
```

---

## 📁 Project Structure

```
POC - PipelineBuilder/
├── input_data/              # All CSV data files (8 files)
│   ├── crm_opportunities.csv
│   ├── crm_accounts.csv
│   ├── crm_contacts.csv
│   ├── crm_activities.csv
│   ├── intent_signals.csv
│   ├── corp_email_threads.csv
│   ├── map_events.csv
│   └── zoominfo_people.csv
│
├── config/                  # Configuration & Rules
│   ├── universal_rules.py   # 22 expert-defined rules
│   ├── competitor_keywords.py
│   └── action_mappings.py
│
├── layers/                  # 5-Layer Architecture
│   ├── layer1_rules_engine.py      # Universal Rules (≥90% precision)
│   ├── layer2_pattern_discovery.py # Pattern Mining (≥70% precision)
│   ├── layer3_ml_ensemble.py       # LGBM + SHAP (≥60% accuracy)
│   ├── layer4_recommendation.py    # Ensemble + Actions
│   └── layer5_llm_explainer.py     # Natural Language Explanations
│
├── features/                # Feature Engineering
│   ├── feature_engineering.py
│   ├── signal_extractors.py
│   ├── temporal_features.py
│   └── buying_committee.py
│
├── utils/                   # Utilities
│   ├── data_loader.py
│   ├── similarity_search.py
│   └── validation.py
│
├── outputs/                 # Generated Files
│   ├── trained_models/
│   ├── discovered_patterns/
│   ├── validation_reports/
│   └── recommendations/
│
├── main_train_pipeline.py   # Training orchestration
├── main_score_pipeline.py   # Real-time scoring
├── test_system.py           # End-to-end tests
│
└── Documentation/
    ├── README.md (this file)
    ├── README_NEW_SYSTEM.md (detailed guide)
    ├── IMPLEMENTATION_PLAN.md (architecture)
    └── IMPLEMENTATION_SUMMARY.md (what was built)
```

---

## 🏗️ Architecture

### 5-Layer Hybrid System

```
Layer 1: Universal Rules Engine (22 Expert Rules)
         ↓
Layer 2: Pattern Discovery (Association Mining + Temporal Analysis)
         ↓
Layer 3: ML Ensemble (LGBM + SHAP + Calibration)
         ↓
Layer 4: Recommendation Engine (Ensemble + Actions + RAG)
         ↓
Layer 5: LLM Explainer (Natural Language + Context)
```

---

## ✨ Key Features

### 🎯 22 Universal Rules
Expert-defined patterns covering:
- New logo opportunities
- Expansion scenarios
- Renewal management
- At-risk detection
- Competitive situations

### 🔍 Pattern Discovery
Automatically discovers patterns like:
- "Intent>80 + C-Level + Email>3 → 78% win rate"
- "Pricing Page + Demo + Decision Maker → 82% win rate"

### 📊 Temporal Analysis
- Engagement velocity trends
- Intent score momentum
- Urgency detection
- Stalled deal flagging

### 👥 Buying Committee Intelligence
- Persona coverage analysis
- Multi-threading score
- Missing persona detection
- Risk assessment

### 🎬 Action Recommendations
Each recommendation includes:
- Specific actions to take
- Priority and urgency
- Expected impact
- Clear reasoning

### 💡 Full Explainability
Every recommendation shows:
- Which signals triggered it
- Which rules/patterns matched
- Historical context (similar deals)
- Confidence scores

---

## 📈 Performance Targets

| Layer | Metric | Target | Purpose |
|-------|--------|--------|---------|
| Layer 1: Rules | Precision | ≥90% | High-confidence expert patterns |
| Layer 2: Patterns | Precision | ≥70% | Discovered patterns |
| Layer 3: ML | Accuracy | ≥60% | Complex case handling |
| Overall System | Accuracy | ≥60% | Realistic performance |

---

## 📤 Output Examples

### Structured JSON
```json
{
  "opportunity_id": "OPP_123",
  "account": "Acme Corp",
  "recommendation_type": "CREATE_HIGH_PRIORITY_OPP",
  "confidence": 0.82,
  "primary_source": "RULE",
  "recommended_actions": [
    {
      "action": "Schedule executive demo",
      "priority": 1,
      "urgency": "HIGH",
      "impact": "+25% win rate"
    }
  ]
}
```

### Natural Language
```
🎯 High-Priority Opportunity for Acme Corp
Confidence: 82%

WHY I'M CONFIDENT:
✓ Universal Rule #3 matched (90% confidence)
✓ Discovered pattern shows 78% win rate
✓ ML model predicts 82% probability

WHAT TO DO:
1. 🔥 Schedule executive demo within 7 days [HIGH]
2. Send competitive comparison guide [MEDIUM]
3. Engage technical buyer [MEDIUM]

Expected outcome: $61K deal, 45 day close cycle
```

---

## 🔧 Configuration

### Data Sources (input_data/)
Place your CSV files in the `input_data/` folder:
- `crm_opportunities.csv`
- `crm_accounts.csv`
- `crm_contacts.csv`
- `crm_activities.csv`
- `intent_signals.csv`
- `corp_email_threads.csv`
- `map_events.csv`
- `zoominfo_people.csv`

### LLM Configuration
**Google Gemini 2.0 Flash is already enabled!** 

To customize, edit `config/llm_config.py`:
```python
LLM_ENABLED = True  # Set to False for template-based explanations
LLM_PROVIDER = 'gemini'  # Options: 'gemini', 'openai', 'anthropic'
GEMINI_API_KEY = "your-api-key"
```

See `GEMINI_SETUP.md` for detailed configuration options.

---

## 📚 Documentation

- **README.md** (this file) - Quick start guide
- **README_NEW_SYSTEM.md** - Comprehensive user guide
- **GEMINI_SETUP.md** - Google Gemini 2.0 Flash configuration
- **IMPLEMENTATION_PLAN.md** - Detailed architecture & design
- **IMPLEMENTATION_SUMMARY.md** - Implementation details
- **CLEANUP_SUMMARY.md** - Project organization changes

---

## 🎯 What's Different from Old System?

| Feature | Old System | New System |
|---------|-----------|------------|
| Rules | 2 simple | **22 comprehensive** |
| Pattern Discovery | None | **Association mining** |
| Explainability | Basic | **Full tracing + SHAP + LLM** |
| Actions | None | **Prioritized recommendations** |
| Accuracy | 96.5% (leakage) | **60-80% (realistic)** |
| Architecture | Monolithic | **5-layer modular** |

---

## 🔄 Workflow

### Training Workflow
1. Load data from `input_data/`
2. Engineer 85+ features
3. Apply 22 universal rules
4. Discover patterns with association mining
5. Train ML model on "messy middle"
6. Generate validation report
7. Save models to `outputs/`

### Scoring Workflow
1. Load trained models
2. Load new opportunity data
3. Engineer features
4. Apply all 5 layers
5. Generate recommendations
6. Create explanations
7. Save results to `outputs/`

---

## 🚨 Troubleshooting

### "Insufficient data for ML training"
- Normal if rules/patterns handle most cases
- System will still work with rules + patterns only

### "No patterns discovered"
- Try lowering `min_support` or `min_confidence`
- Check if you have enough won deals (need ≥30)

### "Performance below target"
- Check for data quality issues
- Review feature engineering
- Adjust rule thresholds

---

## 📞 Support

For questions:
1. Check validation report: `outputs/validation_reports/validation_report.json`
2. Run tests: `python test_system.py`
3. Review documentation in this folder

---

## ✅ System Status

**Status:** Production Ready  
**Version:** 1.0  
**Architecture:** 5-Layer Hybrid (Rules + Patterns + ML + Ensemble + LLM)  
**Last Updated:** November 8, 2025

---

**Built with:** Python, Pandas, Scikit-learn, LightGBM, SHAP, MLxtend

