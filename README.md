# Pipeline Builder - Enhanced Hybrid POC

**5-Layer AI System for Sales Opportunity Prediction & Recommendations**

---

## 🚀 Quick Start

### 1. Set Up Environment

**Activate the virtual environment:**
```bash
# Windows
myenv\Scripts\activate.bat

# PowerShell
myenv\Scripts\Activate.ps1
```

**Install dependencies (if needed):**
```bash
pip install pandas numpy scikit-learn lightgbm shap mlxtend joblib google-generativeai python-dotenv
```

### 2. Configure API Keys

**Create your `.env` file:**
```bash
cp .env.example .env  # Copy template
# Edit .env with your actual API keys
```

**Required for LLM features:**
- `GEMINI_API_KEY` - Google Gemini API key
- Optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

### 3. Train the System
```bash
python main_train_pipeline.py
```

### 4. Score Opportunities
```bash
python main_score_pipeline.py
```

### 5. Run Tests
```bash
python test_system.py
```

---

## 📁 Key Directories

- **`input_data/`** - Place your CSV data files here
- **`config/`** - Configuration files and business rules
- **`layers/`** - 5-layer AI system (Rules, Patterns, ML, Ensemble, LLM)
- **`outputs/`** - Generated models, patterns, and recommendations
- **`.env`** - API keys (create from `.env.example`)

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

- **🎯 22 Expert Rules** - High-precision patterns for sales opportunities
- **🔍 Pattern Discovery** - Automatic mining of winning combinations
- **🧠 ML Ensemble** - LightGBM + SHAP for complex cases
- **💬 AI Explanations** - Natural language summaries via Google Gemini
- **⚡ Rate Limiting** - Automatic API throttling (10 req/min)
- **🔒 Security** - API keys in `.env`, never committed to git

---

## 📤 Sample Output

**AI-Generated Explanation:**
```
Okay, team, we've got a HIGH-SURGE NEW OPPORTUNITY that we need to jump on FAST!

I'm 88% confident this is a hot lead because this prospect is showing strong "surge signals" and a new logo. This triggers our "High-Surge Quick Strike" rule, indicating a high likelihood of conversion if we act fast.

Here's what we need to do RIGHT NOW:
1. CRITICAL: Respond immediately (within 1 hour)! Surge signals demand immediate attention (+45% conversion)
2. CRITICAL: Send a personalized urgent proposal ASAP! High intent demands an immediate offer (+40% conversion)
3. CRITICAL: Schedule an emergency demo within 4 hours! A fast demo is crucial for conversion (+35% conversion)

Expected outcome: $50,000 deal, 30 day close cycle
```

*Powered by Google Gemini 2.0 Flash with automatic rate limiting*

---

## 🔧 Configuration

**Data Setup:** Place your CSV files in the `input_data/` folder.

**LLM Setup:** Copy `.env.example` to `.env` and add your Gemini API key.

**Security:** API keys are stored securely in `.env` and never committed to git.

---


---

## 🔄 How It Works

**Training:** Process historical data → Discover patterns → Train ML models
**Scoring:** Load new opportunities → Apply all 5 layers → Generate AI recommendations

---

## 🚨 Common Issues

- **ML training insufficient**: Normal - system works with rules/patterns only
- **No patterns found**: Check data quality or adjust thresholds in config
- **API errors**: Verify `.env` file and API key validity

---

## 📞 Support

Check `outputs/validation_reports/` for system diagnostics or run `python test_system.py`.

---

## ✅ System Status

**Status:** ✅ **Fully Operational & Production Ready**  
**Version:** 2.0 (LLM Integration Complete)  
**Architecture:** 5-Layer Hybrid (Rules + Patterns + ML + Ensemble + LLM)  
**Last Updated:** November 20, 2025

**Key Achievements:**
- ✅ **LLM Layer Working**: Google Gemini 2.0 Flash integrated
- ✅ **Rate Limiting**: API throttling implemented (10 req/min)
- ✅ **Security Hardened**: API keys in `.env`, git-ignored
- ✅ **Unicode Fixed**: Windows console compatibility
- ✅ **End-to-End Pipeline**: Complete scoring system operational
- ✅ **AI Explanations**: Natural language generation working

---

**Built with:** Python 3.13.1, Pandas, Scikit-learn, LightGBM, SHAP, MLxtend, Google Generative AI, python-dotenv

