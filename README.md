# PhishGuard 🛡️

A lightweight, pre-click URL phishing detection system built with machine learning and a two-tier defensive verification architecture.

## Overview
PhishGuard identifies malicious links in real time without requiring browser downloads or DOM fetching (pre-click protection). It extracts lexical signals directly from URL strings, cross-validates via an apex allowlist, and routes unknown links through a trained Random Forest classifier.

## Architecture
1. **Tier 1 (Instant Allowlist Verification):** Bypasses ML inference for trusted, clean apex domains (e.g., Google, Apple, GitHub) over HTTPS to ensure 0% false positives on critical services.
2. **Tier 2 (Lexical ML Inference):** Evaluates unverified or unfamiliar URLs using a Random Forest model trained on URL-only structural and statistical distributions.

## Performance & Evaluation
Trained and evaluated on the **PhiUSIIL** phishing dataset (~235k URLs) using URL-only lexical features:
* **Accuracy:** 99.99%
* **ROC-AUC:** 0.9999
* **Phishing Recall:** 100.00% (26,970 / 26,970 caught)
* **False Positives:** 7 / 20,189

## Project Structure
* `predict.py` - Two-tier live URL prediction engine
* `train_rf.py` - Random Forest model training pipeline
* `phishguard_model.joblib` - Serialized model and feature weights
* `requirements.txt` - Python dependencies
* `README.md` - Documentation
