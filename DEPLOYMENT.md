# 🚀 HackRx API Deployment Guide

## 📋 Essential Files for Webhook URL

Only these files are needed for your Railway deployment:

### ✅ **Core Files (Required):**
```
hackrx_faiss_api.py      # Main API application
requirements.txt         # Python dependencies
runtime.txt            # Python runtime version
.gitignore             # Git ignore rules
README.md              # Project documentation
```

### ✅ **Optional but Recommended:**
```
test_webhook.py         # Webhook testing script
```

## 🗑️ **Files to Exclude (Development/Testing):**
```
speed_test.py
fast_embedding_alternatives.py
embedding_comparison.py
dynamic_insurance_terms.py
insurance_coverage_test.py
performance_analysis.py
test_api.py
check_api_keys.py
test_api_key.py
```

## 🔧 **GitHub Commit Steps:**

### **1. Initialize Git (if not already done):**
```bash
git init
git add .
```

### **2. Commit Only Essential Files:**
```bash
# Add only the essential files
git add hackrx_faiss_api.py
git add requirements.txt
git add runtime.txt
git add .gitignore
git add README.md

# Commit
git commit -m "🚀 Deploy HackRx API with ultra-fast keyword hashing and Gemini integration"
```

### **3. Push to GitHub:**
```bash
git remote add origin https://github.com/yourusername/policyQuery.git
git branch -M main
git push -u origin main
```



## 📡 Core APIs

- `POST /ingest-url` – Fetches remote PDF/TXT, chunks & indexes it.
- `GET /kb/docs` – Lists stored documents and chunk counts for UI selection.
- `POST /ask-policy` – Takes a natural language question + doc filters, returns Gemini answer.
- `GET /health` – Simple status endpoint for the frontend badge.