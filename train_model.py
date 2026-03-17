"""
ScamShield — AI Model Training Script
=======================================
Trains all 3 models, picks the best one.
Saves with pickle protocol=4 (Hugging Face compatible).

Dataset: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

Supported CSV files (put ALL in same folder as this script):
  - CEAS_08.csv
  - Enron.csv
  - Ling.csv
  - Nazario.csv
  - Nigerian_Fraud.csv
  - phishing_email.csv
  - SpamAssasin.csv

Run:
  python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.utils import resample
import scipy.sparse as sp

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════
CONFIG = {
    "model_output"      : "scamshield_model.pkl",
    "vectorizer_output" : "scamshield_vectorizer.pkl",
    "metadata_output"   : "scamshield_metadata.json",
    "report_output"     : "training_report.txt",
    "test_size"         : 0.2,
    "random_state"      : 42,
    "max_features"      : 15000,
    "ngram_range"       : (1, 3),
    "max_samples"       : 50000,
}

# ══════════════════════════════════════════════════
# DATASET COLUMN MAPPINGS
# ══════════════════════════════════════════════════
DATASET_CONFIGS = {
    "CEAS_08.csv"       : {"text_cols": ["body"],          "label_col": "label", "label_map": {1:1, 0:0}},
    "Enron.csv"         : {"text_cols": ["body"],          "label_col": "label", "label_map": {1:1, 0:0}},
    "Ling.csv"          : {"text_cols": ["body"],          "label_col": "label", "label_map": {1:1, 0:0}},
    "Nazario.csv"       : {"text_cols": ["body"],          "label_col": "label", "label_map": {1:1, 0:0}},
    "Nigerian_Fraud.csv": {"text_cols": ["body"],          "label_col": "label", "label_map": {1:1, 0:0}},
    "phishing_email.csv": {"text_cols": ["text_combined"], "label_col": "label", "label_map": {1:1, 0:0}},
    "SpamAssasin.csv"   : {"text_cols": ["body"],          "label_col": "label", "label_map": {1:1, 0:0}},
}

# ══════════════════════════════════════════════════
# SYNTHETIC EMAILS
# ══════════════════════════════════════════════════
LEGIT_EMAILS = [
    "Your Amazon order #112-3456789 has shipped. Estimated delivery Tuesday March 19. Track your package at amazon.com/your-orders. Thank you for shopping with us.",
    "Your order from Amazon has been delivered. Your package was left at your front door. If you have any questions contact Amazon customer service.",
    "Hello, your order #445-9921234 is on its way. Expected delivery in 2-3 business days. You can track your shipment on our website.",
    "Your FedEx shipment is out for delivery today. Tracking number 7489234823. Delivery window 10am-2pm. No signature required.",
    "UPS delivery update: Your package is scheduled for delivery today by 8pm. Tracking number 1Z999AA10123456784.",
    "Your DHL express shipment has arrived at the local facility. Delivery attempt will be made tomorrow between 9am and 5pm.",
    "Thank you for your purchase. Your order has been confirmed and will be dispatched within 1-2 business days.",
    "Order confirmation: We have received your order and it is being processed. You will be notified when it ships.",
    "Your package could not be delivered today as no one was available. Please reschedule your delivery at ups.com.",
    "Your subscription to Amazon Prime has been renewed for another year. Your card ending in 4242 has been charged $139.",
    "Refund processed: We have refunded $49.99 to your original payment method. Please allow 3-5 business days.",
    "Your Flipkart order has been shipped via Delhivery. Expected delivery by March 20. Track at flipkart.com/orders.",
    "Your Swiggy order is on the way! Your delivery partner Rajesh is heading to your location. Estimated arrival: 25 minutes.",
    "Zomato order confirmed. Your food from Paradise Biryani is being prepared. Estimated delivery time: 35-40 minutes.",
    "Your monthly bank statement for February 2026 is now available. Log in to your online banking portal to view your statement.",
    "Transaction alert: A purchase of $45.99 was made at Starbucks on your card ending 4321 on March 15 at 8:34am.",
    "Your credit card payment of $250.00 has been received and applied to your account. Your new balance is $1,240.50.",
    "Your salary of $3,200.00 has been credited to your account ending 8765 on March 1 2026. Available balance: $4,150.20.",
    "Your SBI account has been credited with Rs 15,000 via NEFT transfer from HDFC Bank on 16-Mar-2026.",
    "Your HDFC credit card bill of Rs 8,450 is due on March 25. Pay now at hdfcbank.com to avoid late fees.",
    "Your fixed deposit of Rs 50,000 has matured. The amount has been credited to your savings account ending 3421.",
    "Your auto payment of $89.99 for Netflix was successfully processed on March 1. Your subscription is active.",
    "Your pull request #142 Fix login bug has been successfully merged into the main branch by a collaborator.",
    "GitHub Actions: Your workflow CI pipeline passed all 24 checks on branch main. Deployment to production completed successfully.",
    "New issue opened on your repository: Issue #89 - API rate limit not returning correct headers. Assigned to you.",
    "Your GitHub Copilot subscription has been renewed. You can manage billing at github.com/settings/billing.",
    "Two-factor authentication was successfully enabled on your GitHub account. This adds an extra layer of security.",
    "Your npm package express-validator version 3.2.1 has been successfully published to the npm registry.",
    "Stack Overflow: Your answer on How to center a div in CSS received 47 upvotes and has been marked as accepted.",
    "Your flight booking is confirmed. PNR: ABC123. IndiGo flight 6E-204 from Hyderabad to Mumbai on March 20 at 07:30.",
    "Your hotel reservation at Marriott Hyderabad is confirmed. Check-in: March 20, Check-out: March 22.",
    "Booking confirmation: Your Ola cab is scheduled for March 19 at 6:00 AM from Hitech City to Rajiv Gandhi Airport.",
    "Your IRCTC ticket is booked. Train 12723 Telangana Express. PNR 4128394721. Journey date: 20-Mar-2026. Seat: S4 45.",
    "MakeMyTrip: Your bus from Hyderabad to Bangalore is confirmed. Departure March 20 at 9:00 PM. Seat 14A.",
    "Your Spotify Premium subscription has been renewed for Rs 119. Your next billing date is April 15 2026.",
    "Your Microsoft 365 subscription has been renewed. Thank you for your continued subscription. No action needed.",
    "Your Notion Plus plan has been renewed for $16. You can manage your subscription at notion.so/settings.",
    "Your Adobe Creative Cloud subscription renews on April 1. Your payment method on file will be charged $54.99.",
    "Your Zoom Pro plan payment of $14.99 has been processed. Your plan is active through April 2026.",
    "Receipt from Apple: You purchased Procreate for $12.99. Your Apple ID is used@icloud.com.",
    "Your Google One storage plan (200 GB) has been renewed for Rs 130. Storage is shared across Gmail Drive and Photos.",
    "LinkedIn: John Smith has accepted your connection request. You are now connected with 342 people.",
    "Your tweet received 1,200 impressions this week. Top tweet: your post about machine learning got 45 likes.",
    "Instagram: Your photo received 234 likes. Keep sharing moments with your followers.",
    "You have a new match on LinkedIn Jobs. Software Engineer at Microsoft in Hyderabad matches your profile.",
    "Your Glassdoor job alert: 5 new Software Engineer jobs in Hyderabad matching your search criteria.",
    "Your appointment with Dr. Sharma is confirmed for March 20 at 11:00 AM at Apollo Hospital Hyderabad.",
    "Your electricity bill for February is Rs 1,240. Due date: March 25. Pay at tsredco.telangana.gov.in.",
    "Your Airtel postpaid bill of Rs 499 is ready. Auto payment will be processed on March 20 from your saved card.",
    "BSNL: Your broadband plan has been renewed for another month. Speed: 100 Mbps. Valid till April 15.",
    "Your Coursera certificate for Machine Learning Specialization is ready. Download at coursera.org/certificates.",
    "Udemy: Your course Python for Beginners has been updated with 3 new lectures on data visualization.",
    "Your exam results for Semester 4 are now available on the student portal. Login to view your grades.",
    "Your application to IIT Hyderabad M.Tech program has been received. You will hear back within 4 weeks.",
    "Receipt: You spent Rs 340 at Cafe Coffee Day on March 15 at 3:45 PM. Payment via Google Pay.",
    "PhonePe: You sent Rs 500 to Ramesh Kumar (9876543210) successfully on March 16 at 2:14 PM.",
    "Google Pay: Payment of Rs 1,200 to BigBasket was successful. Order #BB29341 confirmed.",
    "Paytm: Your recharge of Rs 239 for 9876543210 was successful. Validity: 28 days. Data: 1.5GB/day.",
    "Your Ola Money wallet has been credited with Rs 100 as a cashback reward. Valid for 30 days.",
    "Your CRED coins 2,500 have been credited for your credit card payment. Redeem at cred.club.",
    "Meeting reminder: Team standup tomorrow at 10:00 AM IST on Google Meet. Agenda: sprint review and planning.",
    "Your Zoom meeting has been scheduled for March 20 at 2:00 PM with 5 participants.",
    "Your leave request for March 20-22 has been approved by your manager. Enjoy your time off.",
    "Your performance review for Q1 2026 is scheduled with HR on March 25 at 3 PM.",
    "Slack: You have 3 unread messages in the engineering channel from your teammates.",
    "Your Jira ticket PROJ-1234 has been assigned to you. Priority: High. Due date: March 22.",
]

SPAM_EMAILS = [
    "URGENT! You have won $1,000,000 in the international lottery! Claim your prize immediately by sending your bank account details and a processing fee of $500 via Western Union. Act now before your prize expires!",
    "Dear beneficiary, I am the personal attorney to a deceased client who left $15.5 million dollars. I need your assistance to transfer these funds and you will receive 30% as compensation. Reply with your bank details immediately.",
    "CONGRATULATIONS! Your email has been selected as the lucky winner. You have won an iPhone 15 Pro. Click here to claim your free gift now. Limited time offer expires today!",
    "Your PayPal account has been suspended due to suspicious activity. Verify your identity immediately at paypal-secure-login.xyz or your account will be permanently closed within 24 hours.",
    "Dear customer, we detected unusual login to your account. Confirm your password and credit card number immediately to restore access. Failure to respond will result in account termination.",
    "Make $5000 per week working from home! No experience needed. Guaranteed income. Send your social security number and bank routing number to get started today. Risk free opportunity!",
    "IRS FINAL NOTICE: You owe back taxes and will be arrested if you do not pay immediately via gift cards. Call this number now to avoid legal action and criminal charges.",
    "Your computer is infected with viruses! Call our toll free number immediately to speak with a Microsoft certified technician. Do not turn off your computer. Act now!",
    "Dear account holder, your bank account has been compromised. Wire transfer $200 processing fee via Bitcoin to unlock your frozen funds of $45,000 held in escrow.",
    "Congratulations dear friend! You have been selected to receive a Nigerian government inheritance fund of $8.5 million dollars. Send your details to claim your share immediately.",
]

# ══════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════
SCAM_KEYWORDS = [
    "urgent", "immediately", "act now", "limited time", "winner",
    "congratulations", "won", "lottery", "prize", "claim",
    "verify account", "suspended", "blocked", "locked",
    "bank details", "wire transfer", "bitcoin", "gift card",
    "inheritance", "million dollars", "free money", "verify now",
    "password", "ssn", "social security", "legal action",
    "arrest", "irs", "dear customer", "dear user",
    "dear beneficiary", "final notice", "last chance",
    "guaranteed", "work from home", "earn money",
    "dear friend", "transfer funds", "account holder"
]
URGENCY_WORDS = [
    "urgent", "immediately", "asap", "today", "24 hours",
    "48 hours", "expire", "deadline", "hurry", "last chance",
    "limited", "act fast", "final", "warning", "alert",
    "critical", "emergency", "respond now"
]
FINANCIAL_WORDS = [
    "money", "cash", "dollars", "bitcoin", "transfer", "bank",
    "payment", "invest", "profit", "earn", "income", "salary",
    "lottery", "prize", "winning", "grant", "inheritance",
    "million", "thousand", "refund", "compensation"
]
LEGIT_INDICATOR_WORDS = [
    "order", "shipped", "delivered", "tracking", "confirmed",
    "receipt", "invoice", "subscription", "renewed", "billing",
    "appointment", "scheduled", "booked", "reservation",
    "statement", "credited", "debited", "transaction",
    "your account", "manage your", "customer service",
    "pull request", "merged", "repository", "commit",
    "meeting", "reminder", "standup", "leave approved",
    "recharge", "successful", "processed successfully"
]


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' URL_TOKEN ', text)
    text = re.sub(r'www\.\S+', ' URL_TOKEN ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', text)
    text = re.sub(r'[^\w\s!?$.,]', ' ', text)
    return text.strip().lower()


def extract_features(texts):
    records = []
    for text in texts:
        t     = text.lower()
        words = t.split()
        records.append({
            "char_count"           : len(text),
            "word_count"           : len(words),
            "avg_word_len"         : np.mean([len(w) for w in words]) if words else 0,
            "caps_word_count"      : sum(1 for w in text.split() if w.isupper() and len(w) > 2),
            "caps_ratio"           : sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "exclamation_count"    : text.count("!"),
            "question_count"       : text.count("?"),
            "dollar_count"         : text.count("$"),
            "scam_keyword_count"   : sum(1 for kw in SCAM_KEYWORDS if kw in t),
            "urgency_count"        : sum(1 for w in URGENCY_WORDS if w in t),
            "financial_count"      : sum(1 for w in FINANCIAL_WORDS if w in t),
            "has_http_link"        : 1 if "url_token" in t else 0,
            "has_short_url"        : 1 if any(x in t for x in ["bit.ly","tinyurl","goo.gl","rb.gy"]) else 0,
            "has_dear_customer"    : 1 if "dear customer" in t else 0,
            "has_dear_user"        : 1 if "dear user" in t else 0,
            "has_dear_beneficiary" : 1 if "dear beneficiary" in t else 0,
            "has_account_holder"   : 1 if "account holder" in t else 0,
            "number_count"         : sum(1 for w in words if any(c.isdigit() for c in w)),
            "has_large_number"     : 1 if any(x in t for x in ["million","billion","thousand","lakh","crore"]) else 0,
            "has_threat"           : 1 if any(x in t for x in ["arrest","lawsuit","legal action","court","criminal"]) else 0,
            "has_password_req"     : 1 if any(x in t for x in ["password","pin","credential","login"]) else 0,
            "has_email_token"      : 1 if "email_token" in t else 0,
            "legit_indicator_count": sum(1 for w in LEGIT_INDICATOR_WORDS if w in t),
            "has_order_number"     : 1 if re.search(r'#[\w\-]{5,}', text) else 0,
            "has_tracking_number"  : 1 if re.search(r'\b[A-Z0-9]{10,}\b', text) else 0,
            "has_date_reference"   : 1 if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}', t) else 0,
            "has_rs_amount"        : 1 if re.search(r'rs\.?\s*[\d,]+', t) else 0,
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════
# DATASET LOADER
# ══════════════════════════════════════════════════
def load_single_dataset(filepath, cfg):
    try:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')

        df.columns = df.columns.str.strip().str.lower()
        text_cols  = [c.lower() for c in cfg["text_cols"]]
        label_col  = cfg["label_col"].lower()

        available_text = [c for c in text_cols if c in df.columns]
        if not available_text:
            candidates = [c for c in df.columns if df[c].dtype == object and c != label_col]
            if not candidates:
                print(f"      ⚠ No text column found.")
                return None
            available_text = [candidates[0]]

        if label_col not in df.columns:
            lbl_candidates = [c for c in df.columns if df[c].nunique() <= 10 and c not in available_text]
            if not lbl_candidates:
                return None
            label_col = lbl_candidates[0]

        df["_text"] = df[available_text].fillna("").astype(str).apply(
            lambda row: " ".join(row.values), axis=1
        )

        if df[label_col].dtype == object:
            lbl   = df[label_col].str.lower().str.strip()
            phish = {'spam','phishing','scam','fraud','1','true','yes','malicious'}
            legit = {'ham','legitimate','safe','0','false','no','benign','normal'}
            df["_label"] = lbl.apply(lambda x: 1 if x in phish else (0 if x in legit else None))
        else:
            df["_label"] = df[label_col].map(cfg.get("label_map", {1:1, 0:0}))

        df = df.dropna(subset=["_text","_label"])
        df["_label"] = df["_label"].astype(int)
        df = df[df["_text"].str.len() > 10]

        if len(df) > CONFIG["max_samples"]:
            df = df.sample(CONFIG["max_samples"], random_state=42)

        return df[["_text","_label"]].rename(columns={"_text":"text","_label":"label"})

    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def load_all_datasets():
    print(f"\n{'='*60}")
    print("  SCAMSHIELD — AI MODEL TRAINER")
    print(f"{'='*60}\n")
    print("[1/6] Scanning for dataset files...\n")

    all_dfs      = []
    loaded_files = []

    for filename, cfg in DATASET_CONFIGS.items():
        if not os.path.exists(filename):
            print(f"    ⏭  '{filename}' — not found, skipping")
            continue
        size_kb = os.path.getsize(filename) // 1024
        print(f"    📂 '{filename}' ({size_kb:,} KB) — loading...")
        df = load_single_dataset(filename, cfg)
        if df is not None:
            spam  = (df["label"] == 1).sum()
            legit = (df["label"] == 0).sum()
            print(f"      ✓ {len(df):,} rows  |  Phishing: {spam:,}  |  Legit: {legit:,}")
            all_dfs.append(df)
            loaded_files.append(filename)
        print()

    print(f"    📂 Injecting synthetic training samples...")
    legit_synthetic = pd.DataFrame({"text": LEGIT_EMAILS * 5, "label": [0] * (len(LEGIT_EMAILS) * 5)})
    spam_synthetic  = pd.DataFrame({"text": SPAM_EMAILS  * 3, "label": [1] * (len(SPAM_EMAILS)  * 3)})
    all_dfs.append(pd.concat([legit_synthetic, spam_synthetic], ignore_index=True))
    print(f"      ✓ {len(legit_synthetic)} legit + {len(spam_synthetic)} spam synthetic samples\n")

    if not all_dfs:
        raise ValueError("❌ No CSV files found! Place CSVs in the same folder as this script.")

    combined    = pd.concat(all_dfs, ignore_index=True)
    combined    = combined.drop_duplicates(subset=["text"]).dropna(subset=["text","label"])
    phishing_df = combined[combined["label"] == 1]
    legit_df    = combined[combined["label"] == 0]

    print(f"    ══ COMBINED DATASET ══")
    print(f"    Total      : {len(combined):,}")
    print(f"    Phishing   : {len(phishing_df):,} ({len(phishing_df)/len(combined)*100:.1f}%)")
    print(f"    Legitimate : {len(legit_df):,} ({len(legit_df)/len(combined)*100:.1f}%)")

    ratio = max(len(phishing_df), len(legit_df)) / max(min(len(phishing_df), len(legit_df)), 1)
    if ratio > 3:
        print(f"\n    ⚖  Balancing classes (ratio {ratio:.1f}x)...")
        min_cls = min(len(phishing_df), len(legit_df))
        target  = min(min_cls * 2, max(len(phishing_df), len(legit_df)))
        if len(phishing_df) > len(legit_df):
            phishing_df = resample(phishing_df, n_samples=target, random_state=42)
        else:
            legit_df = resample(legit_df, n_samples=target, random_state=42)
        combined = pd.concat([phishing_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"    ✓  Balanced to {len(combined):,} rows")

    return combined, loaded_files


# ══════════════════════════════════════════════════
# MODEL TRAINING — all 3, best one wins
# ══════════════════════════════════════════════════
def train_and_select(X_combined, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight="balanced",
            max_iter=2000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
    }

    results = {}
    print("\n[4/6] Training all 3 models with 5-fold cross validation...\n")
    for name, model in models.items():
        print(f"    ⏳ Training {name}...")
        scores = cross_val_score(model, X_combined, y_train, cv=5, scoring="f1", n_jobs=-1)
        results[name] = {"model": model, "cv_f1": scores.mean(), "cv_std": scores.std()}
        print(f"    ✓  CV F1: {scores.mean():.4f} ± {scores.std():.4f}\n")

    print(f"    {'─'*42}")
    print(f"    {'Model':<22} {'CV F1':>8}  {'Std':>6}")
    print(f"    {'─'*42}")
    for name, res in results.items():
        print(f"    {name:<22} {res['cv_f1']:>8.4f}  {res['cv_std']:>6.4f}")
    print(f"    {'─'*42}")

    best_name  = max(results, key=lambda k: results[k]["cv_f1"])
    best_model = results[best_name]["model"]
    best_model.fit(X_combined, y_train)
    print(f"\n    ★  Best model → {best_name}  (F1 = {results[best_name]['cv_f1']:.4f})")
    return best_model, best_name


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def main():
    df, loaded_files = load_all_datasets()

    print("\n[2/6] Cleaning and normalizing text...")
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    print(f"    ✓ Clean samples: {len(df):,}")

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"], stratify=y
    )
    print(f"\n[3/6] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n       Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG["max_features"],
        ngram_range=CONFIG["ngram_range"],
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"    ✓ Vocabulary: {len(vectorizer.vocabulary_):,} terms")

    print("       Extracting handcrafted features...")
    X_train_feat   = extract_features(X_train)
    X_test_feat    = extract_features(X_test)
    X_train_feat_s = sp.csr_matrix(X_train_feat.values.astype(float))
    X_test_feat_s  = sp.csr_matrix(X_test_feat.values.astype(float))
    X_train_comb   = sp.hstack([X_train_tfidf, X_train_feat_s])
    X_test_comb    = sp.hstack([X_test_tfidf,  X_test_feat_s])
    print(f"    ✓ {len(X_train_feat.columns)} handcrafted features per sample")

    best_model, best_name = train_and_select(X_train_comb, y_train)

    print("\n[5/6] Evaluating on test set...")
    y_pred = best_model.predict(X_test_comb)

    try:
        y_prob = best_model.predict_proba(X_test_comb)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
    except AttributeError:
        try:
            dec    = best_model.decision_function(X_test_comb)
            y_prob = 1 / (1 + np.exp(-dec))
            auc    = roc_auc_score(y_test, y_prob)
        except:
            auc = None

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate","Phishing"])

    print(f"\n    Accuracy : {acc*100:.2f}%")
    print(f"    F1 Score : {f1:.4f}")
    if auc: print(f"    ROC-AUC  : {auc:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"               Pred Legit  Pred Phish")
    print(f"    True Legit  {cm[0][0]:^10,} {cm[0][1]:^10,}")
    print(f"    True Phish  {cm[1][0]:^10,} {cm[1][1]:^10,}")
    print(f"\n    Full Report:")
    for line in report.split("\n"):
        print(f"    {line}")

    # ── SAVE WITH PROTOCOL=4 ──
    # protocol=4 is compatible with Python 3.8+
    # Fixes KeyError: 118 crash on Hugging Face / Railway
    print(f"\n[6/6] Saving artifacts (protocol=4 for Hugging Face compatibility)...")
    joblib.dump(best_model, CONFIG["model_output"],      protocol=4)
    joblib.dump(vectorizer, CONFIG["vectorizer_output"], protocol=4)
    print(f"    ✓ {CONFIG['model_output']}  ({os.path.getsize(CONFIG['model_output'])//1024/1024:.1f} MB)")
    print(f"    ✓ {CONFIG['vectorizer_output']}  ({os.path.getsize(CONFIG['vectorizer_output'])//1024:.0f} KB)")

    metadata = {
        "model_name"           : best_name,
        "accuracy"             : round(acc, 4),
        "f1_score"             : round(f1, 4),
        "roc_auc"              : round(auc, 4) if auc else None,
        "train_samples"        : int(len(X_train)),
        "test_samples"         : int(len(X_test)),
        "total_samples"        : int(len(df)),
        "vocabulary_size"      : len(vectorizer.vocabulary_),
        "handcrafted_features" : int(len(X_train_feat.columns)),
        "ngram_range"          : list(CONFIG["ngram_range"]),
        "classes"              : ["Legitimate (0)", "Phishing (1)"],
        "datasets_used"        : loaded_files,
        "synthetic_samples"    : f"{len(LEGIT_EMAILS)*5} legit + {len(SPAM_EMAILS)*3} spam",
        "pickle_protocol"      : 4,
    }
    with open(CONFIG["metadata_output"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✓ {CONFIG['metadata_output']}")

    with open(CONFIG["report_output"], "w") as f:
        f.write("SCAMSHIELD MODEL TRAINING REPORT\n")
        f.write("="*55 + "\n\n")
        f.write(f"Best Model   : {best_name}\n")
        f.write(f"Accuracy     : {acc*100:.2f}%\n")
        f.write(f"F1 Score     : {f1:.4f}\n")
        if auc: f.write(f"ROC-AUC      : {auc:.4f}\n")
        f.write(f"Total Samples: {len(df):,}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"             Pred Legit  Pred Phish\n")
        f.write(f"True Legit   {cm[0][0]:^10,} {cm[0][1]:^10,}\n")
        f.write(f"True Phish   {cm[1][0]:^10,} {cm[1][1]:^10,}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  Best Model : {best_name}")
    print(f"  Accuracy   : {acc*100:.2f}%  |  F1: {f1:.4f}")
    print(f"{'='*60}")
    print(f"\n  Upload these 3 files to Hugging Face:")
    print(f"    ✓ {CONFIG['model_output']}")
    print(f"    ✓ {CONFIG['vectorizer_output']}")
    print(f"    ✓ {CONFIG['metadata_output']}")
    print(f"\n  No need to run resave_model.py — protocol=4 already applied!\n")


if __name__ == "__main__":
    main()