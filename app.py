"""
ScamShield — Flask API Backend
================================
Serves the trained ML model as a REST API.

Requirements:
    pip install flask flask-cors scikit-learn pandas numpy joblib scipy

Usage:
    python app.py

Endpoints:
    POST /api/analyze   → Analyze an email
    GET  /api/health    → Health check
    GET  /api/model-info → Model metadata
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import scipy.sparse as sp
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS ON STARTUP
# ─────────────────────────────────────────────
MODEL_PATH      = "scamshield_model1.pkl"
VECTORIZER_PATH = "scamshield_vectorizer1.pkl"
METADATA_PATH   = "scamshield_metadata1.json"

print("\n[ScamShield API] Loading model artifacts...")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "\n❌ Model files not found!\n"
        "   Please run: python train_model.py\n"
        "   Then start this API again.\n"
    )

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
metadata   = json.load(open(METADATA_PATH)) if os.path.exists(METADATA_PATH) else {}

print(f"[ScamShield API] ✓ Model loaded      : {metadata.get('model_name', 'Unknown')}")
print(f"[ScamShield API] ✓ Accuracy          : {metadata.get('accuracy', '?')}")
print(f"[ScamShield API] ✓ Vocabulary size   : {metadata.get('vocabulary_size', '?')}")
print(f"[ScamShield API] ✓ Ready on port 5050\n")


# ─────────────────────────────────────────────
# FEATURE ENGINEERING — must match train_model.py exactly
# ─────────────────────────────────────────────
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

# NEW: legit transactional indicators — must match train_model.py
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


def extract_features(text):
    """Extract handcrafted features — must match train_model.py exactly (25 features)."""
    t     = text.lower()
    words = t.split()

    return {
        # ── original 21 features ──
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
        # ── NEW 4 features added in updated train_model.py ──
        "legit_indicator_count": sum(1 for w in LEGIT_INDICATOR_WORDS if w in t),
        "has_order_number"     : 1 if re.search(r'#[\w\-]{5,}', text) else 0,
        "has_tracking_number"  : 1 if re.search(r'\b[A-Z0-9]{10,}\b', text) else 0,
        "has_date_reference"   : 1 if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}', t) else 0,
        "has_rs_amount"        : 1 if re.search(r'rs\.?\s*[\d,]+', t) else 0,
    }


def build_combined_features(text):
    """Build the combined TF-IDF + handcrafted feature matrix."""
    tfidf_vec  = vectorizer.transform([text])
    hand_feats = extract_features(text)
    hand_vec   = sp.csr_matrix(list(hand_feats.values()))
    return sp.hstack([tfidf_vec, hand_vec])


# ─────────────────────────────────────────────
# RULE-BASED SIGNAL EXTRACTOR
# ─────────────────────────────────────────────
RULE_SETS = {
    "urgency": [
        "urgent", "immediately", "act now", "expires today", "limited time",
        "within 24 hours", "within 48 hours", "asap", "last chance",
        "final notice", "account suspended", "account blocked",
        "verify now", "confirm now", "respond immediately",
        "action required", "critical alert", "don't delay"
    ],
    "financial": [
        "you have won", "you won", "lottery", "prize", "million dollars",
        "inheritance", "wire transfer", "western union", "moneygram",
        "bitcoin", "gift card", "itunes card", "google play card",
        "claim your prize", "free money", "routing number", "ssn",
        "social security", "tax refund"
    ],
    "credential": [
        "verify your account", "confirm your identity", "update your password",
        "login credentials", "reset your password", "unusual activity",
        "suspicious login", "click here to verify", "enter your details",
        "confirm your email", "validate your account"
    ],
    "threats": [
        "legal action", "lawsuit", "arrested", "irs",
        "tax fraud", "account will be closed", "service terminated",
        "debt collection", "overdue payment", "criminal charges"
    ],
    "generic_greeting": [
        "dear customer", "dear user", "dear account holder",
        "valued member", "dear beneficiary", "dear sir/madam",
        "dear client", "to whom it may concern"
    ]
}

SUSPICIOUS_LINK_PATTERNS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "rb.gy", "cutt.ly", "is.gd", "tiny.cc",
    ".xyz/", ".tk/", ".ml/", ".ga/", ".cf/",
    "login-verify", "secure-update", "account-confirm", "verify-now",
    "http://"
]

LEGIT_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "protonmail.com", "amazon.com", "google.com", "microsoft.com",
    "apple.com", "paypal.com", "netflix.com", "facebook.com",
    "twitter.com", "linkedin.com", "instagram.com", "github.com",
    "zoho.com", "ymail.com", "live.com", "hdfcbank.com", "sbi.co.in",
    "icicibank.com", "axisbank.com", "zomato.com", "swiggy.com",
    "flipkart.com", "irctc.co.in", "indigo.in", "airtel.in"
]

FAKE_DOMAIN_PATTERNS = [
    "paypa1", "amaz0n", "micros0ft", "g00gle", "faceb00k",
    "app1e", "netf1ix", "secure-login", "account-verify",
    "update-confirm", "paypal-secure", "amazon-verify"
]


def extract_signals(text, sender="", subject=""):
    all_text    = (text + " " + subject).lower()
    signals     = []
    sender_rows = []

    urg_hits = [w for w in RULE_SETS["urgency"] if w in all_text]
    if len(urg_hits) >= 3:
        signals.append({"type": "flag", "icon": "🚨", "name": "Extreme Urgency Language",
                         "desc": f'{len(urg_hits)} pressure phrases: "{", ".join(urg_hits[:3])}"'})
    elif urg_hits:
        signals.append({"type": "warn", "icon": "⚠️", "name": "Urgency Language",
                         "desc": f'Pressure phrase found: "{urg_hits[0]}"'})

    fin_hits = [w for w in RULE_SETS["financial"] if w in all_text]
    if len(fin_hits) >= 2:
        signals.append({"type": "flag", "icon": "💰", "name": "Financial Scam Patterns",
                         "desc": f'Multiple financial keywords: "{", ".join(fin_hits[:2])}"'})
    elif fin_hits:
        signals.append({"type": "warn", "icon": "💵", "name": "Financial Mention",
                         "desc": f'Suspicious keyword: "{fin_hits[0]}"'})

    cred_hits = [w for w in RULE_SETS["credential"] if w in all_text]
    if len(cred_hits) >= 2:
        signals.append({"type": "flag", "icon": "🔑", "name": "Credential Phishing",
                         "desc": f'Requesting login data: "{", ".join(cred_hits[:2])}"'})
    elif cred_hits:
        signals.append({"type": "warn", "icon": "🔐", "name": "Credential Request",
                         "desc": f'Found: "{cred_hits[0]}"'})

    threat_hits = [w for w in RULE_SETS["threats"] if w in all_text]
    if threat_hits:
        signals.append({"type": "flag", "icon": "⚖️", "name": "Threatening Language",
                         "desc": f'Threat patterns: "{", ".join(threat_hits[:2])}"'})

    greet = next((w for w in RULE_SETS["generic_greeting"] if w in all_text), None)
    if greet:
        signals.append({"type": "warn", "icon": "👤", "name": "Generic Greeting",
                         "desc": f'"{greet}" — real companies use your actual name'})

    caps_count = len(re.findall(r'\b[A-Z]{4,}\b', text + " " + subject))
    excl_count = (text + subject).count("!")
    if caps_count >= 3 or excl_count >= 3:
        signals.append({"type": "warn", "icon": "📢", "name": "Aggressive Formatting",
                         "desc": f"{caps_count} ALL-CAPS words and {excl_count} exclamation marks"})

    if sender:
        s = sender.lower().strip()
        parts = s.split("@")
        sender_rows.append({"key": "SENDER ADDRESS", "value": sender, "cls": "neu"})
        if len(parts) == 2:
            domain      = parts[1]
            domain_parts = domain.split(".")
            base_domain  = ".".join(domain_parts[-2:])
            sender_rows.append({"key": "DOMAIN", "value": domain, "cls": "neu"})

            fake = next((p for p in FAKE_DOMAIN_PATTERNS if p in s), None)
            if fake:
                signals.append({"type": "flag", "icon": "📨", "name": "Spoofed Sender Domain",
                                 "desc": f'"{sender}" contains lookalike pattern "{fake}"'})
                sender_rows.append({"key": "STATUS", "value": "⚠ SUSPICIOUS LOOKALIKE", "cls": "bad"})

            if re.search(r'[a-z][0-9]|[0-9][a-z]', domain_parts[0]) and len(domain_parts[0]) > 3:
                signals.append({"type": "flag", "icon": "🔢", "name": "Number-Letter Substitution",
                                 "desc": f'"{domain}" uses numbers to mimic brand letters'})
                sender_rows.append({"key": "CHAR SPOOF", "value": "DETECTED", "cls": "bad"})

            brands = ["paypal", "amazon", "google", "microsoft", "apple",
                      "netflix", "facebook", "instagram", "twitter", "dropbox"]
            brand_found = next((b for b in brands if b in domain and domain not in LEGIT_DOMAINS), None)
            if brand_found:
                signals.append({"type": "flag", "icon": "🎭", "name": "Brand Impersonation",
                                 "desc": f'"{domain}" impersonates "{brand_found}" — real domain is "{base_domain}"'})
                sender_rows.append({"key": "BRAND SPOOF", "value": f"IMPERSONATING {brand_found.upper()}", "cls": "bad"})

            is_legit = domain in LEGIT_DOMAINS
            sender_rows.append({"key": "RECOGNISED DOMAIN",
                                 "value": "YES ✓" if is_legit else "NO — UNVERIFIED",
                                 "cls": "ok" if is_legit else "warn"})
            if is_legit and not fake and not brand_found:
                signals.append({"type": "ok", "icon": "✅", "name": "Sender Domain OK",
                                 "desc": f'"{domain}" is a recognised legitimate domain'})

    url_pattern = re.compile(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', re.IGNORECASE)
    all_links  = list(set(url_pattern.findall(text + " " + subject)))
    bad_links  = [l for l in all_links if any(p in l.lower() for p in SUSPICIOUS_LINK_PATTERNS)]
    good_links = [l for l in all_links if l not in bad_links]

    if bad_links:
        signals.append({"type": "flag", "icon": "🔗", "name": f"{len(bad_links)} Dangerous Link(s)",
                         "desc": "Shortened or suspicious URLs detected — likely phishing redirects"})
    elif good_links:
        signals.append({"type": "ok", "icon": "🔗", "name": "Links Appear Safe",
                         "desc": f"{len(good_links)} link(s) found with no suspicious patterns"})

    if not signals:
        signals.append({"type": "ok", "icon": "✅", "name": "No Scam Patterns Found",
                         "desc": "No known phishing or scam indicators detected"})
        signals.append({"type": "ok", "icon": "🛡️", "name": "Content Appears Clean",
                         "desc": "Language patterns do not match common scam templates"})

    return signals, sender_rows, bad_links, good_links


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status"   : "online",
        "service"  : "ScamShield API",
        "model"    : metadata.get("model_name", "Unknown"),
        "accuracy" : metadata.get("accuracy", "?"),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify(metadata)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data    = request.get_json()
        text    = str(data.get("text",    "")).strip()
        sender  = str(data.get("sender",  "")).strip()
        subject = str(data.get("subject", "")).strip()

        if not text and not sender and not subject:
            return jsonify({"error": "Please provide at least one field: text, sender, or subject"}), 400

        combined_text = f"{subject} {sender} {text}".strip()

        X_combined = build_combined_features(combined_text)
        ml_pred    = int(model.predict(X_combined)[0])

        try:
            ml_prob = float(model.predict_proba(X_combined)[0][1])
        except AttributeError:
            try:
                decision = model.decision_function(X_combined)[0]
                ml_prob  = float(1 / (1 + np.exp(-decision)))
            except:
                ml_prob = 0.9 if ml_pred == 1 else 0.1

        signals, sender_rows, bad_links, good_links = extract_signals(text, sender, subject)

        rule_score = 0
        for sig in signals:
            if sig["type"] == "flag": rule_score += 25
            elif sig["type"] == "warn": rule_score += 12
        rule_score = min(rule_score, 100)

        threat_score = int((ml_prob * 60) + (rule_score * 0.40))
        threat_score = max(0, min(threat_score, 100))

        if threat_score <= 25:
            level   = "safe"
            verdict = "LIKELY SAFE"
            desc    = "No significant scam indicators detected. This email appears to be legitimate."
            advice  = "✅ This email appears safe. No major red flags were detected. However, never share passwords or financial details over email regardless of who it appears to be from."
        elif threat_score <= 60:
            level   = "warn"
            verdict = "SUSPICIOUS EMAIL"
            desc    = "Several warning signs detected. This email shows patterns common in phishing and scam emails."
            advice  = "⚠️ Treat this email with caution. Do NOT click any links. Do NOT provide personal information. If it claims to be from a company, contact that company through their official website — not through this email."
        else:
            level   = "danger"
            verdict = "HIGH THREAT — LIKELY SCAM"
            desc    = "Multiple strong scam indicators detected. Highly consistent with phishing or fraud."
            advice  = "🚨 Do NOT engage with this email. This has a high probability of being a scam or phishing attempt. Do NOT click any links, do NOT reply, do NOT provide any personal or financial information. Mark as spam and delete immediately."

        return jsonify({
            "score"        : threat_score,
            "level"        : level,
            "verdict"      : verdict,
            "description"  : desc,
            "confidence"   : round(ml_prob, 4),
            "ml_prediction": ml_pred,
            "signals"      : signals,
            "sender_rows"  : sender_rows,
            "bad_links"    : bad_links,
            "good_links"   : good_links,
            "advice"       : advice,
            "analyzed_at"  : datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  ScamShield API — Starting...")
    print("="*50)
    print("  Endpoints:")
    print("    POST http://localhost:5000/api/analyze")
    print("    GET  http://localhost:5000/api/health")
    print("    GET  http://localhost:5000/api/model-info")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)