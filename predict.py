import re
import joblib
import pandas as pd
from urllib.parse import urlparse

# 1. Load the saved model and feature blueprint
print("Loading PhishGuard model...")
saved_artifact = joblib.load("phishguard_model.joblib")
model = saved_artifact["model"]
model_features = saved_artifact["features"]

def extract_features_from_url(url: str) -> pd.DataFrame:
    """Extracts URL-only lexical metrics aligned with PhiUSIIL distributions."""
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    
    # 1. IP Check
    is_ip = int(bool(re.search(r"^(?:\d{1,3}\.){3}\d{1,3}$", domain)))
    
    # 2. Subdomains: Normalize apex domains (e.g. google.com vs www.google.com)
    dots_in_domain = domain.count(".")
    if is_ip:
        subdomains = 0
    else:
        # If no www prefix, count apex domain as 1 logical base subdomain
        subdomains = dots_in_domain if domain.startswith("www.") else max(1, dots_in_domain)
    
    # 3. Character counts & lengths
    letters = len(re.findall(r"[a-zA-Z]", url))
    digits = len(re.findall(r"\d", url))
    special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", url))
    
    total_len = max(len(url), 1)
    letter_ratio = letters / total_len
    digit_ratio = digits / total_len
    special_ratio = special_chars / total_len
    
    tld = domain.split(".")[-1].lower() if "." in domain else ""
    is_https = int(parsed.scheme.lower() == "https")
    
    # Common trusted public TLDs
    trusted_tlds = {"com", "org", "net", "edu", "gov", "de", "io", "uk", "ca"}
    is_trusted_tld = tld in trusted_tlds

    # 4. Feature heuristics matching PhiUSIIL baseline values
    if is_ip:
        similarity_index = 0.0
        tld_prob = 0.0
        char_prob = 0.1
        continuation_rate = 1.0
    else:
        similarity_index = 100.0 if is_trusted_tld else 50.0
        tld_prob = 0.98 if is_trusted_tld else 0.40
        char_prob = 0.90 if is_https else 0.40
        continuation_rate = 0.05
        
    raw_features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "TLDLength": len(tld),
        "NoOfSubDomain": subdomains,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": special_chars,
        "SpacialCharRatioInURL": special_ratio,
        "IsHTTPS": is_https,
        "CharContinuationRate": continuation_rate,
        "URLSimilarityIndex": similarity_index,
        "TLDLegitimateProb": tld_prob,
        "URLCharProb": char_prob,
        "HasObfuscation": int("%" in url or "@" in url),
        "NoOfObfuscatedChar": url.count("%") + url.count("@"),
        "ObfuscationRatio": (url.count("%") + url.count("@")) / total_len,
    }
    
    feature_df = pd.DataFrame([raw_features])
    for col in model_features:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
            
    return feature_df[model_features]
# Trusted root domains that skip model inference if clean
TRUSTED_APEX_DOMAINS = {
    "google.com", "google.de", "youtube.com", "apple.com", "microsoft.com",
    "github.com", "amazon.com", "wikipedia.org", "linkedin.com", "netflix.com"
}

def test_url(url: str):
    """Predicts whether a URL is Safe or Phishing with Tier 1 + Tier 2 validation."""
    clean_url = url.strip().strip("<>[]()\"'")
    formatted_url = clean_url if "://" in clean_url else "http://" + clean_url
    
    parsed = urlparse(formatted_url)
    domain = (parsed.hostname or "").lower()
    
    # Strip leading 'www.' for domain matching
    root_domain = domain[4:] if domain.startswith("www.") else domain
    
    # TIER 1: Allowlist Verification
    # (Matches root domain, requires HTTPS, and checks for clean paths)
    if root_domain in TRUSTED_APEX_DOMAINS and parsed.scheme.lower() == "https":
        # Ensure no hidden at-symbol attacks (e.g., https://google.com@evil.com)
        if "@" not in clean_url and not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", clean_url):
            print("\n" + "-" * 50)
            print(f"URL:        {clean_url}")
            print("Status:     SAFE (Legitimate - Verified Domain)")
            print("Confidence: 100.00%")
            print("-" * 50)
            return

    # TIER 2: Machine Learning Inference (Random Forest)
    try:
        features = extract_features_from_url(formatted_url)
    except Exception as e:
        print(f"\n[ERROR] Could not parse URL '{url}': {e}")
        return

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    class_probs = dict(zip(model.classes_, probabilities))
    
    if prediction == 1:
        status = "SAFE (Legitimate)"
        confidence = class_probs.get(1, 0.0) * 100
    else:
        status = "PHISHING (Suspicious)"
        confidence = class_probs.get(0, 0.0) * 100

    print("\n" + "-" * 50)
    print(f"URL:        {clean_url}")
    print(f"Status:     {status}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Raw Probs: Phishing(0)={class_probs.get(0, 0):.2f}, Safe(1)={class_probs.get(1, 0):.2f})")
    print("-" * 50)
    
if __name__ == "__main__":
    print("\n=== PhishGuard Live URL Tester ===")
    print("Type a URL to test, or type 'exit' to quit.\n")
    
    while True:
        target = input("Enter URL > ").strip()
        if not target or target.lower() in ["exit", "quit", "q"]:
            print("Exiting PhishGuard.")
            break
        test_url(target)