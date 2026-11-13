"""Loads the trained model and turns a parsed email into an evidence object.

Pipeline (paper sections IV, VI and VII):
    parsed email -> TF-IDF + calibrated Logistic Regression  -> p
                 -> deterministic rules                        -> r
                 -> risk aggregation   R = alpha*p + (1-alpha)*r
                 -> severity policy    Safe / Suspicious / Phishing
                 -> advisory actions chosen from the rule ids and severity
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np

import detector

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
MAX_TEXT_CHARS = 5000  # same cut-off used in training

SEVERITY_LABEL = {"safe": "Safe", "suspicious": "Suspicious", "phishing": "Phishing"}


def model_text(p: dict) -> str:
    return f"{p['subject']}\n{p['body']}"[:MAX_TEXT_CHARS]


def severity_from(R: float, flags: list[dict], t_low: float, t_high: float) -> str:
    """Severity policy (paper section VII).

    1. Thresholds on the combined risk R:  R < t_low -> Safe,  R >= t_high -> Phishing,  else Suspicious.
    2. Deterministic rules can raise the level, never lower it: a prediction can be uncertain
       while a single strong rule still warrants caution.
         - two or more high-severity rule hits  -> Phishing
         - one high-severity rule hit           -> at least Suspicious
    """
    level = 2 if R >= t_high else 1 if R >= t_low else 0
    n_high = sum(f["severity"] == "high" for f in flags)
    if n_high >= 2:
        level = 2
    elif n_high == 1:
        level = max(level, 1)
    return ("safe", "suspicious", "phishing")[level]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


# --------------------------------------------------------------------------
# Advisory engine: deterministic, built from structured evidence only.
# --------------------------------------------------------------------------

SEVERITY_ADVICE = {
    "safe": "No strong warning signs were found. Still verify any unexpected request through a channel you already trust.",
    "suspicious": "Some warning signs were found but the evidence is incomplete. Do not click links or open attachments until you have confirmed the sender independently.",
    "phishing": "Several strong warning signs were found. Do not click links, open attachments or reply. Report the message and reach the service by typing its official address yourself.",
}

RULE_ADVICE = {
    "R01_SENDER_LOOKALIKE": "Check the sender using a phone number or website you already know, not the details in this email.",
    "R02_DISPLAY_NAME_MISMATCH": "Check the sender using a phone number or website you already know, not the details in this email.",
    "R03_REPLY_TO_MISMATCH": "Do not reply to this email. Replies would go to a different address than the sender.",
    "R04_AUTH_FAIL": "The sender's identity checks failed, so the message may not come from who it claims.",
    "R05_IP_LINK": "Do not open the links. Type the official website address yourself.",
    "R06_SHORTENER": "Do not open shortened links. They hide where they lead.",
    "R07_SUSPICIOUS_TLD": "Do not open the links. Type the official website address yourself.",
    "R08_LINK_LOOKALIKE": "Do not open the links. Type the official website address yourself.",
    "R09_LINK_TEXT_MISMATCH": "Do not open the links. The visible address is not where the link actually goes.",
    "R11_URGENCY": "Urgent deadlines and threats are a common pressure tactic. Take time to verify before acting.",
    "R12_SENSITIVE_REQUEST": "Never share passwords, OTPs, card, Aadhaar or PAN details by email.",
    "R13_MONEY_WORDING": "Do not pay fees or send gift cards to claim a parcel or a prize.",
    "R14_RISKY_ATTACHMENT": "Do not open the attachment.",
}


def advisories(severity: str, flags: list[dict]) -> list[str]:
    out = [SEVERITY_ADVICE[severity]]
    if severity != "safe":
        for f in flags:
            tip = RULE_ADVICE.get(f["id"])
            if tip and tip not in out:
                out.append(tip)
    return out


def fallback_summary(ev: dict) -> str:
    """Plain-text summary built without any LLM, used when Gemini is off or fails."""
    sev = SEVERITY_LABEL[ev["severity"]].lower()
    pct = round(ev["R"] * 100)
    parts = [f"The combined risk score is {pct} out of 100, which falls in the {sev} range."]
    if ev["rules"]:
        top = sorted(ev["rules"], key=lambda f: -f["weight"])[:2]
        parts.append("The rule checks found: " + "; ".join(f["message"] for f in top) + ".")
    if ev["indicators"]:
        parts.append("The model was influenced most by the words: "
                     + ", ".join(i["term"] for i in ev["indicators"][:4]) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class Engine:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.vec = bundle["vectorizer"]
        self.lr = bundle["model"]
        self.platt_a, self.platt_c = bundle["platt"]
        self.alpha = bundle["alpha"]
        self.t_low = bundle["t_low"]
        self.t_high = bundle["t_high"]
        self.terms = np.asarray(self.vec.get_feature_names_out())
        self.coef = self.lr.coef_[0]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Engine":
        return cls(joblib.load(path))

    def severity(self, R: float, flags: list[dict]) -> str:
        return severity_from(R, flags, self.t_low, self.t_high)

    def _indicators(self, X, top_n: int = 6) -> tuple[list[dict], list[dict]]:
        """Per-email attribution for the linear model: coefficient x TF-IDF value."""
        idx = X.indices
        if len(idx) == 0:
            return [], []
        contrib = X.data * self.coef[idx]
        order = np.argsort(contrib)
        pos = [(self.terms[idx[i]], float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:top_n]
        neg = [(self.terms[idx[i]], float(contrib[i])) for i in order if contrib[i] < 0][:3]
        top = pos[0][1] if pos else 1.0
        return (
            [{"term": t, "weight": round(w / top, 3)} for t, w in pos],
            [{"term": t, "weight": round(abs(w) / top, 3)} for t, w in neg],
        )

    def analyze(self, parsed: dict) -> dict:
        flags = detector.rule_checks(parsed)
        r = detector.rule_score(flags)
        X = self.vec.transform([model_text(parsed)])
        raw = float(self.lr.decision_function(X)[0])
        p = _sigmoid(self.platt_a * raw + self.platt_c)
        R = self.alpha * p + (1.0 - self.alpha) * r
        sev = self.severity(R, flags)
        indicators, safe_terms = self._indicators(X)
        return {
            "p": p, "r": r, "R": R, "alpha": self.alpha,
            "severity": sev,
            "indicators": indicators,
            "safe_terms": safe_terms,
            "rules": flags,
            "advice": advisories(sev, flags),
        }
