"""Train and evaluate the baseline classifiers described in the paper.

Follows the leakage-resistant workflow of paper section VI:
    1. remove duplicate / near-duplicate messages
    2. split into train / validation / held-out test (the test set is used once)
    3. fit TF-IDF and every model on the training split only
    4. tune alpha and the severity thresholds on the validation split only
    5. report metrics on the test split (paper Table III) with an ablation (section IX-B)

Usage:
    python train.py                   # uses every CSV in data/
    python train.py --max-rows 6000   # quick smoke test on a subset

Data: put CSV files in data/. Supported columns (case-insensitive):
    label  : "Spam/Ham", "label", "class" or "type"   (spam/phishing/1 = unsafe, ham/legit/0 = safe)
    body   : "Message", "body" or "text"
    subject: "Subject" (optional)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import detector
from engine import MAX_TEXT_CHARS, MODEL_PATH, Engine, model_text, severity_from

ROOT = Path(__file__).parent
SEED = 42
POS = {"spam", "phishing", "phish", "unsafe", "malicious", "fraud", "1", "true"}
NEG = {"ham", "legit", "legitimate", "safe", "benign", "0", "false"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _label(v) -> int | None:
    s = str(v).strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None


def load_data(data_dir: Path, only: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for f in sorted(data_dir.glob("*.csv")):
        if only and f.name not in only:
            continue
        df = pd.read_csv(f)
        low = {c.lower().strip(): c for c in df.columns}
        label_col = next((low[k] for k in ("spam/ham", "label", "class", "type") if k in low), None)
        body_col = next((low[k] for k in ("message", "body", "text", "email text", "email_text") if k in low), None)
        if not label_col or not body_col:
            print(f"  skipping {f.name}: needs a label column and a body/message column")
            continue
        subj = df[low["subject"]].fillna("").astype(str) if "subject" in low else ""
        out = pd.DataFrame({
            "subject": subj,
            "body": df[body_col].fillna("").astype(str),
            "y": df[label_col].map(_label),
            "source": f.name,
        })
        bad = out["y"].isna().sum()
        if bad:
            print(f"  {f.name}: dropped {bad} rows with an unknown label")
        frames.append(out.dropna(subset=["y"]))
        print(f"  loaded {f.name}: {len(frames[-1]):,} rows")
    if not frames:
        raise SystemExit("No usable CSV files found in data/. See the docstring in train.py.")
    data = pd.concat(frames, ignore_index=True)
    data["y"] = data["y"].astype(int)
    return data


def dedupe(data: pd.DataFrame) -> pd.DataFrame:
    """Drop exact and near duplicates (same first 1,000 letters/digits); drop label conflicts."""
    norm = (data["subject"] + " " + data["body"]).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True).str[:1000]
    key = norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
    data = data.assign(key=key)
    data = data[norm.str.len() > 0]
    conflict = data.groupby("key")["y"].nunique()
    data = data[~data["key"].isin(conflict[conflict > 1].index)]
    return data.drop_duplicates("key").drop(columns="key").reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def candidate_thresholds(score: np.ndarray) -> np.ndarray:
    u = np.unique(np.round(score, 4))
    if len(u) > 400:
        u = np.unique(np.round(np.quantile(score, np.linspace(0, 1, 401)), 4))
    return u[1:] if len(u) > 1 else u  # never "flag everything"


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best_t, best_f = 0.5, -1.0
    for t in candidate_thresholds(score):
        f = f1_score(y, (score >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def scores_at(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    return {
        "threshold": round(float(thr), 4),
        "accuracy": div(tn + tp, tn + fp + fn + tp),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(score)) > 1 else 0.5,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(raw_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(raw_val.reshape(-1, 1), y_val)
    return float(cal.coef_[0][0]), float(cal.intercept_[0])


def apply_platt(raw: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(ab[0] * raw + ab[1])))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--max-rows", type=int, default=0, help="optional cap for quick tests")
    ap.add_argument("--files", nargs="*", default=None,
                    help="only use these CSV file names from the data folder (default: all)")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading data...")
    raw = load_data(Path(args.data_dir), args.files)
    n_raw = len(raw)
    data = dedupe(raw)
    print(f"{n_raw:,} rows -> {len(data):,} after removing duplicates and label conflicts")
    if args.max_rows and len(data) > args.max_rows:
        data = data.groupby("y", group_keys=False).sample(frac=args.max_rows / len(data), random_state=SEED)
        data = data.reset_index(drop=True)
        print(f"Using a stratified subset of {len(data):,} rows")

    idx = np.arange(len(data))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=data["y"], random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=data["y"].iloc[rest], random_state=SEED)
    y = data["y"].to_numpy()
    print(f"Split: train {len(tr):,} | validation {len(va):,} | test {len(te):,}")

    # Parsed emails, text and rule evidence
    print("Computing rule evidence...")
    parsed = [detector.parsed_from_text(s, b[:MAX_TEXT_CHARS]) for s, b in zip(data["subject"], data["body"])]
    texts = [model_text(p) for p in parsed]
    flag_lists = [detector.rule_checks(p) for p in parsed]
    r_all = np.array([detector.rule_score(f) for f in flag_lists])
    rule_feats = np.array([
        [sum(f["severity"] == s for f in fl) for s in ("high", "medium", "low")] + [r]
        for fl, r in zip(flag_lists, r_all)
    ], dtype=float)

    # TF-IDF fitted on the training split only
    print("Fitting TF-IDF...")
    # Alphabetic tokens only: years, phone prefixes and ids are corpus-specific
    # (the Enron mail is from 2000-2002, the spam from 2004-2005) and would not transfer.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=20000,
                          sublinear_tf=True, strip_accents="unicode",
                          token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    X_tr = vec.fit_transform([texts[i] for i in tr])
    X_va = vec.transform([texts[i] for i in va])
    X_te = vec.transform([texts[i] for i in te])
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    results: list[dict] = []

    def record(name: str, score_va: np.ndarray, score_te: np.ndarray, extra: dict | None = None) -> float:
        thr = best_threshold(y_va, score_va)
        row = {"config": name, **scores_at(y_te, score_te, thr), **(extra or {})}
        results.append(row)
        print(f"  {name:<44} F1 {row['f1']:.3f}  P {row['precision']:.3f}  R {row['recall']:.3f}  FPR {row['fpr']:.3f}")
        return thr

    print("Training and evaluating...")
    # 1. Rules only
    record("Rule-only baseline", r_all[va], r_all[te])

    # 2. TF-IDF + Logistic Regression (the model that gets deployed)
    lr = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear", random_state=SEED).fit(X_tr, y_tr)
    ab = platt(lr.decision_function(X_va), y_va)
    p_va = apply_platt(lr.decision_function(X_va), ab)
    p_te = apply_platt(lr.decision_function(X_te), ab)
    record("NLP only: TF-IDF + Logistic Regression", p_va, p_te)

    # 3. Decision Tree and 4. Random Forest, calibrated the same way
    for name, model in [
        ("NLP only: TF-IDF + Decision Tree",
         DecisionTreeClassifier(min_samples_leaf=2, random_state=SEED)),
        ("NLP only: TF-IDF + Random Forest",
         RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)),
    ]:
        model.fit(X_tr, y_tr)
        ab_m = platt(logit(model.predict_proba(X_va)[:, 1]), y_va)
        record(name,
               apply_platt(logit(model.predict_proba(X_va)[:, 1]), ab_m),
               apply_platt(logit(model.predict_proba(X_te)[:, 1]), ab_m))

    # 5. ML combined: TF-IDF + rule features in one Logistic Regression
    comb = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear", random_state=SEED)
    comb.fit(hstack([X_tr, csr_matrix(rule_feats[tr])]).tocsr(), y_tr)
    record("ML combined: TF-IDF + rule features (LR)",
           comb.predict_proba(hstack([X_va, csr_matrix(rule_feats[va])]).tocsr())[:, 1],
           comb.predict_proba(hstack([X_te, csr_matrix(rule_feats[te])]).tocsr())[:, 1])

    # 6. Hybrid: R = alpha * p + (1 - alpha) * r, alpha chosen on validation only.
    # alpha is limited to 0.5-0.9 so the rule evidence can never be switched off:
    # Enron-Spam has no sender headers, so validation cannot measure the sender rules' value.
    best = (-1.0, 0.8)
    for a in np.round(np.arange(0.5, 0.91, 0.05), 2):
        R_va = a * p_va + (1 - a) * r_all[va]
        f = f1_score(y_va, (R_va >= best_threshold(y_va, R_va)).astype(int), zero_division=0)
        if f >= best[0]:
            best = (f, float(a))
    alpha = best[1]
    R_va = alpha * p_va + (1 - alpha) * r_all[va]
    R_te = alpha * p_te + (1 - alpha) * r_all[te]
    t_flag = record(f"Hybrid (deployed): alpha={alpha:.2f} model + rules", R_va, R_te, {"deployed": True})

    # Severity policy thresholds, chosen on validation only (target 99%):
    #   Phishing   = smallest R where flagged emails are >= 99% truly unsafe
    #   Safe       = largest  R where emails below it are  >= 99% truly safe
    #   Suspicious = everything in between (the evidence is incomplete or conflicting)
    TARGET, MIN_SUPPORT = 0.99, 30
    cands = candidate_thresholds(R_va)
    t_high = next((float(t) for t in cands
                   if (R_va >= t).sum() >= MIN_SUPPORT and y_va[R_va >= t].mean() >= TARGET), 0.9)
    t_low = next((float(t) for t in cands[::-1]
                  if (R_va < t).sum() >= MIN_SUPPORT and 1 - y_va[R_va < t].mean() >= TARGET), float(cands[0]))
    t_low = min(t_low, t_high)
    print(f"alpha={alpha:.2f}  Safe < {t_low:.3f}  Suspicious < {t_high:.3f}  Phishing >= {t_high:.3f}")

    sev_te = [severity_from(R_te[k], flag_lists[i], t_low, t_high) for k, i in enumerate(te)]
    sev_te = np.array(sev_te)
    sev_counts = {}
    for name in ("safe", "suspicious", "phishing"):
        m = sev_te == name
        sev_counts[name] = {"n": int(m.sum()), "share_unsafe": float(y_te[m].mean()) if m.any() else None}
    print("Test severity bands (n, share truly unsafe):",
          {k: (v["n"], None if v["share_unsafe"] is None else round(v["share_unsafe"], 3)) for k, v in sev_counts.items()})

    # Save the deployed bundle
    top = np.argsort(lr.coef_[0])
    terms = vec.get_feature_names_out()
    bundle = {
        "vectorizer": vec, "model": lr, "platt": ab,
        "alpha": alpha, "t_low": t_low, "t_high": t_high,
        "sklearn_version": sklearn.__version__,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # Latency of the full online path (rules + TF-IDF + model + advisory)
    eng = Engine.load()
    sample = [parsed[i] for i in np.random.default_rng(SEED).choice(te, size=min(300, len(te)), replace=False)]
    eng.analyze(sample[0])  # warm up
    times = []
    for p in sample:
        s = time.perf_counter()
        eng.analyze(p)
        times.append((time.perf_counter() - s) * 1000)

    # Modern hand-written demo emails (never used for training)
    demo = []
    samples_dir = ROOT / "samples"
    if samples_dir.exists():
        for item in detector.load_samples(samples_dir):
            ev = eng.analyze(item)
            demo.append({"file": item["filename"], "subject": item["subject"], "expected": item["label"],
                         "severity": ev["severity"], "risk": round(ev["R"] * 100),
                         "p": round(ev["p"], 3), "r": round(ev["r"], 3)})

    metrics = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "files": sorted(raw["source"].unique().tolist()),
            "rows_raw": int(n_raw), "rows_after_dedupe": int(len(data)),
            "unsafe_share": round(float(data["y"].mean()), 3),
            "train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
        },
        "alpha": alpha, "t_low": t_low, "t_high": t_high, "policy_target": TARGET,
        "severity_on_test": sev_counts,
        "results": results,
        "latency_ms": {"median": round(float(np.median(times)), 2), "p95": round(float(np.percentile(times, 95)), 2)},
        "top_unsafe_terms": [terms[i] for i in top[::-1][:15]],
        "top_safe_terms": [terms[i] for i in top[:15]],
        "demo_samples": demo,
    }
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Latency per email: median {metrics['latency_ms']['median']} ms, p95 {metrics['latency_ms']['p95']} ms")
    print(f"Done in {time.time() - t0:.0f} s. Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()
