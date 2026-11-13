"""Streamlit front end for the explainable phishing detection and advisory system.

Pipeline per email: trained TF-IDF classifier + deterministic rules -> risk score ->
severity (Safe / Suspicious / Phishing) -> advisory actions. Gemini is optional and only
rewrites the verified evidence in plain words; it never decides the result.
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import detector
from engine import MODEL_PATH, SEVERITY_LABEL, Engine, fallback_summary

ROOT = Path(__file__).parent
SAMPLES_DIR = ROOT / "samples"
METRICS_PATH = ROOT / "models" / "metrics.json"
SHOW_EXPECTED = False  # True shows each sample's true folder label and a running accuracy line
SEV_COLOR = {"safe": "green", "suspicious": "orange", "phishing": "red"}

st.set_page_config(page_title="Explainable phishing detector", page_icon=":shield:", layout="wide")
st.session_state.setdefault("results", {})


# ---------- helpers ----------

@st.cache_resource(show_spinner="Loading model...")
def get_engine() -> Engine:
    return Engine.load()


@st.cache_data(show_spinner=False)
def get_samples(signature: tuple) -> list[dict]:
    # `signature` changes when files are added, removed or edited, so the cache refreshes.
    return detector.load_samples(SAMPLES_DIR)


def samples_signature() -> tuple:
    return tuple((str(p), p.stat().st_mtime) for p in sorted(SAMPLES_DIR.rglob("*.eml")))


def get_api_key() -> str | None:
    key = st.session_state.get("api_key_input") or os.environ.get("GEMINI_API_KEY") \
        or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return key


def analyze_item(item: dict, engine: Engine, api_key: str | None = None) -> None:
    """Run the local pipeline; add a Gemini explanation when a key is available."""
    ev = engine.analyze(item)
    res = {"ev": ev, "gemini": None, "gemini_error": None}
    if api_key:
        try:
            res["gemini"] = detector.explain_with_gemini(ev, api_key) or None
        except Exception as exc:
            res["gemini_error"] = str(exc)[:200]
    st.session_state.results[item["key"]] = res


def severity_text(item: dict) -> str:
    res = st.session_state.results.get(item["key"])
    return SEVERITY_LABEL[res["ev"]["severity"]] if res else "-"


def risk_text(item: dict):
    res = st.session_state.results.get(item["key"])
    return round(res["ev"]["R"] * 100) if res else None


# ---------- load model ----------

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first (see the README).")
    st.stop()
engine = get_engine()

with st.sidebar:
    st.header("Settings")
    st.text_input("Gemini API key (optional)", type="password", key="api_key_input",
                  help="Adds a plain-language explanation. The detection itself works without it.")
    st.caption("Without a key, the app still classifies every email and shows the evidence and advice.")

api_key = get_api_key()
emails = get_samples(samples_signature())

st.title("Explainable phishing detector")
tab_analyze, tab_eval = st.tabs(["Analyze", "Model evaluation"])


# ---------- analyze tab ----------

with tab_analyze:
    top_left, top_right = st.columns([5, 1])
    mode = top_left.radio("Source", ["Folder samples", "Paste an email"], horizontal=True,
                          label_visibility="collapsed")
    if mode == "Folder samples" and top_right.button("Run all", help="Classify every email in the folder"):
        for it in emails:
            analyze_item(it, engine)  # local only: fast and free

    left, right = st.columns([1, 1.25], gap="medium")
    selected = None

    with left:
        if mode == "Folder samples":
            st.subheader(f"Inbox ({len(emails)} emails)")
            if not emails:
                st.info("Put .eml files in samples/legit, samples/spam or samples/phishing.")
            else:
                rows = []
                for it in emails:
                    row = {"From": it["from_addr"] or it["from_name"], "Subject": it["subject"],
                           "Date": it["date_short"], "Result": severity_text(it), "Risk": risk_text(it)}
                    if SHOW_EXPECTED:
                        row["Expected"] = it["label"]
                    rows.append(row)
                event = st.dataframe(
                    pd.DataFrame(rows), hide_index=True, on_select="rerun",
                    selection_mode="single-row", key="inbox",
                    column_config={"Risk": st.column_config.NumberColumn("Risk /100")},
                )
                if event.selection.rows:
                    selected = emails[event.selection.rows[0]]
                done = [it for it in emails if it["key"] in st.session_state.results]
                if SHOW_EXPECTED and done:
                    ok = sum((it["label"] == "legit")
                             == (st.session_state.results[it["key"]]["ev"]["severity"] == "safe")
                             for it in done)
                    st.caption(f"Safe vs flagged correct: {ok}/{len(done)} analyzed")
        else:
            st.subheader("Paste an email")
            pasted = st.text_area(
                "Raw email or just the body", height=320,
                placeholder="Paste the full email (with headers) or just the message text. "
                            "Press Ctrl+Enter when done.")
            if pasted.strip():
                selected = detector.parse_pasted(pasted)
                selected["key"] = "paste:" + hashlib.md5(pasted.encode()).hexdigest()[:12]
                selected["label"] = None

    with right:
        if selected is None:
            st.info("Select an email from the list to see it here." if mode == "Folder samples"
                    else "Paste an email on the left.")
        else:
            with st.container(border=True):
                sender = f"`{selected['from_addr'] or 'unknown'}`"
                if selected["from_name"]:
                    sender += f" ({selected['from_name']})"
                st.markdown(f"**From:** {sender}")
                if selected["reply_to"]:
                    st.markdown(f"**Reply-to:** `{selected['reply_to']}`")
                st.markdown(f"**Subject:** `{selected['subject']}`")
                st.markdown(f"**Links:** {len(selected['links'])} found &nbsp;|&nbsp; "
                            f"**Attachments:** {', '.join(selected['attachments']) or 'none'}")
                with st.container(height=170):
                    st.text(selected["body"][:2500] or "(empty body)")
                if selected["links"]:
                    with st.expander("Links in this email"):
                        for l in selected["links"]:
                            shown = f"{l['text']}  ->  " if l["text"] else ""
                            st.code(f"{shown}{l['url']}", language=None)
                if st.button("Analyze", key=f"analyze_{selected['key']}", type="primary"):
                    with st.spinner("Analyzing..."):
                        analyze_item(selected, engine, api_key)

            res = st.session_state.results.get(selected["key"])
            if res:
                ev = res["ev"]
                sev = ev["severity"]
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"### :{SEV_COLOR[sev]}[{SEVERITY_LABEL[sev]}]")
                    c2.markdown(f"### {round(ev['R'] * 100)} / 100")
                    c2.caption("Combined risk score")
                    st.progress(min(1.0, ev["R"]))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Model probability", f"{ev['p']:.2f}", help="Calibrated probability from the trained TF-IDF classifier.")
                    m2.metric("Rule score", f"{ev['r']:.2f}", help="Combined strength of the deterministic rule hits.")
                    m3.metric("Weight on model", f"{ev['alpha']:.2f}", help="R = alpha x model + (1 - alpha) x rules, alpha chosen on validation data.")

                    st.markdown("**What to do**")
                    st.info(ev["advice"][0])
                    for tip in ev["advice"][1:]:
                        st.markdown(f"- {tip}")

                    st.markdown("**Explanation**")
                    if res["gemini"]:
                        st.write(res["gemini"])
                        st.caption("Written by Gemini from the evidence below. It does not change the result.")
                    else:
                        st.write(fallback_summary(ev))
                        if res["gemini_error"]:
                            st.caption(f"Gemini explanation unavailable ({res['gemini_error']}). Showing the built-in summary.")
                        elif not api_key:
                            st.caption("Built-in summary. Add a Gemini API key in the sidebar for a plain-language explanation.")

                    with st.expander("Evidence behind this result", expanded=True):
                        st.markdown("**Rule hits**")
                        if ev["rules"]:
                            st.dataframe(
                                pd.DataFrame([{"Rule": f["id"], "Severity": f["severity"], "Detail": f["message"]}
                                              for f in ev["rules"]]),
                                hide_index=True, width="stretch")
                        else:
                            st.write("No rule hits.")
                        st.markdown("**Words that pushed the model toward unsafe**")
                        if ev["indicators"]:
                            st.dataframe(
                                pd.DataFrame(ev["indicators"]).rename(columns={"term": "Word or phrase", "weight": "Relative influence"}),
                                hide_index=True, width="stretch",
                                column_config={"Relative influence": st.column_config.ProgressColumn(
                                    "Relative influence", min_value=0, max_value=1, format="%.2f")})
                        else:
                            st.write("No words pushed the model toward unsafe.")
                        if ev["safe_terms"]:
                            st.caption("Words that pushed toward safe: " + ", ".join(t["term"] for t in ev["safe_terms"]))

                    if SHOW_EXPECTED and selected.get("label"):
                        ok = (selected["label"] == "legit") == (sev == "safe")
                        st.caption(f"Expected: {selected['label']} | Detected: {sev} ({'match' if ok else 'mismatch'})")


# ---------- evaluation tab ----------

with tab_eval:
    if not METRICS_PATH.exists():
        st.info("No metrics yet. Run `python train.py` to train the model and write models/metrics.json.")
    else:
        m = json.loads(METRICS_PATH.read_text())
        ds = m["dataset"]
        st.subheader("How the model was trained and tested")
        a, b, c, d = st.columns(4)
        a.metric("Emails after de-duplication", f"{ds['rows_after_dedupe']:,}")
        b.metric("Training", f"{ds['train']:,}")
        c.metric("Validation", f"{ds['validation']:,}")
        d.metric("Held-out test", f"{ds['test']:,}")
        st.caption(
            f"Data: {', '.join(ds['files'])} ({ds['rows_raw']:,} rows before removing duplicates; "
            f"{ds['unsafe_share'] * 100:.0f}% labelled spam). TF-IDF, every model and the calibration were fitted on "
            f"the training split. The weight alpha and the severity thresholds were chosen on validation data. "
            f"The test split was used once. Trained {m['trained_at']} with scikit-learn {m['sklearn_version']}.")

        st.subheader("Results on the held-out test set")
        res = pd.DataFrame(m["results"])
        table = res[["config", "precision", "recall", "f1", "fpr", "fnr", "auc"]].rename(columns={
            "config": "Configuration", "precision": "Precision", "recall": "Recall", "f1": "F1",
            "fpr": "False-positive rate", "fnr": "False-negative rate", "auc": "AUC"})
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns[1:]})
        st.caption("Positive class = spam / unsafe. Each row uses the threshold that maximised F1 on the validation split.")

        dep = next(r for r in m["results"] if r.get("deployed"))
        left_e, right_e = st.columns(2)
        with left_e:
            st.markdown("**Confusion matrix, deployed hybrid**")
            st.dataframe(pd.DataFrame(
                [[dep["tn"], dep["fp"]], [dep["fn"], dep["tp"]]],
                index=["Actually safe", "Actually unsafe"], columns=["Predicted safe", "Predicted unsafe"]),
                width="stretch")
            st.markdown("**Speed (full pipeline, one email)**")
            st.write(f"Median {m['latency_ms']['median']} ms, 95th percentile {m['latency_ms']['p95']} ms")
        with right_e:
            st.markdown("**Severity bands on the test set**")
            st.dataframe(pd.DataFrame([
                {"Band": SEVERITY_LABEL[k], "Emails": v["n"],
                 "Truly unsafe": None if v["share_unsafe"] is None else f"{v['share_unsafe'] * 100:.1f}%"}
                for k, v in m["severity_on_test"].items()]), hide_index=True, width="stretch")
            st.caption(f"Safe: below {m['t_low']:.2f}. Phishing: {m['t_high']:.2f} or above, "
                       f"chosen so that each side is at least {m['policy_target'] * 100:.0f}% pure on validation data. "
                       "Strong rule hits can raise a band, never lower it.")

        if m["demo_samples"]:
            st.subheader("Hand-written modern examples (never used in training)")
            demo = pd.DataFrame(m["demo_samples"])[["subject", "expected", "severity", "risk"]]
            demo["severity"] = demo["severity"].map(SEVERITY_LABEL)
            st.dataframe(demo.rename(columns={"subject": "Subject", "expected": "Folder label",
                                              "severity": "Result", "risk": "Risk /100"}),
                         hide_index=True, width="stretch")
            st.caption("These 10 emails were written for this project. They are a smoke test, not a benchmark.")

        with st.expander("Words the model weights most"):
            t1, t2 = st.columns(2)
            t1.markdown("**Toward unsafe**")
            t1.write(", ".join(m["top_unsafe_terms"]))
            t2.markdown("**Toward safe**")
            t2.write(", ".join(m["top_safe_terms"]))
            st.caption("Words such as 'enron' and 'houston' show the model has partly learned the training corpus "
                       "itself, which is one reason scores on other mail should be read with care.")

        with st.expander("Limitations", expanded=True):
            st.markdown(
                "- **Training data is spam versus normal mail, not phishing.** Enron-Spam is English email from the 2000s. "
                "Modern phishing, UPI or KYC scams and professionally written fraud are under-represented.\n"
                "- **The model has learned corpus quirks**, such as Enron names counting as safe.\n"
                "- **The dataset has no sender headers**, so the sender and authentication rules cannot be validated on it. "
                "They are included by design and the weight on rules is kept between 0.1 and 0.5.\n"
                "- **No URL-only baseline**: few messages in this dataset contain links.\n"
                "- **Explanation quality is not measured**: fidelity, stability and a user study are future work.\n"
                "- **Gemini text is not evaluated.** It only rewrites the evidence and does not affect the result.")
