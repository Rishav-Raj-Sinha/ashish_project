"""Streamlit front end for the explainable phishing detection and advisory system.

Pipeline per email: trained TF-IDF classifier + deterministic rules -> risk score ->
severity (Safe / Suspicious / Phishing) -> advisory actions. Everything runs locally.
"""

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

import detector
from engine import MODEL_PATH, SEVERITY_LABEL, Engine, fallback_summary

ROOT = Path(__file__).parent
SAMPLES_DIR = ROOT / "samples"
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


def analyze_item(item: dict, engine: Engine) -> None:
    """Run the local pipeline (trained model + rules + advisory) and store the result."""
    st.session_state.results[item["key"]] = {"ev": engine.analyze(item)}


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

emails = get_samples(samples_signature())

st.title("Explainable phishing detector")
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
                    analyze_item(selected, engine)

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
                st.write(fallback_summary(ev))

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