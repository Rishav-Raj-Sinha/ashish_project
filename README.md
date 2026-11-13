# Explainable phishing detector

A Streamlit app that implements the design in *Explainable Phishing Detection and Advisory
System Using AI Rules and Natural Language Processing*: a trained classifier plus deterministic
rules, a combined risk score, a Safe / Suspicious / Phishing severity policy, and advice on what
to do next. Gemini is optional and only rewrites the verified evidence in plain words.

## How it follows the paper

| Paper | In this project |
|---|---|
| Section V-B/C, NLP and rule features | `train.py` fits TF-IDF word and phrase features. `detector.py` holds the rules (look-alike domains, link mismatch, urgency, credential requests...) with ids, severities and explanations |
| Section IV-A, interpretable baselines | Logistic Regression (deployed), Decision Tree and Random Forest are trained and compared |
| Section IV-B, risk aggregation | `R = alpha * p + (1 - alpha) * r`, with alpha chosen on validation data |
| Section VII, severity and advisory | `engine.py`: thresholds on R, strong rule hits can raise a level but never lower it, advice is chosen from rule ids, not free text |
| Section VII, generative model constraint | Gemini gets only the evidence object (never the email body) and cannot change the result |
| Section VI, leakage-resistant workflow | Duplicates removed, then train / validation / test split, test set used once |
| Section IX, evaluation | Precision, recall, F1, false-positive and false-negative rate, AUC, latency, and an ablation (rules only, NLP only, ML combined, hybrid). Shown in the app's "Model evaluation" tab |

## Folder structure

```
email-fraud-detector/
  app.py              Streamlit app (Analyze tab and Model evaluation tab)
  engine.py           loads the model, risk score, severity policy, advisory engine
  detector.py         .eml parsing, rule evidence, Gemini explainer
  train.py            training, calibration, ablation and metrics
  download_data.py    downloads the training CSV into data/
  requirements.txt    pinned versions
  models/
    model.joblib      trained model (committed so the app runs without retraining)
    metrics.json      results shown in the Evaluation tab
  samples/            demo emails shown in the inbox list
    legit/  spam/  phishing/
  data/               training CSV (not committed)
```

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

The trained model is included, so this works immediately with no API key. Click a row, click
Analyze. Paste a Gemini key in the sidebar if you also want a plain-language explanation.

## Retrain it

```
python download_data.py      # fetches the Enron-Spam CSV into data/
python train.py              # about a minute; rewrites models/
```

To add your own data, put more CSV files in `data/`. Columns (any case): a label column named
`Spam/Ham`, `label`, `class` or `type` (values spam, phishing or 1 mean unsafe; ham, legit or 0
mean safe), a body column named `Message`, `body` or `text`, and optionally `Subject`. A
phishing-specific corpus such as Nazario or CEAS-08 would improve it a lot. If you retrain with
a different scikit-learn version, update the pin in `requirements.txt` to match.

## Samples

Drop `.eml` files into `samples/legit`, `samples/spam` or `samples/phishing` and refresh. The
folder name is only a hidden label. Set `SHOW_EXPECTED = True` near the top of `app.py` to show
it next to the result.

## Deploy on Streamlit Community Cloud

1. Put the project on GitHub. Keep `models/model.joblib` in the repo. `data/` is ignored on purpose.
2. At share.streamlit.io choose the repo, branch `main`, and `app.py`. In Advanced settings pick Python 3.12 or 3.13.
3. Optional: add `GEMINI_API_KEY = "your-key"` under secrets for Gemini explanations. Without it the app still works fully.

## Limitations to state when presenting

- The training data is Enron-Spam: English email from the early 2000s, labelled spam or normal. It is
  not a phishing corpus, and modern phishing (UPI and KYC scams, polished fraud) is under-represented.
- The model partly learned the corpus itself (words like "enron" and "houston" count as safe).
- The dataset has no sender headers, so the sender and authentication rules could not be validated on it.
- The 10 sample emails were written for this project. They are a smoke test, not a benchmark.
- Explanation fidelity, stability and a user study (paper section IX-D) are not done.
