"""Web interface:  streamlit run app/streamlit_app.py  (opens http://localhost:8501)."""

import json

import pandas as pd
import streamlit as st

from spamdetector.config import COMPARISON_PATH, FIGURES_DIR, METRICS_PATH, UNSEEN_RESULTS_PATH
from spamdetector.predict import load_all_models

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="wide")

SPAM_EXAMPLES = {
    "🏆 Prize scam": "Congratulations! You won Rs 10000. Claim your prize now!",
    "🏦 Fake bank alert": "URGENT! Your bank account is blocked. Click http://bit.ly/abc to verify now",
    "🎁 Free offer": "FREE entry in a weekly competition! Text WIN to 80086 now",
    "📞 Free ringtone": "Free ringtone! Reply STOP to unsubscribe. Txt to 87077 now",
}
HAM_EXAMPLES = {
    "👋 Friend": "Hey, are you coming to college tomorrow?",
    "📚 College": "Please send me the lab record by evening",
    "🍽️ Family": "Mom said dinner is at 8, don't be late",
    "💼 Work": "Meeting moved to 3 PM in conference room B. Please share the slides before that.",
}

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 3.2rem; max-width: 1250px;}
.hero {padding: 1.4rem 1.8rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #1e3a8a 0%, #4338ca 55%, #7c3aed 100%);}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .35rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.result {padding: 1.2rem 1.4rem; border-radius: 14px; border: 2px solid; margin: .4rem 0 .8rem 0;}
.result.spam {background: rgba(220, 38, 38, .12); border-color: #dc2626;}
.result.ham {background: rgba(22, 163, 74, .12); border-color: #16a34a;}
.result .verdict {font-size: 2rem; font-weight: 800; line-height: 1.1;}
.result.spam .verdict {color: #dc2626;}
.result.ham .verdict {color: #16a34a;}
.result .sub {margin-top: .35rem; font-size: 1rem; opacity: .9;}
.chip {display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .85rem;
       font-weight: 600; background: rgba(120, 120, 120, .18); margin-left: .4rem;}
.small-note {font-size: .88rem; opacity: .75;}
.mini {border-radius: 12px; padding: .7rem .9rem; border-left: 6px solid; background: rgba(120,120,120,.10);}
.mini.spam {border-color: #dc2626;} .mini.ham {border-color: #16a34a;}
.mini .mname {font-size: .85rem; opacity: .75;}
.mini .mverdict {font-size: 1.25rem; font-weight: 700; margin: .1rem 0;}
.mini.spam .mverdict {color: #dc2626;} .mini.ham .mverdict {color: #16a34a;}
.mini .mprob {font-size: .85rem; opacity: .8;}
div[data-testid="stMetric"] {background: rgba(120, 120, 120, .10); border-radius: 12px; padding: .8rem 1rem;}
</style>
<div class="hero">
  <h1>📩 Email &amp; SMS Spam Detector</h1>
  <p>Machine learning with TF-IDF · Naive Bayes · Logistic Regression · SVM</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    return load_all_models()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


try:
    models = get_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

metrics = get_metrics()
best_name = metrics["best_model"] if metrics else next(iter(models))

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    names = list(models)
    model_name = st.radio("Model", names, index=names.index(best_name),
                          help="The model marked 'selected' won the cross-validation.",
                          format_func=lambda n: f"{n} (selected)" if n == best_name else n)
    threshold = st.slider("Spam threshold", 0.10, 0.90, 0.50, 0.05,
                          help="A message is called spam when its spam probability is at least this value. "
                               "Lower = catch more spam but more false alarms. Higher = fewer false alarms "
                               "but more missed spam.")
    if metrics:
        d = metrics["dataset"]
        st.divider()
        st.caption("Training data")
        st.write(f"**{d['messages_after_dedup']:,}** messages  \n"
                 f"{d['ham']:,} ham · {d['spam']:,} spam")

classifier = models[model_name]
classifier.threshold = threshold

st.session_state.setdefault("message", "")
st.session_state.setdefault("analyze", False)


def use_example(text: str) -> None:
    st.session_state["message"] = text
    st.session_state["analyze"] = True


def start_analysis() -> None:
    st.session_state["analyze"] = True


tab_check, tab_perf, tab_unseen, tab_about = st.tabs(
    ["🔍 Check a message", "📊 Model performance", "🧪 Unseen-message test", "ℹ️ About"])

# ---------------- tab 1: check ----------------
with tab_check:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.text_area("Paste an SMS or email text", key="message", height=140,
                     placeholder="Type or paste a message here, or pick an example on the right")
        st.button("Analyze message", type="primary", on_click=start_analysis)

        text = st.session_state["message"].strip()
        if st.session_state["analyze"] and not text:
            st.warning("Please enter a message first.")
        elif st.session_state["analyze"]:
            result = classifier.predict(text)
            css, verdict = ("spam", "🚫 SPAM") if result.is_spam else ("ham", "✅ NOT SPAM")
            st.markdown(
                f'<div class="result {css}"><div class="verdict">{verdict}</div>'
                f'<div class="sub">Spam probability <b>{result.spam_probability:.1%}</b>'
                f'<span class="chip">Confidence: {result.confidence_label}</span></div></div>',
                unsafe_allow_html=True)
            st.progress(float(result.spam_probability), text="Spam probability")

            st.markdown("**What do the other models say?**")
            columns = st.columns(len(models))
            for column, (name, other) in zip(columns, models.items()):
                other.threshold = threshold
                answer = other.predict(text)
                css = "spam" if answer.is_spam else "ham"
                label = "SPAM" if answer.is_spam else "NOT SPAM"
                column.markdown(
                    f'<div class="mini {css}"><div class="mname">{name}</div>'
                    f'<div class="mverdict">{label}</div>'
                    f'<div class="mprob">{answer.spam_probability:.1%} spam</div></div>',
                    unsafe_allow_html=True)

            words = classifier.explain(text, top_n=6)
            if words:
                st.markdown("**Words that influenced this decision**")
                table = pd.DataFrame(
                    {"Towards spam": [max(score, 0) for _, score in words],
                     "Towards not spam": [max(-score, 0) for _, score in words]},
                    index=[word for word, _ in words])
                st.bar_chart(table, horizontal=True, stack=True, color=["#dc2626", "#16a34a"], height=230)
                st.caption("Red words push the message towards spam, green words towards not spam. "
                           "Longer bar = stronger influence.")
            else:
                st.caption("Word-level explanations are not available for the SVM model.")

    with right:
        st.markdown("**🚫 Try a spam example**")
        for label, example in SPAM_EXAMPLES.items():
            st.button(label, key=f"s_{label}", on_click=use_example, args=(example,), use_container_width=True)
        st.markdown("**✅ Try a genuine example**")
        for label, example in HAM_EXAMPLES.items():
            st.button(label, key=f"h_{label}", on_click=use_example, args=(example,), use_container_width=True)

# ---------------- tab 2: performance ----------------
with tab_perf:
    if not metrics:
        st.info("Run `spam-detector train` to generate the performance report.")
    else:
        res = metrics["models"][model_name]["test"]
        st.subheader(f"{model_name}: results on the held-out test set")
        st.caption(f"{metrics['dataset']['test_size']:,} messages that were never used for training or tuning. "
                   "Spam is the positive class.")
        cards = st.columns(5)
        for column, (label, key) in zip(cards, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                                ("Recall", "recall"), ("F1-score", "f1"), ("ROC AUC", "roc_auc")]):
            column.metric(label, f"{res[key]:.3f}")
        st.markdown(
            f"<p class='small-note'>{res['fp']} genuine messages were wrongly flagged as spam and "
            f"{res['fn']} spam messages were missed.</p>", unsafe_allow_html=True)

        st.subheader("All models compared")
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.columns[1:]})
                     .highlight_max(subset=comparison.columns[1:], color="rgba(22,163,74,.35)"),
                     hide_index=True, use_container_width=True)
        st.caption("CV F1 = mean F1 over 5-fold cross-validation (used to select the model). "
                   "The other columns are test-set scores. Green = best in the column.")

        c1, c2 = st.columns(2)
        c1.image(str(FIGURES_DIR / "model_comparison.png"), caption="Model comparison")
        c2.image(str(FIGURES_DIR / "roc_curves.png"), caption="ROC curves")
        st.image(str(FIGURES_DIR / "confusion_matrices.png"), caption="Confusion matrices")
        st.image(str(FIGURES_DIR / "top_spam_words.png"), caption="Words most associated with spam")

# ---------------- tab 3: unseen ----------------
with tab_unseen:
    if not (metrics and "unseen" in metrics and UNSEEN_RESULTS_PATH.exists()):
        st.info("Run `spam-detector train` to generate the unseen-message test.")
    else:
        unseen = metrics["unseen"]
        st.subheader("Testing on messages the models have never seen")
        st.write(
            f"{unseen['messages']} hand-written messages (30 spam, 30 genuine) that are **not** in the training "
            "data and were **never used for tuning**. They include modern Indian-style scams (fake KYC, UPI "
            "refunds, courier fees), Hinglish chat and genuine automatic messages such as OTPs and bank alerts.")
        rows = [{"Model": n, "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
                 "F1-score": r["f1"], "Spam caught": f"{r['tp']} of {r['tp'] + r['fn']}",
                 "Genuine flagged": f"{r['fp']} of {r['fp'] + r['tn']}"} for n, r in unseen["models"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-score"]}),
                     hide_index=True, use_container_width=True)
        st.warning(
            "Scores are clearly lower than on the original test set. The models were trained on older "
            "English SMS, so many modern scams (phishing, job and investment offers) are missed. "
            "With only 60 messages these numbers are a rough guide, not a precise measurement.")

        results = pd.read_csv(UNSEEN_RESULTS_PATH)
        column = f"{model_name} spam probability"
        results["predicted"] = (results[column] >= threshold).map({True: "spam", False: "ham"})
        results["correct"] = results["predicted"] == results["label"]
        only_mistakes = st.checkbox(f"Show only the mistakes made by {model_name}", value=True)
        shown = results[~results["correct"]] if only_mistakes else results
        st.dataframe(shown[["message", "label", "predicted", column, "category"]]
                     .rename(columns={column: "spam probability"}), hide_index=True, use_container_width=True)

# ---------------- tab 4: about ----------------
with tab_about:
    st.markdown(
        """
**How it works**

1. The text is cleaned: lowercase; links, money and numbers become tokens; punctuation is removed.
2. TF-IDF converts the text into numbers (single words and word pairs).
3. The classifier returns the probability that the message is spam.
4. The *spam threshold* in the sidebar decides where that probability becomes SPAM.

**What "confidence" means:** how far the probability is from 50%. Above 90% = High, 75 to 90% = Medium,
below 75% = Low. Treat Low-confidence results with care.

**Limitations**

- Trained on an English SMS dataset from the 2000s, so modern phishing and Hinglish messages are often missed.
- Only the text is analysed (not the sender, link reputation or attachments).
- A learning project, not a replacement for a production spam filter.
""")
