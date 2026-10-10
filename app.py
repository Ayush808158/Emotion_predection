"""Step 3 - Web application (Streamlit).  Run:  streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from data_cleaning import clean_text

MODEL_PATH = Path(__file__).parent / "models" / "emotion_model.joblib"
EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲"}
EXAMPLES = [
    "i feel so happy and grateful for everything today",
    "i am terrified of what might happen tomorrow",
    "i cannot believe how rude he was to me i am furious",
    "i feel lonely and i miss her so much",
    "i adore spending time with my family",
    "i was shocked when i opened the door",
]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def predict(text: str, model: dict):
    """Return (emotion, confidence, {emotion: probability}) or None if text has no letters."""
    cleaned = clean_text(text)
    if not cleaned:
        return None
    pipe = model["pipeline"]
    probs = pipe.predict_proba([cleaned])[0]
    ranked = sorted(zip(pipe.classes_, probs), key=lambda p: p[1], reverse=True)
    return ranked[0][0], float(ranked[0][1]), {k: float(v) for k, v in ranked}


def set_example(text: str):
    st.session_state["user_text"] = text


st.set_page_config(page_title="Emotion Detector", page_icon="🎭")
st.title("🎭 Emotion Detector")
st.caption("NLP project: classifies English text into joy, sadness, anger, fear, love or surprise.")

if not MODEL_PATH.exists():
    st.error("Model file missing. Run `python train_model.py` first (it creates models/emotion_model.joblib).")
    st.stop()
try:
    model = load_model()
except Exception as exc:  # usually a scikit-learn version mismatch
    st.error(f"Could not load the model. Re-run `python train_model.py` in this environment.\n\n{exc}")
    st.stop()

meta = model.get("meta", {})
if meta.get("sklearn_version") not in (None, sklearn.__version__):
    st.warning(f"Model trained with scikit-learn {meta['sklearn_version']}, but {sklearn.__version__} "
               "is installed. If anything looks wrong, re-run `python train_model.py`.")

st.text_area("Enter a sentence (English):", key="user_text", max_chars=500, height=110,
             placeholder="e.g. i feel so happy today")
with st.expander("Try an example"):
    for ex in EXAMPLES:
        st.button(ex, on_click=set_example, args=(ex,), key=f"ex_{ex}")

if st.button("Detect emotion", type="primary"):
    result = predict(st.session_state.get("user_text", ""), model)
    if result is None:
        st.info("Please type a sentence that contains some letters.")
    else:
        emotion, confidence, probs = result
        st.subheader(f"{EMOJI.get(emotion, '')} {emotion.capitalize()}")
        st.write(f"Confidence: **{confidence:.1%}**")
        if confidence < 0.5:
            st.caption("The model is not very sure - the text may express mixed emotions.")
        st.bar_chart(pd.Series(probs, name="probability").sort_values(), horizontal=True)

with st.sidebar:
    st.header("About the model")
    st.write(meta.get("best_model", "TF-IDF + linear classifier"))
    if "test_accuracy" in meta:
        st.metric("Test accuracy", f"{meta['test_accuracy']:.1%}")
        st.metric("Macro F1", f"{meta['test_macro_f1']:.1%}")
    st.caption("Trained on the dair-ai *emotion* dataset (English tweets, 6 emotions). Works best on short, "
               "first-person sentences like \"i feel ...\"; may struggle with sarcasm, slang or other languages.")
