"""Step 3 - Web application (Streamlit).  Run:  streamlit run app.py

Tab 1: Emotion detector (your trained TF-IDF + calibrated Linear SVM model).
Tab 2: Reply assistant (LLM + tool calling + zero/one/few-shot prompting, see llm_assistant.py).

The API key is read from .streamlit/secrets.toml (GROQ_API_KEY), so you add it only once.
"""
import html
import os
import re
from pathlib import Path

import joblib
import sklearn
import streamlit as st

import llm_assistant as llm
from data_cleaning import clean_text

MODEL_PATH = Path(__file__).parent / "models" / "emotion_model.joblib"

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲"}
COLOR = {"joy": "#F2B705", "sadness": "#4C8DF6", "anger": "#EF5350",
         "fear": "#9575E8", "love": "#EC5C93", "surprise": "#26B79F"}

# (chip label, sentence)
DETECT_EXAMPLES = [
    ("😄 Joy", "i feel so happy and grateful for everything today"),
    ("😨 Fear", "i am terrified of what might happen tomorrow"),
    ("😠 Anger", "i cannot believe how rude he was to me i am furious"),
    ("😢 Sadness", "i feel lonely and i miss her so much"),
    ("❤️ Love", "i adore spending time with my family"),
    ("😲 Surprise", "i was shocked when i opened the door"),
]
REPLY_EXAMPLES = [
    ("😢 Bad news", "i failed my exam and i do not know how to tell my parents"),
    ("😠 Frustrated", "you never listen to me and i am so tired of it"),
    ("😄 Good news", "i got selected for the national team i am over the moon"),
]
MODE_HELP = {
    "zero-shot": "Instructions only, no examples.",
    "one-shot": "Instructions plus 1 worked example.",
    "few-shot": "Instructions plus 3 worked examples.",
}


# ---------- model helpers ----------
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


def set_text(key: str, text: str):
    st.session_state[key] = text


@st.cache_resource
def get_client(api_key: str):
    return llm.make_client(api_key)


ROOT = Path(__file__).parent
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"
SECRETS_FILES = [SECRETS_FILE, ROOT / "secrets.toml"]   # the proper place first, then next to app.py


def _key_from_file():
    """Read GROQ_API_KEY from secrets.toml (quotes optional, works from any folder)."""
    for path in SECRETS_FILES:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        match = re.search(r"^\s*GROQ_API_KEY\s*=\s*[\"']?([^\"'\s#]+)", text, re.M)
        if match:
            return match.group(1).strip()
    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key: str):
    """(models this key can use, error message or None)."""
    try:
        return llm.list_chat_models(get_client(api_key)), None
    except Exception as exc:
        return [], llm.friendly_error(exc)


def get_api_key():
    """secrets.toml -> environment variable -> sidebar box."""
    candidates = []
    try:
        candidates.append(st.secrets.get("GROQ_API_KEY"))
    except Exception:  # no secrets file found by Streamlit
        pass
    candidates += [_key_from_file(), os.environ.get("GROQ_API_KEY")]
    for key in candidates:
        if key and "paste-your" not in str(key):  # skip the untouched template
            return str(key).strip()
    return st.session_state.get("api_key_input") or None


# ---------- UI building blocks ----------
def esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def emotion_card(emotion: str, confidence: float | None, note: str = ""):
    color = COLOR.get(emotion, "#888888")
    conf = f"Confidence {confidence:.1%}" if confidence is not None else ""
    note_html = f'<div class="emo-note">{esc(note)}</div>' if note else ""
    st.markdown(
        f'<div class="emo-card" style="--c:{color}">'
        f'<div class="emo-emoji">{EMOJI.get(emotion, "")}</div>'
        f'<div><div class="emo-name">{esc(emotion.capitalize())}</div>'
        f'<div class="emo-conf">{conf}</div>{note_html}</div></div>',
        unsafe_allow_html=True,
    )


def prob_bars(probs: dict, top: str):
    rows = []
    for emo, p in probs.items():
        c = COLOR.get(emo, "#888888")
        weight = "1" if emo == top else ".5"
        rows.append(
            f'<div class="pb-row"><span class="pb-label">{EMOJI.get(emo, "")} {esc(emo)}</span>'
            f'<div class="pb-track"><div class="pb-fill" style="width:{p * 100:.1f}%;background:{c};opacity:{weight}"></div></div>'
            f'<span class="pb-val">{p:.1%}</span></div>')
    st.markdown(f'<div class="pb">{"".join(rows)}</div>', unsafe_allow_html=True)


def bubble(text: str, emotion: str | None, label: str = ""):
    color = COLOR.get(emotion or "", "#8B7CF6")
    label_html = f'<div class="bubble-label">{esc(label)}</div>' if label else ""
    st.markdown(f'<div class="bubble" style="--c:{color}">{label_html}{esc(text)}</div>',
                unsafe_allow_html=True)


def example_chips(examples, key_prefix: str, target_key: str):
    cols = st.columns(len(examples) if len(examples) <= 3 else 3)
    for i, (label, sentence) in enumerate(examples):
        cols[i % len(cols)].button(label, key=f"{key_prefix}_{i}", on_click=set_text,
                                   args=(target_key, sentence), use_container_width=True)


# ---------- page ----------
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=DM+Sans:wght@400;500;700&display=swap');

.stApp, .stMarkdown, .stTextArea textarea, .stTextInput input, .stButton button,
.stRadio label, .stTabs button, [data-testid="stSidebar"] {font-family: 'DM Sans', sans-serif;}
.block-container {max-width: 800px; padding-top: 2rem; padding-bottom: 3rem;}

/* hero */
.hero {padding: .4rem 0 1.4rem 0;}
.hero-emojis {font-size: 1.5rem; letter-spacing: .5rem; margin-bottom: .6rem;}
.hero-title {font-family: 'Outfit', sans-serif; font-size: 2.7rem; font-weight: 700; line-height: 1.1;
             margin: 0 0 .6rem 0; letter-spacing: -.02em;}
.hero-sub {font-size: 1.05rem; opacity: .72; max-width: 34rem; line-height: 1.55; margin: 0;}

/* tabs and buttons */
.stTabs [data-baseweb="tab-list"] {gap: 1.4rem;}
.stTabs button p {font-size: 1.02rem; font-weight: 600;}
.stButton button {border-radius: 10px; font-weight: 500;}
.stTextArea textarea {border-radius: 12px; font-size: 1.02rem; line-height: 1.5;}

/* result card */
.emo-card {display: flex; align-items: center; gap: 1.1rem; padding: 1.1rem 1.4rem; margin: 1rem 0 .9rem 0;
           border-radius: 16px; border: 1px solid var(--c);
           background: rgba(128,128,128,.10);
           background: color-mix(in srgb, var(--c) 14%, transparent);
           border-color: color-mix(in srgb, var(--c) 55%, transparent);}
.emo-emoji {font-size: 3.2rem; line-height: 1;}
.emo-name {font-family: 'Outfit', sans-serif; font-size: 2.1rem; font-weight: 700; color: var(--c); line-height: 1.1;}
.emo-conf {opacity: .8; margin-top: .2rem;}
.emo-note {opacity: .65; font-size: .9rem; margin-top: .35rem;}

/* probability bars */
.pb {margin: .2rem 0 1rem 0;}
.pb-row {display: grid; grid-template-columns: 7.2rem 1fr 3.6rem; align-items: center; gap: .8rem; margin: .42rem 0;}
.pb-label {font-size: .95rem; text-transform: capitalize;}
.pb-track {height: 10px; border-radius: 99px; background: rgba(128,128,128,.18); overflow: hidden;}
.pb-fill {height: 100%; border-radius: 99px;}
.pb-val {font-size: .9rem; text-align: right; opacity: .8; font-variant-numeric: tabular-nums;}

/* pipeline strip */
.flow {display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .6rem; margin: .4rem 0 1.2rem 0;}
.flow-step {padding: .7rem .9rem; border-radius: 12px; background: rgba(128,128,128,.10); font-size: .88rem; line-height: 1.4;}
.flow-step b {display: block; font-size: .95rem; margin-bottom: .1rem;}
.flow-n {display: inline-block; width: 1.35rem; height: 1.35rem; line-height: 1.35rem; text-align: center;
         border-radius: 50%; background: #8B7CF6; color: white; font-size: .78rem; font-weight: 700; margin-right: .4rem;}

/* reply bubble */
.bubble {border-left: 5px solid var(--c); background: rgba(128,128,128,.10); padding: 1rem 1.2rem;
         border-radius: 4px 14px 14px 4px; line-height: 1.6; margin: .5rem 0 .8rem 0; font-size: 1.02rem;}
.bubble-label {font-weight: 700; font-size: .85rem; margin-bottom: .35rem; color: var(--c);}

/* tool call log */
.toolcall {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .85rem; padding: .7rem .9rem;
           border-radius: 10px; background: rgba(128,128,128,.12); overflow-x: auto; white-space: pre;}

@media (max-width: 640px) {
  .hero-title {font-size: 2rem;}
  .flow {grid-template-columns: 1fr;}
  .pb-row {grid-template-columns: 5.6rem 1fr 3.4rem;}
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="hero-emojis">😄 😢 😠 😨 ❤️ 😲</div>'
    '<div class="hero-title">What is this sentence feeling?</div>'
    '<p class="hero-sub">Type a sentence and a trained model reads the emotion behind it. '
    'Then let an AI assistant help you write the right reply.</p>'
    '</div>', unsafe_allow_html=True)

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


def detect_for_tool(text: str) -> dict:
    """The function the LLM calls as its `detect_emotion` tool."""
    result = predict(text, model)
    if result is None:
        return {"error": "The text has no letters."}
    emotion, confidence, _ = result
    return {"emotion": emotion, "confidence": round(confidence, 3)}


# ---------- sidebar ----------
with st.sidebar:
    st.header("The model")
    st.write(meta.get("best_model", "TF-IDF + linear classifier"))
    if "test_accuracy" in meta:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{meta['test_accuracy']:.1%}")
        c2.metric("Macro F1", f"{meta['test_macro_f1']:.1%}")
    st.caption("Trained on the dair-ai *emotion* dataset (English tweets, 6 emotions). Works best on short, "
               "first-person sentences like \"i feel ...\"; may struggle with sarcasm, slang or other languages.")

    st.divider()
    st.header("Reply assistant")
    if get_api_key() and not st.session_state.get("api_key_input"):
        st.success("API key connected.")
    else:
        st.warning("No API key found.")
        st.caption("Looked for the key in these files:")
        st.code("\n".join(f"{'FOUND ' if f.exists() else 'missing'}  {f}" for f in SECRETS_FILES),
                language=None, wrap_lines=True)
        st.text_input("Groq API key", type="password", key="api_key_input",
                      help="Better: put it once in .streamlit/secrets.toml so you never type it again.")
    _key = get_api_key()
    available, model_error = fetch_models(_key) if _key else ([], None)
    if model_error:
        st.error(model_error)
    elif available:
        st.caption(f"{len(available)} models available on your account.")
    with st.expander("Advanced", expanded=bool(model_error)):
        if available:
            llm_model = st.selectbox("LLM model", available, index=0)
        else:
            llm_model = st.text_input("LLM model name", value=llm.DEFAULT_MODEL)

if "llm_model" not in globals():
    llm_model = llm.DEFAULT_MODEL

# ---------- tabs ----------
tab_detect, tab_reply = st.tabs(["Detect emotion", "Draft a reply"])

# ----- Tab 1: detector -----
with tab_detect:
    st.text_area("Your sentence (English)", key="user_text", max_chars=500, height=110,
                 placeholder="e.g. i feel so happy today")
    st.caption("Or start from an example")
    example_chips(DETECT_EXAMPLES, "dx", "user_text")

    if st.button("Detect emotion", type="primary", use_container_width=True):
        result = predict(st.session_state.get("user_text", ""), model)
        if result is None:
            st.info("Type a sentence that contains some letters.")
        else:
            emotion, confidence, probs = result
            note = "The model is not very sure. The text may express mixed emotions." if confidence < 0.5 else ""
            emotion_card(emotion, confidence, note)
            st.caption("How likely each emotion is")
            prob_bars(probs, emotion)

# ----- Tab 2: LLM reply assistant -----
with tab_reply:
    st.markdown(
        '<div class="flow">'
        '<div class="flow-step"><b><span class="flow-n">1</span>You paste a message</b>One you received from someone.</div>'
        '<div class="flow-step"><b><span class="flow-n">2</span>Tool call</b>The AI runs your trained emotion model.</div>'
        '<div class="flow-step"><b><span class="flow-n">3</span>Reply drafted</b>Written in a tone that fits the emotion.</div>'
        '</div>', unsafe_allow_html=True)

    st.text_area("Message you received", key="reply_text", max_chars=500, height=110,
                 placeholder="e.g. i am so tired of being ignored")
    st.caption("Or start from an example")
    example_chips(REPLY_EXAMPLES, "rx", "reply_text")

    mode = st.radio("Prompting style", llm.MODES, index=2, horizontal=True)
    st.caption(MODE_HELP[mode])

    col_a, col_b = st.columns(2)
    run_one = col_a.button("Draft reply", type="primary", use_container_width=True)
    run_all = col_b.button("Compare all 3 styles", use_container_width=True)

    if run_one or run_all:
        message = st.session_state.get("reply_text", "").strip()
        api_key = get_api_key()
        if not clean_text(message):
            st.info("Paste a message that contains some letters.")
        elif not api_key:
            st.warning("Add your Groq API key in .streamlit/secrets.toml, or paste it in the sidebar.")
        else:
            try:
                client = get_client(api_key)
                modes = llm.MODES if run_all else (mode,)
                outputs = []
                with st.spinner("Asking the model..."):
                    for m in modes:
                        outputs.append(llm.generate_reply(client, message, m, detect_for_tool, llm_model))
            except Exception as exc:
                st.error(llm.friendly_error(exc))
            else:
                first = outputs[0]
                if first["emotion"]:
                    emotion_card(first["emotion"], first["confidence"])
                    with st.expander("See the tool call"):
                        st.markdown(
                            f'<div class="toolcall">detect_emotion(text="{esc(message[:80])}")\n'
                            f'-> emotion: {esc(str(first["emotion"]))}, confidence: {first["confidence"]}</div>',
                            unsafe_allow_html=True)
                else:
                    st.caption("The AI did not call the emotion tool for this message.")

                if run_all:
                    cols = st.columns(3)
                    for col, out in zip(cols, outputs):
                        with col:
                            bubble(out["reply"], out["emotion"], label=out["mode"])
                else:
                    bubble(first["reply"], first["emotion"], label="Suggested reply")
                    st.code(first["reply"], language=None, wrap_lines=True)

                history = st.session_state.setdefault("history", [])
                for out in outputs:
                    history.append({"message": message, "mode": out["mode"],
                                    "emotion": out["emotion"], "reply": out["reply"]})
                del history[:-10]

    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"Recent replies ({len(history)})"):
            for h in reversed(history[-5:]):
                st.markdown(f"{EMOJI.get(h['emotion'] or '', '')} **{h['mode']}**")
                st.text(h["message"])
                st.write(h["reply"])
                st.divider()
