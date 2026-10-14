"""Step 4 - GenAI layer: emotion-aware reply assistant.

The LLM receives a message the user got from someone else. It first CALLS A TOOL
(`detect_emotion`, which runs your trained Scikit-learn model), then writes a reply
in a tone that fits the detected emotion.

Three prompting styles are supported so you can compare them:
  zero-shot : instructions only
  one-shot  : instructions + 1 worked example
  few-shot  : instructions + 3 worked examples

This file does not import Streamlit, so you can also test it from a terminal.
Needs:  pip install groq   and an API key in the GROQ_API_KEY environment variable.
"""
import json

DEFAULT_MODEL = "openai/gpt-oss-120b"   # alternative: "openai/gpt-oss-20b" (faster). Model lists change, so
                                        # this can be overridden in the app sidebar under Advanced.
MODES = ("zero-shot", "one-shot", "few-shot")

# Tried first when the account has them; everything else the account offers is listed after these.
PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
_NOT_CHAT = ("whisper", "orpheus", "playai", "guard", "tts", "safeguard", "compound", "embed")

# ---- Tool definition (what the LLM sees) -------------------------------------------------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "detect_emotion",
        "description": ("Classify the emotion of an English message as joy, sadness, anger, fear, love "
                        "or surprise, using a trained machine learning model. Returns the emotion and "
                        "a confidence between 0 and 1."),
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The message to classify."}},
            "required": ["text"],
        },
    },
}]

# ---- Prompts -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You help a person reply to a message they received.

Steps:
1. Call the detect_emotion tool on the received message. Always do this first.
2. Write a short reply (2 to 4 sentences) that the person can send back, in a tone that fits the emotion:
   - joy: share the happiness, keep it warm and light
   - sadness: be gentle and supportive, do not rush to fix things
   - anger: stay calm, acknowledge the feeling, do not argue
   - fear: be reassuring and steady
   - love: be sincere and affectionate
   - surprise: react naturally and ask a simple follow-up
3. If the confidence is below 0.5, keep the tone neutral and polite.

Write only the reply text, in English, with no preamble."""

EXAMPLES = [
    {
        "message": "i cannot believe you forgot my birthday again i am so angry",
        "emotion": "anger",
        "reply": ("I am sorry, you are right to be upset. I should have remembered, and I would like to "
                  "make it up to you. Can we talk tonight?"),
    },
    {
        "message": "i feel so lonely since i moved to this new city",
        "emotion": "sadness",
        "reply": ("That sounds really hard, and I am glad you told me. I am here whenever you want to talk. "
                  "Would a video call this weekend help?"),
    },
    {
        "message": "i just got the job i am so happy",
        "emotion": "joy",
        "reply": ("Congratulations, this is wonderful news! You worked hard for it and you deserve it. "
                  "Let's celebrate soon."),
    },
]


def _examples_block(n: int) -> str:
    parts = []
    for i, ex in enumerate(EXAMPLES[:n], start=1):
        parts.append(f"Example {i}\nReceived message: {ex['message']}\n"
                     f"Detected emotion: {ex['emotion']}\nReply: {ex['reply']}")
    return "\n\n".join(parts)


def build_system_prompt(mode: str) -> str:
    """zero-shot = 0 examples, one-shot = 1, few-shot = 3."""
    n = {"zero-shot": 0, "one-shot": 1, "few-shot": 3}[mode]
    if n == 0:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\nHere are worked examples of the expected style:\n\n{_examples_block(n)}"


# ---- API calls ---------------------------------------------------------------------------
def _extra(model: str) -> dict:
    """gpt-oss models are reasoning models; low effort keeps replies fast and saves free-tier tokens."""
    return {"reasoning_effort": "low"} if "gpt-oss" in model else {}


def make_client(api_key: str):
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The 'groq' package is missing. Run: pip install groq") from exc
    return Groq(api_key=api_key)


def list_chat_models(client) -> list:
    """Ask Groq which models THIS API key can use, keep chat models, best choices first."""
    ids = [m.id for m in client.models.list().data]
    chat = [i for i in ids if not any(w in i.lower() for w in _NOT_CHAT)]
    return [m for m in PREFERRED_MODELS if m in chat] + sorted(m for m in chat if m not in PREFERRED_MODELS)


def generate_reply(client, message: str, mode: str, detect_fn, model: str = DEFAULT_MODEL) -> dict:
    """Run the full flow and return {reply, emotion, confidence, tool_called, mode}.

    detect_fn(text) must return {"emotion": str, "confidence": float} or {"error": str}.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    messages = [
        {"role": "system", "content": build_system_prompt(mode)},
        {"role": "user", "content": f"Received message: {message}"},
    ]

    # Call 1: force the model to use the tool so the ML model is always part of the pipeline.
    try:
        first = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "detect_emotion"}},
            temperature=0.2, extra_body=_extra(model),
        )
    except Exception as exc:
        if "tool_choice" not in str(exc).lower():
            raise
        # Some models reject a forced function; let the model decide (the prompt tells it to call the tool).
        first = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto",
            temperature=0.2, extra_body=_extra(model),
        )
    assistant_msg = first.choices[0].message
    tool_calls = assistant_msg.tool_calls or []

    emotion, confidence = None, None
    if tool_calls:
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in tool_calls],
        })
        for tc in tool_calls:
            if tc.function.name == "detect_emotion":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = detect_fn(args.get("text") or message)
                emotion = result.get("emotion", emotion)
                confidence = result.get("confidence", confidence)
            else:
                result = {"error": f"unknown tool {tc.function.name}"}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    # Call 2: the model now writes the reply using the tool result (tools disabled).
    final = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="none", temperature=0.7,
        extra_body=_extra(model),
    )
    return {
        "reply": (final.choices[0].message.content or "").strip(),
        "emotion": emotion,
        "confidence": confidence,
        "tool_called": bool(tool_calls),
        "mode": mode,
    }


def friendly_error(exc: Exception) -> str:
    """Turn an API exception into a message the user can act on."""
    text = str(exc)
    low = text.lower()
    if "401" in low or "invalid api key" in low or "invalid_api_key" in low:
        return "The API key was rejected. Check that you copied the full key."
    if "429" in low or "rate limit" in low or "rate_limit" in low:
        return "Rate limit reached. Wait a minute and try again, or use one prompting style instead of all 3."
    if "connection" in low or "timed out" in low or "timeout" in low:
        return "Could not reach Groq. Check your internet, VPN or firewall and try again."
    if "model" in low and any(w in low for w in ("not found", "decommissioned", "does not exist",
                                                   "not available", "access", "permission")):
        return ("This model is not available for your account. Pick another one in the sidebar under "
                f"Advanced, for example openai/gpt-oss-20b.\n\nDetails: {text[:300]}")
    return f"The API request failed: {text[:400]}"


if __name__ == "__main__":
    # Quick terminal test:  set GROQ_API_KEY, then run  python llm_assistant.py
    import os

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set the GROQ_API_KEY environment variable first.")

    import joblib
    from data_cleaning import clean_text

    bundle = joblib.load("models/emotion_model.joblib")
    pipe = bundle["pipeline"]

    def detect(text):
        cleaned = clean_text(text)
        probs = pipe.predict_proba([cleaned])[0]
        i = probs.argmax()
        return {"emotion": pipe.classes_[i], "confidence": round(float(probs[i]), 3)}

    out = generate_reply(make_client(key), "i am terrified about my exam tomorrow", "few-shot", detect)
    print(out)
