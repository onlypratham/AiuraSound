from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# Model loading (global, for reuse)
# =========================================================

MODEL_NAME = "google/flan-t5-small"  # small, good for instructions

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None


def load_local_chat_model():
    """
    Load the Flan-T5 model and tokenizer once and reuse them.
    """
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return _tokenizer, _model


# =========================================================
# Prompt helpers
# =========================================================

def _build_recommendations_summary(recs: pd.DataFrame, max_songs: int = 5) -> str:
    """
    Make a short text summary of recommended songs for the prompt.
    """
    if recs is None or recs.empty:
        return "No songs were recommended."

    recs_small = recs.head(max_songs)

    lines = []
    for _, row in recs_small.iterrows():
        uri = row.get("uri", "unknown_uri")
        energy = row.get("energy", None)
        valence = row.get("valence", None)
        danceability = row.get("danceability", None)

        pieces = [f"{uri}"]
        if energy is not None:
            pieces.append(f"energy={energy:.2f}")
        if valence is not None:
            pieces.append(f"valence={valence:.2f}")
        if danceability is not None:
            pieces.append(f"danceability={danceability:.2f}")

        lines.append(" | ".join(pieces))

    return "\n".join(lines)


def _build_prompt(user_message, predicted_mood, recommendations_df=None):
    recs_text = _build_recommendations_summary(recommendations_df)

    prompt = f"""
You are a friendly AI mood assistant.

The user's mood was predicted as: {predicted_mood}

User input: "{user_message}"

Recommended songs:
{recs_text}

Tasks:
1. Explain briefly WHY this mood was predicted (mention words from user input).
2. Suggest 2-3 simple ways to improve or support the mood.
3. If the user is asking a direct question, answer it clearly.

Keep response short (5-6 lines), supportive, and simple.
Avoid medical advice.
"""
    return prompt.strip()


# =========================================================
# Public function
# =========================================================

def generate_chat_response(
    user_message: str,
    predicted_mood: str,
    recommendations_df: Optional[pd.DataFrame] = None,
    max_new_tokens: int = 128
) -> str:
    """
    Generate a chatbot reply using a local Flan-T5 model (no API key).

    Parameters
    ----------
    user_message : str
        The user's question or comment.
    predicted_mood : str
        Mood name: 'sad', 'happy', 'energetic', or 'calm'.
    recommendations_df : DataFrame or None
        Recommended songs to reference in the explanation.
    max_new_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Generated reply text.
    """
    tokenizer, model = load_local_chat_model()

    prompt = _build_prompt(user_message, predicted_mood, recommendations_df)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()
