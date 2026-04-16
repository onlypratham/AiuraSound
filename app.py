import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.recommender import (
    load_uri_dataset,
    recommend_songs_by_label,
    label_to_mood_name,
    mood_name_to_label,
)

from src.chatbot import generate_chat_response


# =========================================================
# Paths and loaders
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILE = MODELS_DIR / "mood_pipeline.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"


@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Convert JSON keys back to int for label_to_mood if needed
    label_to_mood = {int(k): v for k, v in metadata.get("label_to_mood", {}).items()}
    feature_columns = metadata["feature_columns"]

    return model, metadata, label_to_mood, feature_columns


@st.cache_data
def load_uri_data():
    return load_uri_dataset()


model, metadata, label_to_mood, feature_columns = load_model_and_metadata()
df_uri = load_uri_data()


# =========================================================
# Simple text -> mood mapping and feature profiles
# =========================================================

def map_text_to_mood_name(text: str) -> str:
    """
    Basic keyword-based mapping from free text to mood name.
    """
    text = text.lower()

    mood_keywords = {
        "sad": ["sad", "down", "depressed", "cry", "heartbroken", "lonely", "upset"],
        "happy": ["happy", "excited", "great", "awesome", "joyful", "good", "glad"],
        "energetic": ["energetic", "pumped", "motivated", "workout", "active", "dance"],
        "calm": ["calm", "relaxed", "peaceful", "chill", "soft", "quiet", "sleep"],
    }

    for mood, words in mood_keywords.items():
        if any(word in text for word in words):
            return mood

    # Default fallback
    return "calm"


def build_feature_vector_for_mood(mood_name: str) -> dict:
    """
    Create a representative feature vector for a given mood name,
    matching the model's feature columns.
    """
    profiles = {
        "happy": {
            "duration (ms)": 180000.0,
            "danceability": 0.8,
            "energy": 0.8,
            "loudness": -5.0,
            "speechiness": 0.08,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "liveness": 0.2,
            "valence": 0.85,
            "tempo": 125.0,
            "spec_rate": 3.0e-7,
        },
        "sad": {
            "duration (ms)": 210000.0,
            "danceability": 0.4,
            "energy": 0.3,
            "loudness": -12.0,
            "speechiness": 0.05,
            "acousticness": 0.6,
            "instrumentalness": 0.0,
            "liveness": 0.15,
            "valence": 0.2,
            "tempo": 75.0,
            "spec_rate": 3.0e-7,
        },
        "energetic": {
            "duration (ms)": 200000.0,
            "danceability": 0.85,
            "energy": 0.95,
            "loudness": -4.0,
            "speechiness": 0.09,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "liveness": 0.3,
            "valence": 0.75,
            "tempo": 140.0,
            "spec_rate": 3.0e-7,
        },
        "calm": {
            "duration (ms)": 240000.0,
            "danceability": 0.5,
            "energy": 0.4,
            "loudness": -10.0,
            "speechiness": 0.04,
            "acousticness": 0.5,
            "instrumentalness": 0.1,
            "liveness": 0.1,
            "valence": 0.5,
            "tempo": 90.0,
            "spec_rate": 3.0e-7,
        },
    }

    base = profiles.get(mood_name, profiles["calm"])

    # Ensure we only keep columns the model expects, and fill missing ones with 0
    feature_vec = {col: base.get(col, 0.0) for col in feature_columns}
    return feature_vec


def explain_mood(user_text: str, mood: str) -> str:
    text = user_text.lower()

    mood_keywords = {
        "sad": ["sad", "down", "depressed", "cry", "heartbroken", "lonely", "empty", "worthless", "anxious", "worried", "upset"],
        "happy": ["happy", "excited", "great", "awesome", "joyful", "good", "glad", "love", "crush"],
        "energetic": ["energetic", "pumped", "motivated", "workout", "active", "dance", "party"],
        "calm": ["calm", "relaxed", "peaceful", "chill", "soft", "quiet", "sleep", "tired"]
    }

    detected_words = [w for w in mood_keywords.get(mood, []) if w in text]

    if detected_words:
        words_str = ", ".join(detected_words[:3])
        return f"I predicted **{mood}** because your input included words like *{words_str}*, which are linked to this mood."
    else:
        return f"I predicted **{mood}** based on the overall tone of your message, even though no strong keywords were found."

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="AiuraSound - Mood-Based Music Recommender",
    layout="centered",
)

st.title("AiuraSound 🎵")
st.subheader("Mood-Based Music Recommender with AI Chatbot")

st.write(
    "Type how you're feeling, and I'll predict a mood, recommend some songs, "
    "and ask an AI assistant to briefly explain the choices and share a few gentle tips."
)

user_text = st.text_area(
    "Describe your current mood:",
    placeholder="Example: I feel low and tired but want something gentle and calming.",
    height=120,
)



if st.button("Get recommendations"):
    if not user_text.strip():
        st.warning("Please describe your mood first.")
    else:
        text_lower = user_text.lower()

        # 1) Strong rules first
        if any(word in text_lower for word in ["love", "in love", "crush", "romantic"]):
            base_mood = "happy"
        elif any(word in text_lower for word in ["workout", "gym", "run", "dance", "party", "jump"]):
            base_mood = "energetic"
        elif any(word in text_lower for word in ["sleep", "sleepy", "tired", "relax", "relaxed", "calm", "peaceful", "chill", "soft", "quiet", "cozy"]):
            base_mood = "calm"
        elif any(word in text_lower for word in ["sad", "down", "depressed", "cry", "heartbroken", "lonely", "empty", "worthless", "anxious", "worried", "upset"]):
            base_mood = "sad"
        else:
            # 2) Tie-breaker based on positive vs negative words
            positive_words = ["love", "happy", "great", "amazing", "awesome", "excited", "joy", "crush", "grateful"]
            negative_words = ["sad", "lonely", "tired", "anxious", "angry", "upset", "depressed", "worried", "stressed"]

            pos = sum(w in text_lower for w in positive_words)
            neg = sum(w in text_lower for w in negative_words)

            if pos > neg and pos > 0:
                base_mood = "happy"
            elif neg > pos and neg > 0:
                base_mood = "sad"
            else:
                # Neutral fallback: assume calm rather than sad
                base_mood = "calm"
                
        predicted_mood_name = base_mood
        pred_label = mood_name_to_label(base_mood)
        # Show predicted mood
        st.markdown("### Predicted Mood")
        st.success(f"{predicted_mood_name}")

        # Recommend songs
        recs = recommend_songs_by_label(df_uri, label=pred_label, n_songs=10)
        st.markdown("### Recommended Songs")
        if recs.empty:
            st.warning("No songs found.")
        else:
            st.dataframe(recs, use_container_width=True)
        st.markdown("### Why this mood?")
        explanation = explain_mood(user_text, predicted_mood_name)
        st.info(explanation)
