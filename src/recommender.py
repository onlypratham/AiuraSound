from pathlib import Path
from typing import List, Optional
import pandas as pd


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

URI_DATA_FILE = DATA_DIR / "278k_labelled_uri.csv"


# =========================================================
# Load URI dataset
# =========================================================
def load_uri_dataset() -> pd.DataFrame:
    """
    Load the 278k_labelled_uri dataset.

    Expected columns (from your sample):
    - Unnamed: 0 (index, will be dropped)
    - duration (ms)
    - danceability
    - energy
    - loudness
    - speechiness
    - acousticness
    - instrumentalness
    - liveness
    - valence
    - tempo
    - spec_rate
    - labels  (0 = sad, 1 = happy, 2 = energetic, 3 = calm)
    - uri     (spotify:track:...)
    """
    df = pd.read_csv(URI_DATA_FILE)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


# =========================================================
# Core recommendation function
# =========================================================
def recommend_songs_by_label(
    df_uri: pd.DataFrame,
    label: int,
    n_songs: int = 10,
    shuffle: bool = True,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Recommend songs for a given numeric label.

    Parameters
    ----------
    df_uri : DataFrame
        DataFrame loaded from 278k_labelled_uri.csv.
    label : int
        Predicted label (0 = sad, 1 = happy, 2 = energetic, 3 = calm).
    n_songs : int, optional
        Number of songs to return (default 10).
    shuffle : bool, optional
        If True, randomly sample songs of this label.
        If False, return the first n_songs (e.g., for deterministic behavior).
    seed : int or None, optional
        Random seed for reproducible sampling when shuffle=True.

    Returns
    -------
    DataFrame
        Recommended songs with useful columns including uri.
    """
    if "labels" not in df_uri.columns:
        raise ValueError("Column 'labels' not found in URI dataset.")

    label_df = df_uri[df_uri["labels"] == label].copy()

    if label_df.empty:
        return pd.DataFrame(columns=df_uri.columns)

    # Choose which columns to return to the UI
    display_cols: List[str] = [
        "uri",
        "duration (ms)",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "spec_rate",
        "labels"
    ]
    display_cols = [c for c in display_cols if c in label_df.columns]

    if shuffle:
        # Randomly sample up to n_songs for variety
        n = min(n_songs, len(label_df))
        recs = label_df.sample(n=n, random_state=seed)[display_cols]
    else:
        # Deterministic: just take the first n_songs
        recs = label_df.head(n_songs)[display_cols]

    return recs.reset_index(drop=True)


# =========================================================
# Helper: convert label -> mood name (for consistency)
# =========================================================
LABEL_TO_MOOD = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm"
}

MOOD_TO_LABEL = {v: k for k, v in LABEL_TO_MOOD.items()}


def label_to_mood_name(label: int) -> str:
    """
    Convert numeric label to human-readable mood name.
    """
    return LABEL_TO_MOOD.get(label, f"unknown_{label}")


def mood_name_to_label(mood: str) -> int:
    """
    Convert mood name to numeric label.
    Raises KeyError if mood not found.
    """
    mood_lower = mood.strip().lower()
    return MOOD_TO_LABEL[mood_lower]
