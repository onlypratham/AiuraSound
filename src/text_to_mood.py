from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

LABEL_TO_MOOD = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm"
}
MOOD_TO_LABEL = {v: k for k, v in LABEL_TO_MOOD.items()}

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_sentence_model = None
_faiss_index = None
_index_labels = None  # array of label ints (0–3)


def load_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(MODEL_NAME)
    return _sentence_model


def build_faiss_index():
    """
    Build a tiny semantic index of prototype sentences for each mood.
    Call this once at startup.
    """
    global _faiss_index, _index_labels

    if _faiss_index is not None:
        return _faiss_index, _index_labels

    model = load_sentence_model()

    examples = [
    # sad
    ("sad", "I feel lonely and down."),
    ("sad", "I am heartbroken and want to cry."),
    ("sad", "Everything feels heavy and depressing."),
    ("sad", "I have no energy and feel empty."),
    # happy
    ("happy", "I feel great and full of joy."),
    ("happy", "I am in love and excited about life."),
    ("happy", "Today I am really happy and optimistic."),
    ("happy", "I am smiling a lot and feel grateful."),
    # energetic
    ("energetic", "I am pumped up and full of energy."),
    ("energetic", "I want to dance and move a lot."),
    ("energetic", "I'm ready to work out and push hard."),
    ("energetic", "I feel hyper and can't sit still."),
    # calm
    ("calm", "I just want something relaxing and peaceful."),
    ("calm", "I feel calm and want soft, gentle music."),
    ("calm", "I'm tired and want to chill and unwind."),
    ("calm", "I am at peace and everything feels quiet.")
]


    sentences = [s for _, s in examples]
    moods = [m for m, _ in examples]
    labels = np.array([MOOD_TO_LABEL[m] for m in moods], dtype=np.int64)

    embeddings = model.encode(sentences, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _faiss_index = index
    _index_labels = labels

    return _faiss_index, _index_labels


def predict_mood_from_text(text: str, k: int = 5) -> str:
    if not text.strip():
        return "calm"

    model = load_sentence_model()
    index, index_labels = build_faiss_index()

    emb = model.encode([text], convert_to_numpy=True)
    D, I = index.search(emb, k)  # D = distances, I = indices

    neighbor_labels = index_labels[I[0]]

    # Majority vote
    values, counts = np.unique(neighbor_labels, return_counts=True)
    majority_label = int(values[np.argmax(counts)])

    return LABEL_TO_MOOD[majority_label]

