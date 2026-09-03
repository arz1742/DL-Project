"""
src/utils/beam_search.py

Beam search decoding for caption generation, as used by both mother papers
(Kavitha & Karpagam's beam-search-optimized decoder, and the reconstruction
scoring in the CNN+GRU paper's multi-stage decoding).
"""

import numpy as np


def beam_search_decode(
    model,
    feature_vector: np.ndarray,
    tokenizer,
    max_length: int,
    beam_width: int = 5,
) -> str:
    """
    Generates a caption for a single image feature vector using beam search.

    model expects inputs [feature_vector_batch, partial_caption_batch] and
    outputs a softmax distribution over the vocabulary at each timestep,
    matching the models built in src/models/caption_model.py.
    """
    start_id = tokenizer.word2idx["<start>"]
    end_id = tokenizer.word2idx["<end>"]
    pad_id = tokenizer.word2idx["<pad>"]

    feature_batch = np.expand_dims(feature_vector, axis=0)

    # Each beam entry: (token_id_sequence, cumulative_log_prob)
    beams = [([start_id], 0.0)]

    for _ in range(max_length - 1):
        candidates = []
        for sequence, score in beams:
            if sequence[-1] == end_id:
                # Already finished — keep it as-is, don't extend further
                candidates.append((sequence, score))
                continue

            padded = sequence + [pad_id] * (max_length - len(sequence))
            padded = np.array(padded[:max_length]).reshape(1, -1)

            preds = model.predict([feature_batch, padded], verbose=0)[0]
            next_token_probs = preds

            top_k_ids = np.argsort(next_token_probs)[-beam_width:]
            for token_id in top_k_ids:
                prob = next_token_probs[token_id]
                log_prob = np.log(prob + 1e-10)
                candidates.append((sequence + [int(token_id)], score + log_prob))

        # Keep only the top `beam_width` candidates for the next round
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == end_id for seq, _ in beams):
            break

    best_sequence, _ = max(beams, key=lambda x: x[1])
    return tokenizer.decode(best_sequence, strip_special=True)


def greedy_decode(model, feature_vector: np.ndarray, tokenizer, max_length: int) -> str:
    """Faster, simpler alternative to beam search — useful as a baseline comparison."""
    start_id = tokenizer.word2idx["<start>"]
    end_id = tokenizer.word2idx["<end>"]
    pad_id = tokenizer.word2idx["<pad>"]

    feature_batch = np.expand_dims(feature_vector, axis=0)
    sequence = [start_id]

    for _ in range(max_length - 1):
        padded = sequence + [pad_id] * (max_length - len(sequence))
        padded = np.array(padded[:max_length]).reshape(1, -1)

        preds = model.predict([feature_batch, padded], verbose=0)[0]
        next_token_id = int(np.argmax(preds))
        sequence.append(next_token_id)

        if next_token_id == end_id:
            break

    return tokenizer.decode(sequence, strip_special=True)
