"""
src/utils/beam_search.py

Caption generation for trained models, batched across many images at once
(the old version called the model once per beam per word per image, which
took close to an hour per config on the full test set).

  CaptionGenerator.greedy(states)          best next word at every step
  CaptionGenerator.beam(states, k)         keeps the k most probable partial captions
  CaptionGenerator.beam_trace(state, k)    every beam at every step, for the beam-tree figure (S4)
  CaptionGenerator.word_probabilities(...) top-5 candidate words at each step (dashboard)

The image encoder runs once per image; only the text decoder runs per step.
Scores are summed log-probabilities, as in Kavitha & Karpagam (2025);
length_penalty > 0 divides by length**length_penalty instead.
"""

import numpy as np

from src.models.caption_model import encoder_and_decoder


class CaptionGenerator:
    def __init__(self, model, tokenizer, max_length: int):
        self.encoder, self.decoder = encoder_and_decoder(model)
        self.tok = tokenizer
        self.seq_len = max_length - 1
        self.start = tokenizer.word2idx["<start>"]
        self.end = tokenizer.word2idx["<end>"]
        self.pad = tokenizer.word2idx["<pad>"]

    # ------------------------------------------------------------------ helpers
    def encode_images(self, features: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Image features (pooled or conv4) -> decoder initial states (N, units)."""
        out = [self.encoder.predict_on_batch(np.asarray(features[i:i + batch_size], np.float32))
               for i in range(0, len(features), batch_size)]
        return np.concatenate(out, axis=0)

    def _probs_at(self, states: np.ndarray, seqs: np.ndarray, pos) -> np.ndarray:
        """Next-word distribution after reading seqs up to (and including) position pos."""
        probs = self.decoder.predict_on_batch([states, seqs])  # (B, T, V)
        if np.isscalar(pos):
            return probs[:, pos, :]
        return probs[np.arange(len(seqs)), pos, :]

    def to_words(self, seq) -> str:
        out = []
        for t in seq:
            t = int(t)
            if t == self.end:
                break
            if t not in (self.start, self.pad):
                out.append(self.tok.idx2word.get(t, "<unk>"))
        return " ".join(out)

    # ------------------------------------------------------------------ greedy
    def greedy(self, states: np.ndarray, chunk: int = 512) -> list[str]:
        captions = []
        for i in range(0, len(states), chunk):
            s = states[i:i + chunk]
            seqs = np.full((len(s), self.seq_len), self.pad, np.int32)
            seqs[:, 0] = self.start
            done = np.zeros(len(s), bool)
            for t in range(self.seq_len - 1):
                probs = self._probs_at(s, seqs, t)
                probs[:, [self.pad, self.start]] = -1.0   # never generate these mid-caption
                nxt = probs.argmax(-1).astype(np.int32)
                nxt[done] = self.pad
                seqs[:, t + 1] = nxt
                done |= nxt == self.end
                if done.all():
                    break
            captions += [self.to_words(q[1:]) for q in seqs]
        return captions

    # ------------------------------------------------------------------ beam
    def beam(self, states: np.ndarray, k: int = 5, length_penalty: float = 0.0,
             chunk: int = 100) -> list[str]:
        if k == 1:
            return self.greedy(states)
        captions = []
        for i in range(0, len(states), chunk):
            captions += self._beam_chunk(states[i:i + chunk], k, length_penalty)[0]
        return captions

    def _beam_chunk(self, s: np.ndarray, k: int, length_penalty: float, trace: bool = False):
        B, T = len(s), self.seq_len
        st = np.repeat(s, k, axis=0)                                   # (B*k, units)
        seqs = np.full((B * k, T), self.pad, np.int32)
        seqs[:, 0] = self.start
        scores = np.full((B, k), -np.inf)
        scores[:, 0] = 0.0                                              # one live beam at the start
        finished = np.zeros((B, k), bool)
        lengths = np.zeros((B, k), np.int32)
        history = []

        for t in range(T - 1):
            logp = np.log(self._probs_at(st, seqs, t) + 1e-12).reshape(B, k, -1)  # (B, k, V)
            V = logp.shape[-1]
            logp[..., [self.pad, self.start]] = -np.inf   # never generate these mid-caption
            # A finished beam can only "continue" with padding, at no cost.
            logp[finished] = -np.inf
            logp[finished, self.pad] = 0.0
            cand = (scores[..., None] + logp).reshape(B, k * V)
            top = np.argpartition(-cand, k - 1, axis=1)[:, :k]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(cand, top, 1), 1), 1)
            src_beam, word = top // V, (top % V).astype(np.int32)

            rows = (np.arange(B)[:, None] * k + src_beam).reshape(-1)
            seqs = seqs[rows].copy()
            seqs[:, t + 1] = word.reshape(-1)
            prev_finished = np.take_along_axis(finished, src_beam, 1)
            lengths = np.take_along_axis(lengths, src_beam, 1) + (~prev_finished)
            finished = prev_finished | (word == self.end)
            scores = np.take_along_axis(cand, top, 1)
            if trace:
                history.append({"parent": src_beam[0].tolist(), "word": word[0].tolist(),
                                "score": scores[0].tolist(), "finished": finished[0].tolist()})
            if finished.all():
                break

        final = scores / np.maximum(lengths, 1) ** length_penalty if length_penalty else scores
        best = final.argmax(1)
        best_seqs = seqs.reshape(B, k, T)[np.arange(B), best]
        return [self.to_words(q[1:]) for q in best_seqs], history, final

    def beam_trace(self, state: np.ndarray, k: int = 3) -> dict:
        """All beams at every step for ONE image — data for the beam-search tree (S4)."""
        caps, history, final = self._beam_chunk(state[None], k, 0.0, trace=True)
        for step in history:
            step["word"] = [self.tok.idx2word.get(w, "<unk>") for w in step["word"]]
        return {"caption": caps[0], "steps": history, "final_scores": final[0].tolist()}

    # ------------------------------------------------------------------ inspection
    def word_probabilities(self, state: np.ndarray, caption: str, top: int = 5) -> list[dict]:
        """For each word of `caption`, the model's top-`top` candidates at that step."""
        words = caption.split() + ["<end>"]
        seq = np.full((1, self.seq_len), self.pad, np.int32)
        seq[0, 0] = self.start
        ids = [self.tok.word2idx.get(w, self.tok.word2idx["<unk>"]) for w in words]
        for j, w in enumerate(ids[:self.seq_len - 1]):
            seq[0, j + 1] = w
        probs = self.decoder.predict_on_batch([state[None], seq])[0]
        out = []
        for j, w in enumerate(words[:self.seq_len - 1]):
            p = probs[j]
            best = np.argsort(-p)[:top]
            out.append({"word": w, "prob": float(p[ids[j]]),
                        "top": [(self.tok.idx2word.get(int(b), "<unk>"), float(p[b])) for b in best]})
        return out
