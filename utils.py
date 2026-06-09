"""Data preparation and inference helpers for the from-scratch RNN.

This module sits between raw text and the multi-layer `RNN` class in
`rnn_scratch_multi_layer.py`:

* `generate_dataset` turns a corpus into the sliding-window training tensors the
  RNN expects, in either character mode (one-hot inputs) or word mode
  (pre-trained word-vector inputs).
* `train_test_split` carves those tensors into train and test partitions.
* `evaluate` scores a trained model on held-out data.
* `predict_next` / `generate` run a *trained* model forward to produce text.

Tensor convention (batch-first, shared with `rnn_scratch_multi_layer.RNN`):
    m   : number of training sequences (batch size)
    T_x : time steps per sequence
    n_x : input feature size  (vocab size for chars, vector size for words)
    n_y : output feature size (vocab size)
so inputs are (m, T_x, n_x) and targets are (m, T_x, n_y).
"""

import numpy as np


def generate_dataset(
        words_or_text,
        T_x,
        is_char=True,
        word_vectors=[],
        seq_length=10,
    ):
    """Build sliding-window training tensors from a corpus.

    A window of length ``T_x`` is slid over the corpus. For each window the input
    is the chunk of tokens and the target is that same chunk shifted one token to
    the left, so the model learns to predict the *next* token at every step.

    Args:
        words_or_text : the corpus -- a raw string (char mode) or a list of words
                        (word mode).
        T_x           : number of time steps per training sequence.
        is_char       : True  -> character-level model, inputs are one-hot vectors.
                        False -> word-level model, inputs are word-embedding vectors.
        word_vectors  : a gensim KeyedVectors / dict mapping word -> vector. Only
                        used (and required) when ``is_char`` is False.
        seq_length    : reserved tail margin for the character model so the last
                        windows never index past the end of the corpus.

    Returns:
        input_sequences  : inputs,  shape (m, T_x, n_x)
        output_sequences : one-hot targets, shape (m, T_x, n_y)
        vocab_to_index   : dict mapping token -> feature index
        index_to_vocab   : dict mapping feature index -> token (used to decode)
    """
    # In word mode, drop any token that the embedding doesn't know about -- there
    # is no vector to feed the network for an out-of-vocabulary word. gensim
    # exposes membership via `.key_to_index`; a plain dict is queried directly.
    if not is_char:
        try:
            words_or_text = [w for w in words_or_text if w in word_vectors.key_to_index]
        except AttributeError:
            words_or_text = [w for w in words_or_text if w in word_vectors]

    # Build the vocabulary: the sorted set of unique tokens, plus the two maps
    # that translate between a token and its column index in the one-hot tensors.
    vocabs = sorted(list(set(words_or_text)))
    vocab_to_index = {c: i for i, c in enumerate(vocabs)}
    index_to_vocab = {i: c for i, c in enumerate(vocabs)}

    if is_char:
        n_x = len(vocabs)       # input size  = vocab size (one-hot characters)
        n_y = len(vocabs)       # output size = vocab size
        m = len(words_or_text) - T_x - seq_length - 1   # number of windows that fit
    else:
        # Input feature size is the embedding dimension. gensim exposes it as
        # `.vector_size`; for a plain dict we read the length of any vector.
        try:
            n_x = word_vectors.vector_size
        except AttributeError:
            n_x = len(word_vectors[vocabs[0]])
        n_y = len(vocabs)
        m = len(words_or_text) - T_x - 1

    input_sequences = np.zeros((m, T_x, n_x))
    output_sequences = np.zeros((m, T_x, n_y))

    # Fill window i, step t with token (i+t); the target is the next token (i+t+1).
    for i in range(m):
        for t in range(T_x):
            if is_char:
                # One-hot encode the input character.
                input_sequences[i, t, vocab_to_index[words_or_text[i+t]]] = 1

            else:
                # Use the dense word vector as the input feature.
                input_sequences[i, t, :] = word_vectors[words_or_text[i+t]]
            # Targets are always one-hot over the vocabulary, in both modes.
            output_sequences[i, t, vocab_to_index[words_or_text[i+t+1]]] = 1

    print(f"Generated {m} training sequences of length {T_x} from a corpus of "
          f"{len(words_or_text)} tokens, with a vocabulary of {len(vocabs)} unique "
          f"{'characters' if is_char else 'words'}.")
    return input_sequences, output_sequences, vocab_to_index, index_to_vocab


def train_test_split(X, Y, test_size=0.2, shuffle=True, seed=0):
    """Split the generated tensors into train and test partitions.

    The sequences live along axis 0 (the ``m`` axis), so the split is taken
    over that axis and the same row indices are used for X and Y to keep
    each input paired with its target.

    Args:
        X         : input tensor,  shape (m, T_x, n_x).
        Y         : target tensor, shape (m, T_x, n_y).
        test_size : fraction of the ``m`` sequences to hold out for testing.
        shuffle   : shuffle the sequence order before splitting.
        seed      : RNG seed so the split is reproducible.

    Returns:
        X_train, X_test, Y_train, Y_test
    """
    m = X.shape[0]
    indices = np.arange(m)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)

    n_test = int(round(m * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    print(f"Split {m} sequences -> {X_train.shape[0]} train / {X_test.shape[0]} test.")
    return X_train, X_test, Y_train, Y_test


def evaluate(model, X, Y):
    """Score a trained model on data X/Y.

    For classification returns next-token accuracy (fraction of timesteps whose
    argmax prediction matches the one-hot target). For regression returns the
    mean-squared-error loss. Lower MSE / higher accuracy is better.
    """
    y_pred = model.predict(X)                       # (m, T_x, n_y)
    if model.task == "classification":
        # Compare predicted class (argmax over the feature axis) against the target.
        correct = np.argmax(y_pred, axis=-1) == np.argmax(Y, axis=-1)
        return float(np.mean(correct))
    return model.compute_loss(y_pred, Y)


def _encode_sequence(tokens, model, embedding, is_char):
    """Encode a list of tokens into one input sequence of shape (1, T, n_x).

    Char mode -> one-hot rows; word mode -> stacked embedding vectors. This is
    the same encoding `generate_dataset` produces, for a single sequence.
    """
    T = len(tokens)
    x = np.zeros((1, T, model.n_x))
    for t, tok in enumerate(tokens):
        if is_char:
            x[0, t, embedding[tok]] = 1.0          # one-hot the character
        else:
            x[0, t, :] = embedding[tok]            # dense word vector
    return x


def predict_next(model, embedding, decoder, seed_word, is_char=False):
    """Return the single most likely next token after ``seed_word`` (argmax).

    Works for any model exposing ``predict`` (the manual RNN and the TensorFlow
    / PyTorch wrappers all do), so the same call compares across frameworks.

    Args:
        model     : a trained model with a ``predict`` method and ``n_x``/``n_y``.
        embedding : how to turn a token into the model's input vector --
                    a ``vocab_to_index`` map in char mode, or a word-vector
                    lookup (gensim KeyedVectors / dict) in word mode.
        decoder   : ``index_to_vocab`` map, used to turn the predicted index
                    back into a token.
        seed_word : the token to condition on.
        is_char   : True for the character model, False for the word model.
    """
    x = _encode_sequence([seed_word], model, embedding, is_char)
    y_pred = model.predict(x)                      # (1, T, n_y)
    idx = int(np.argmax(y_pred[0, -1, :]))         # distribution at the last step
    return decoder[idx]


def generate(model, embedding, decoder, seed_word, num_words=50, is_char=False, sample=True):
    """Generate text autoregressively, starting from ``seed_word``.

    At each step the tokens generated so far are re-encoded and run through
    ``model.predict``; the distribution at the final timestep gives the next
    token, which is appended and fed back in. Because it relies only on
    ``predict``, the same routine drives the manual RNN and the TensorFlow /
    PyTorch wrappers identically (no per-model hidden-state handling).

    Args:
        model     : a trained model with a ``predict`` method.
        embedding : token -> input-vector lookup (see ``predict_next``).
        decoder   : ``index_to_vocab`` map for turning indices back into tokens.
        seed_word : the token to start generation from.
        num_words : how many tokens to generate after the seed.
        is_char   : True joins output with '' (chars), False joins with ' ' (words).
        sample    : True  -> draw the next token from the predicted distribution
                             (varied output);
                    False -> always take the argmax (deterministic, tends to repeat).
    """
    tokens = [seed_word]

    for _ in range(num_words):
        x = _encode_sequence(tokens, model, embedding, is_char)
        y_pred = model.predict(x)                  # (1, T, n_y)
        probs = y_pred[0, -1, :]                   # next-token scores at last step
        if sample:
            probs = np.clip(probs, 1e-12, None)
            probs = probs / probs.sum()            # renormalise to a valid pmf
            idx = int(np.random.choice(model.n_y, p=probs))
        else:
            idx = int(np.argmax(probs))
        tokens.append(decoder[idx])

    return ''.join(tokens) if is_char else ' '.join(tokens)