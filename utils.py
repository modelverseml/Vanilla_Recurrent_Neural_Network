"""Data preparation and inference helpers for the from-scratch RNN.

This module sits between raw text and the `RNN` class in `rnn_scratch.py`:

* `generate_dataset` turns a corpus into the sliding-window training tensors the
  RNN expects, in either character mode (one-hot inputs) or word mode
  (pre-trained word-vector inputs).
* `predict_next` / `generate` run a *trained* model forward to produce text.

Tensor convention (shared with `rnn_scratch.RNN`):
    n_x : input feature size  (vocab size for chars, vector size for words)
    n_y : output feature size (vocab size)
    m   : number of training sequences
    T_x : time steps per sequence
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
        input_sequences  : inputs,  shape (n_x, m, T_x)
        output_sequences : one-hot targets, shape (n_y, m, T_x)
        vocab_to_index   : dict mapping token -> column index
        index_to_vocab   : dict mapping column index -> token (used to decode)
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

    input_sequences = np.zeros((n_x, m, T_x))
    output_sequences = np.zeros((n_y, m, T_x))

    # Fill window i, step t with token (i+t); the target is the next token (i+t+1).
    for i in range(m):
        for t in range(T_x):
            if is_char:
                # One-hot encode the input character.
                input_sequences[vocab_to_index[words_or_text[i+t]], i, t] = 1

            else:
                # Use the dense word vector as the input feature.
                input_sequences[:, i, t] = word_vectors[words_or_text[i+t]]
            # Targets are always one-hot over the vocabulary, in both modes.
            output_sequences[vocab_to_index[words_or_text[i+t+1]], i, t] = 1

    print(f"Generated {m} training sequences of length {T_x} from a corpus of "
          f"{len(words_or_text)} tokens, with a vocabulary of {len(vocabs)} unique "
          f"{'characters' if is_char else 'words'}.")
    return input_sequences, output_sequences, vocab_to_index, index_to_vocab


def predict_next(model, embedding, decoder, seed_word, is_char=False):
    """Return the single most likely next token after ``seed_word`` (argmax).

    Args:
        model     : a trained ``RNN`` instance.
        embedding : how to turn a token into the model's input vector --
                    a ``vocab_to_index`` map in char mode, or a word-vector
                    lookup (gensim KeyedVectors / dict) in word mode.
        decoder   : ``index_to_vocab`` map, used to turn the predicted index
                    back into a token.
        seed_word : the token to condition on.
        is_char   : True for the character model, False for the word model.
    """
    if is_char:
        x = np.zeros((model.n_x, 1))              # one-hot vector for input character
        x[embedding[seed_word], 0] = 1
    else:
        x = embedding[seed_word].reshape(-1, 1)          # input word vector, shape (vector_size, 1)
    a_prev = np.zeros((model.n_a, 1))           # fresh hidden state
    _,_, y_pred = model.rnn_cell_forward(x, a_prev)   # softmax distribution over vocab
    idx = int(np.argmax(y_pred))
    return decoder[idx]                   # map local vocab index back to a word


def generate(model, embedding, decoder, seed_word, num_words=50, is_char=False, sample=True):
    """Generate text autoregressively, starting from ``seed_word``.

    Each predicted token is fed back in as the next input and the hidden state is
    carried forward, so the model continues its own sequence one token at a time.

    Args:
        model     : a trained ``RNN`` instance.
        embedding : token -> input-vector lookup (see ``predict_next``).
        decoder   : ``index_to_vocab`` map for turning indices back into tokens.
        seed_word : the token to start generation from.
        num_words : how many tokens to generate after the seed.
        is_char   : True joins output with '' (chars), False joins with ' ' (words).
        sample    : True  -> draw the next token from the predicted distribution
                             (varied output);
                    False -> always take the argmax (deterministic, tends to repeat).
    """
    if is_char:
        x = np.zeros((model.n_x, 1))              # one-hot vector for input character
        x[embedding[seed_word], 0] = 1
    else:
        x = embedding[seed_word].reshape(-1, 1)     # one input word vector

    a_prev = np.zeros((model.n_a, 1))           # hidden state carried across steps
    result = [seed_word]

    for _ in range(num_words):
        _, a_prev, y_pred = model.rnn_cell_forward(x, a_prev)   # one step forward
        if sample:
            idx = np.random.choice(model.n_y, p=y_pred.ravel())
        else:
            idx = np.argmax(y_pred)
        word = decoder[int(idx)]
        result.append(word)
        
        if is_char:
            x = np.zeros((model.n_x, 1))          # one-hot vector for input character
            x[embedding[word], 0] = 1
        else:
            x = embedding[word].reshape(-1, 1)      # feed predicted word back in as next input

    return ' '.join(result) if not is_char else ''.join(result)