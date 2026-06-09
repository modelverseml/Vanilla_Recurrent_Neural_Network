# RNN from Scratch — Derivation & Implementation

A vanilla **Recurrent Neural Network built from scratch in NumPy** — no deep-learning
framework for the model itself. This repository has two halves:

1. **The theory** — a complete, hand-derived account of how an RNN works: the forward
   recurrence, the softmax + cross-entropy gradient, and full **Backpropagation Through
   Time (BPTT)**, with every step shown explicitly and illustrated.
2. **The code** — that derivation turned directly into a readable, stacked (multi-layer)
   NumPy implementation, trained with BPTT and used to generate text character-by-character
   and word-by-word. The same architecture is also rebuilt in **TensorFlow/Keras** and
   **PyTorch**, and all three are compared side by side on the same data.

> Educational project: the goal is to make the mechanics of an RNN explicit and
> readable, not to be fast or state-of-the-art.

---

# Part 1 — How an RNN Works (Derivation)

A complete mathematical derivation of forward propagation and backpropagation through
time (BPTT) for a vanilla RNN, including the softmax gradient, the cross-entropy loss
gradient, and the vector/matrix gradient rules.

> **Convention.** This derivation uses the standard **column-vector** form, e.g.
> `a = tanh(Wax·x + Waa·a_prev + ba)` and `∂L/∂Wax = (∂L/∂a_raw)·xᵀ`. The code uses the
> equivalent **batch-first / row-vector** layout (data shaped `(m, T_x, n_x)`, so
> `a = tanh(x·Wax + a_prev·Waa + ba)` with weights stored transposed). Every formula
> below still holds — the two forms are exact transposes of each other, so the gradients
> are identical and only the orientation differs.

## Table of Contents

- [RNN from Scratch — Derivation \& Implementation](#rnn-from-scratch--derivation--implementation)
- [Part 1 — How an RNN Works (Derivation)](#part-1--how-an-rnn-works-derivation)
  - [Table of Contents](#table-of-contents)
  - [1. RNN Architecture Overview](#1-rnn-architecture-overview)
  - [2. Forward Propagation](#2-forward-propagation)
  - [3. Softmax — Definition \& Gradient](#3-softmax--definition--gradient)
    - [Case i: j = i (diagonal)](#case-i-j--i-diagonal)
    - [Case ii: j ≠ i (off-diagonal)](#case-ii-j--i-off-diagonal)
  - [4. Loss Function — Cross-Entropy](#4-loss-function--cross-entropy)
  - [5. Gradient of Loss w.r.t. Logits](#5-gradient-of-loss-wrt-logits)
    - [Case i: j = m](#case-i-j--m)
    - [Case ii: j ≠ m](#case-ii-j--m)
  - [6. Gradient of Vectors and Matrices](#6-gradient-of-vectors-and-matrices)
  - [7. Backpropagation Through Time (BPTT)](#7-backpropagation-through-time-bptt)
    - [Output layer gradients](#output-layer-gradients)
    - [Hidden state gradients](#hidden-state-gradients)
    - [Weight gradients](#weight-gradients)
  - [8. Summary of Gradient Equations](#8-summary-of-gradient-equations)
- [Part 2 — The Code](#part-2--the-code)
  - [Pipeline](#pipeline)
  - [Project structure](#project-structure)
    - [`rnn_scratch_multi_layer.py` — the from-scratch model](#rnn_scratch_multi_layerpy--the-from-scratch-model)
    - [`rnn_tensorflow.py` / `rnn_pytorch.py` — framework versions](#rnn_tensorflowpy--rnn_pytorchpy--framework-versions)
    - [`compare.py` — side-by-side comparison](#comparepy--side-by-side-comparison)
    - [`utils.py` — data \& inference helpers](#utilspy--data--inference-helpers)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Reference](#reference)

---

## 1. RNN Architecture Overview

A vanilla RNN processes a sequential input `x⟨t⟩` and maintains a hidden state `a⟨t⟩`
that is carried across time steps. The same cell — the same weights — is applied at
every step, and the hidden state is the only channel through which information from the
past reaches the present.

![Unrolled RNN across time steps](Images/RNN.png)

**Parameters (shared across all time steps):**
- `Wax` — weight matrix: input → hidden
- `Waa` — weight matrix: hidden → hidden (recurrent)
- `Wya` — weight matrix: hidden → output
- `ba`, `by` — bias vectors

---

## 2. Forward Propagation

At each time step `t` the cell mixes the current input `x⟨t⟩` with the previous hidden
state `a⟨t-1⟩`, squashes it through `tanh`, and projects the result to an output.

![Single RNN cell — forward pass](Images/rnn_forward_pass.png)

**Hidden state (pre-activation):**
```
a_raw⟨t⟩ = Waa · a⟨t-1⟩ + Wax · x⟨t⟩ + ba
```

**Hidden state (post-activation):**
```
a⟨t⟩ = tanh(a_raw⟨t⟩)
```

**Output logits:**
```
y⟨t⟩ = Wya · a⟨t⟩ + by
```

**Output probabilities (softmax):**
```
ŷ⟨t⟩ = softmax(y⟨t⟩)
```

Running this cell for every step of the sequence — feeding each step's hidden state
into the next — is the full forward pass:

![Forward pass unrolled over the whole sequence](Images/rnn_forward_sequence.png)

---

## 3. Softmax — Definition & Gradient

The softmax of logit vector `y` at index `i` is:

$$s_i = \frac{e^{y_i}}{\sum_{k=1}^{n} e^{y_k}}$$

We can write this as `s_i = h(y) / g(y)` where:

$$h(y) = e^{y_i}, \qquad g(y) = \sum_{k=1}^{n} e^{y_k}$$

The derivative with respect to `y_j` (quotient rule):

$$\frac{\partial s_i}{\partial y_j} = \frac{h'(y)\, g(y) - g'(y)\, h(y)}{(g(y))^2}$$

We need:

$$\frac{\partial h(y)}{\partial y_j} = h'(y) = e^{y_i} \quad \text{(if } i = j\text{, else 0 → constant)}$$

$$\frac{\partial g(y)}{\partial y_j} = \frac{\partial}{\partial y_j} \sum_{k=1}^{n} e^{y_k} = e^{y_j}$$

### Case i: j = i (diagonal)

When `i = j`, `h(y) = e^{y_i}` and `g'(y) = e^{y_i}`:

$$\frac{\partial s_i}{\partial y_j} = \frac{e^{y_i} \cdot \sum e^{y_k} - e^{y_i} \cdot e^{y_i}}{(\sum e^{y_k})^2}$$

$$= \frac{e^{y_i}}{\sum e^{y_k}} \left(1 - \frac{e^{y_j}}{\sum e^{y_k}}\right)$$

$$\boxed{\frac{\partial s_i}{\partial y_j} = s_i (1 - s_j)} \quad \text{when } j = i$$

### Case ii: j ≠ i (off-diagonal)

When `i ≠ j`, `h'(y) = 0` (since `e^{y_i}` does not depend on `y_j`):

$$\frac{\partial s_i}{\partial y_j} = \frac{0 - e^{y_j} \cdot e^{y_i}}{(\sum e^{y_k})^2} = -s_i \cdot s_j$$

$$\boxed{\frac{\partial s_i}{\partial y_j} = -s_i s_j} \quad \text{when } j \neq i$$

**Combined Jacobian of softmax:**

$$\frac{\partial s_i}{\partial y_j} = \begin{cases} s_i(1 - s_j) & \text{if } j = i \\ -s_i s_j & \text{if } j \neq i \end{cases}$$

---

## 4. Loss Function — Cross-Entropy

For a correct class index `m`, the cross-entropy loss is:

$$\ell = -\log(s_m)$$

where:

$$s_m = \frac{e^{y_m}}{\sum_{k} e^{y_k}}$$

The gradient with respect to `s_m`:

$$\frac{\partial \ell}{\partial s_m} = -\frac{1}{s_m}$$

---

## 5. Gradient of Loss w.r.t. Logits

By the chain rule, the loss gradient flows back through the softmax to the logits:

![Loss gradient through softmax and -log](Images/loss_gradient.webp)

$$\frac{\partial \ell}{\partial y_j} = \frac{\partial \ell}{\partial s_m} \cdot \frac{\partial s_m}{\partial y_j}$$

### Case i: j = m

Using `∂s_m/∂y_j = s_m(1 - s_j)`:

$$\frac{\partial \ell}{\partial y_j} = -\frac{1}{s_m} \cdot s_m(1 - s_j) = -(1 - s_j) = s_j - 1$$

$$\boxed{\frac{\partial \ell}{\partial y_j} = s_m - 1} \quad \text{if } j = m$$

### Case ii: j ≠ m

Using `∂s_m/∂y_j = -s_m · s_j`:

$$\frac{\partial \ell}{\partial y_j} = -\frac{1}{s_m} \cdot (-s_m \cdot s_j) = s_j$$

$$\boxed{\frac{\partial \ell}{\partial y_j} = s_j} \quad \text{if } j \neq m$$

**Combined:**

$$\frac{\partial \ell}{\partial y_j} = \begin{cases} s_m - 1 & \text{if } j = m \\ s_j & \text{if } j \neq m \end{cases}$$

> **Intuition:** This is simply `ŷ - one_hot(true_label)` — the predicted probability
> vector minus the ground-truth indicator. Elegant! This is exactly the
> `y_pred - Y` you'll see in the code.

---

## 6. Gradient of Vectors and Matrices

For a linear transformation `y = Wx`, the gradients are:

![Vector / matrix gradients for y = Wx](Images/vector_gradients.webp)

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$$

$$\frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial y}$$

**Intuition:** The weight gradient is the outer product of the upstream gradient and the
input. The input gradient backpropagates the upstream error through the transpose of the
weight matrix. These two rules are all we need to differentiate every linear step in the
RNN.

---

## 7. Backpropagation Through Time (BPTT)

BPTT applies the chain rule to the unrolled network, walking the sequence in reverse and
accumulating gradients into the shared weights. A single cell's backward pass routes the
incoming hidden-state gradient through `tanh` and out to each parameter and to the
previous step:

![Single RNN cell — backward pass](Images/rnn_cell_backward.png)

### Output layer gradients

Given `y⟨t⟩ = Wya · a⟨t⟩ + by`:

$$\frac{\partial L}{\partial y^{\langle t \rangle}} = \hat{y}^{\langle t \rangle} - \mathbf{1}[\text{true label}] \quad \text{(from softmax + cross-entropy above)}$$

$$\frac{\partial L}{\partial W_{ya}} = \frac{\partial L}{\partial y^{\langle t \rangle}} \cdot (a^{\langle t \rangle})^T$$

$$\frac{\partial L}{\partial a^{\langle t \rangle}}\bigg|_{\text{from output}} = W_{ya}^T \cdot \frac{\partial L}{\partial y^{\langle t \rangle}}$$

### Hidden state gradients

The total gradient at `a⟨t⟩` receives contributions from both the current output and the
next time step:

$$\frac{\partial L}{\partial a^{\langle t \rangle}} = W_{ya}^T \cdot \frac{\partial L}{\partial y^{\langle t \rangle}} + W_{aa}^T \cdot \frac{\partial L}{\partial a_{\text{next}}}$$

Through the tanh activation (where `a⟨t⟩ = tanh(a_raw⟨t⟩)`):

$$\frac{\partial L}{\partial a_{\text{raw}}^{\langle t \rangle}} = \frac{\partial L}{\partial a^{\langle t \rangle}} \cdot (1 - (a^{\langle t \rangle})^2)$$

since `d/dx [tanh(x)] = 1 - tanh²(x)`.

### Weight gradients

$$\frac{\partial L}{\partial W_{aa}} = \frac{\partial L}{\partial a_{\text{raw}}^{\langle t \rangle}} \cdot (a^{\langle t-1 \rangle})^T$$

$$\frac{\partial L}{\partial W_{ax}} = \frac{\partial L}{\partial a_{\text{raw}}^{\langle t \rangle}} \cdot (x^{\langle t \rangle})^T$$

$$\frac{\partial L}{\partial a^{\langle t-1 \rangle}} = W_{aa}^T \cdot \frac{\partial L}{\partial a_{\text{raw}}^{\langle t \rangle}}$$

Collecting every per-cell parameter gradient, expanded through the `tanh`:

![Per-cell parameter gradients with tanh expanded](Images/rnn_cell_backprop_gradients.png)

The gradient flows back in time via `a⟨t-1⟩`, enabling the RNN to learn long-range
dependencies — in theory. In practice the repeated multiplication by `Waa` leads to the
**vanishing / exploding gradient problem**.

Putting the whole computational graph together — from the loss through softmax,
cross-entropy, the output projection, the `tanh`, and back to `Wxh`, `Whh` and the
previous hidden state — gives the complete BPTT picture:

![Full BPTT computational-graph derivation](Images/back-propagration-full-derivation.webp)

---

## 8. Summary of Gradient Equations

| Quantity | Gradient |
|---|---|
| Softmax `∂sᵢ/∂yⱼ` (j=i) | `sᵢ(1 − sⱼ)` |
| Softmax `∂sᵢ/∂yⱼ` (j≠i) | `−sᵢsⱼ` |
| Loss `∂ℓ/∂yⱼ` (j=m) | `sₘ − 1` |
| Loss `∂ℓ/∂yⱼ` (j≠m) | `sⱼ` |
| `∂L/∂Wya` | `(∂L/∂y) · aᵀ` |
| `∂L/∂a` (from output) | `Wyaᵀ · (∂L/∂y)` |
| `∂L/∂a_raw` | `(∂L/∂a) · (1 − a²)` |
| `∂L/∂Waa` | `(∂L/∂a_raw) · a⟨t−1⟩ᵀ` |
| `∂L/∂Wax` | `(∂L/∂a_raw) · x⟨t⟩ᵀ` |
| `∂L/∂a⟨t−1⟩` | `Waaᵀ · (∂L/∂a_raw)` |




---

# Part 2 — The Code

The implementation turns the derivation above directly into NumPy as a **stacked
(multi-layer) RNN**, trained with full BPTT and used to generate text both
character-by-character and word-by-word.

To put the from-scratch model in context, the same architecture is then built two more
ways — with **TensorFlow/Keras** and with **PyTorch** — and all three are compared on the
same data, architecture, and hyper-parameters. The notebook runs this 3-way comparison
across a one-hot character model and four word-embedding encoders (Word2Vec, pre-trained
GloVe, FastText, and a pre-trained BERT Transformer).

## Pipeline

```
raw text → vocabulary → sliding-window sequences → train/test split
        → train RNN (BPTT) → evaluate (train/test accuracy) → predict / generate
```

- **Character model** — inputs are one-hot character vectors; the RNN learns surface
  character statistics and generates text one character at a time.
- **Word model** — inputs are dense word-embedding vectors (Word2Vec / GloVe /
  FastText / BERT); the RNN predicts a probability distribution over the vocabulary and
  generates text one word at a time.
- **3-way comparison** — the manual NumPy RNN, a Keras model, and a PyTorch model are
  trained on the same split and compared by train/test accuracy and sample generations.

## Project structure

```
RNN/
├── rnn_scratch_multi_layer.py  # the from-scratch RNN: stacked layers, forward, loss, BPTT, training
├── rnn_tensorflow.py           # KerasRNN — same architecture/interface, built with TensorFlow/Keras
├── rnn_pytorch.py              # TorchRNN — same architecture/interface, built with PyTorch
├── compare.py                  # compare_models — train/test accuracy + generation across models
├── utils.py                    # data prep + split + evaluate + inference (predict_next, generate)
├── rnn_scratch_single_layer.py # the original single-layer RNN (kept for reference)
├── rnn_building_scratch.ipynb  # end-to-end walkthrough + 3-way comparison (char + 4 word encoders)
├── Images/                     # diagrams used in this README
├── requirements.txt            # Python dependencies
└── README.md
```

### `rnn_scratch_multi_layer.py` — the from-scratch model

`RNN` stacks one or more recurrent layers (set with `hidden_layers`, e.g. `(50,)` for one
layer or `(100, 64)` for two), carries a per-layer hidden state across `T_x` time steps,
and is trained with plain gradient descent over the gradients accumulated by BPTT. Each
method maps onto a section of the derivation above.

| Method | Role | Derivation |
| --- | --- | --- |
| `initialize_parameters` | per-layer weights `Wax[l], Waa[l], ba[l]` plus the output layer `Wya, by` | §1 |
| `layer_forward` | run one recurrent layer over the whole sequence | §2 |
| `rnn_forward` | stack the layers, then apply the output projection at every step | §2 |
| `rnn_cell_forward` | one timestep through the whole stack (used for text generation) | §2 |
| `compute_loss` | cross-entropy (classification) or MSE (regression) | §4 |
| `layer_backward` / `rnn_backward` | BPTT for one layer / down through the stack, with gradient clipping | §5–§7 |
| `update_parameters` | one gradient-descent step | — |
| `train` | the full loop: forward → loss → backward → update | — |
| `predict` | forward pass returning predictions `(m, T_x, n_y)` | §2 |

It supports two tasks:

- `task="classification"` — softmax output + cross-entropy loss (one-hot targets).
- `task="regression"` — linear output + mean-squared-error loss (real-valued targets).

In both cases the per-step output gradient reduces to `y_pred - Y`, exactly the
`ŷ − y_true` derived in §5.

> **Layout note.** The tensors use the standard **batch-first** layout
> `(m, T_x, n_x)`, so examples are rows and a step is `x @ Wax + a @ Waa + b`
> (weights to the *right* of the data). The Part 1 derivation writes the same
> step with column vectors as `Wax · x + Waa · a + b`; the two are transposes of
> each other and produce identical results.

### `rnn_tensorflow.py` / `rnn_pytorch.py` — framework versions

`KerasRNN` and `TorchRNN` mirror the from-scratch model: they take the **same constructor
inputs** (`X, Y, hidden_layers, learning_rate, iterations, task`) and expose the same
`train()` / `predict()` interface, so the helpers in `utils.py` and `compare.py` work on
them unchanged. They are standalone — each trains natively (stacked `SimpleRNN` /
`nn.RNN`, Adam optimizer) and `predict` runs its own framework forward, returning the
same `(m, T_x, n_y)` layout. They do **not** share weights with the manual model.

> Because the frameworks optimize with **Adam** and the from-scratch model with plain
> **SGD + gradient clipping**, their learned weights and accuracies differ — this is a
> realistic "library vs scratch" comparison, not a bit-for-bit match.

### `compare.py` — side-by-side comparison

`compare_models(models, X_train, Y_train, X_test, Y_test, ...)` takes a
`{name: trained_model}` mapping and prints a **train/test accuracy** table (or MSE for
regression) plus a sample generation for each model. Training is done by the caller, so
you can compare all three models or just the manual one.

### `utils.py` — data & inference helpers

- `generate_dataset(...)` — slides a window of length `T_x` over the corpus and builds
  the `(m, T_x, n_x)` input and `(m, T_x, n_y)` target tensors, in either character or
  word mode.
- `train_test_split(...)` — splits the sequences into train/test partitions.
- `evaluate(...)` — next-token accuracy (classification) or MSE (regression) on given data.
- `predict_next(...)` — one token in, the single most likely next token out (argmax).
- `generate(...)` — autoregressive generation, optionally sampling from the predicted
  distribution for more varied output.

`predict_next` / `generate` rely only on a model's `predict` method, so the same calls
drive the manual RNN and both framework wrappers identically.

**Tensor convention** used throughout:

| Symbol | Meaning |
| --- | --- |
| `n_x` | input feature size (vocab size for chars, vector size for words) |
| `n_y` | output feature size (vocab size) |
| `m`   | number of training sequences |
| `T_x` | time steps per sequence |

## Setup

```bash
# (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# register the environment as a Jupyter kernel (optional)
python -m ipykernel install --user --name=rnn-env --display-name="Python (rnn-env)"
```

> The notebook downloads pre-trained embeddings/models (GloVe via `gensim.downloader`,
> BERT via `transformers`) on first run, so the first execution needs an internet
> connection and some disk space.

## Usage

Open the notebook for the full walkthrough:

```bash
jupyter notebook rnn_building_scratch.ipynb
```

Or use the modules directly — character-level example:

```python
from rnn_scratch_multi_layer import RNN
from utils import generate_dataset, train_test_split, evaluate, predict_next, generate

text = "your training corpus here ..."

# build one-hot character sequences, then split into train/test
X, Y, char_to_index, index_to_char = generate_dataset(
    text, T_x=10, is_char=True, word_vectors=[], seq_length=25
)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# train a stacked RNN (hidden_layers sets the number and size of layers)
model = RNN(X_train, Y_train, hidden_layers=(50,), learning_rate=0.001, iterations=1000)
model.train()

# evaluate + generate
print("test accuracy:", evaluate(model, X_test, Y_test))
print(predict_next(model, char_to_index, index_to_char, seed_word="M", is_char=True))
print(generate(model, char_to_index, index_to_char, seed_word="A", num_words=100, is_char=True))
```

Word-level example (using gensim word vectors as the encoder):

```python
from gensim.models import Word2Vec

sentences = [s.split() for s in corpus_lines]
words = [w for s in sentences for w in s]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1).wv

X, Y, vocab_to_index, index_to_vocab = generate_dataset(words, T_x=5, is_char=False, word_vectors=w2v)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = RNN(X_train, Y_train, hidden_layers=(100, 64), learning_rate=0.01, iterations=2000, task="classification")
model.train()

print(predict_next(model, w2v, index_to_vocab, "Machine"))
print(generate(model, w2v, index_to_vocab, seed_word="Machine", num_words=10, sample=True))
```

Compare all three implementations on the same data:

```python
from rnn_scratch_multi_layer import RNN
from rnn_tensorflow import KerasRNN
from rnn_pytorch import TorchRNN
from compare import compare_models

cfg = dict(hidden_layers=(50,), learning_rate=0.001, iterations=1000, task="classification")
models = {
    "manual (numpy)": RNN(X_train, Y_train, **cfg),
    "tensorflow":     KerasRNN(X_train, Y_train, **cfg),
    "pytorch":        TorchRNN(X_train, Y_train, **cfg),
}
for m in models.values():
    m.train()

# train/test accuracy table + a sample generation from each model
compare_models(models, X_train, Y_train, X_test, Y_test,
               embedding=char_to_index, decoder=index_to_char,
               seed_word="M", is_char=True, num_gen=100)
```



---

## Reference

The architecture diagrams and the overall framing of the forward/backward passes follow
the **[DeepLearning.AI Sequence Models course](https://www.coursera.org/learn/nlp-sequence-models)**
on Coursera (taught by Andrew Ng). The from-scratch NumPy implementation and the
hand-worked gradient derivations in this repository are built on the notation and
intuition from that course.
