# RNN from Scratch — Derivation & Implementation

A vanilla **Recurrent Neural Network built from scratch in NumPy** — no deep-learning
framework for the model itself. This repository has two parts:

1. **The theory** — a complete, hand-derived account of how an RNN works: the forward
   recurrence, the softmax + cross-entropy gradient, and full **Backpropagation Through
   Time (BPTT)**, with every step shown explicitly and illustrated.
2. **A full-stack sentiment app** — the same vanilla RNN applied to a real task:
   classifying product-review sentiment (negative / neutral / positive). It is implemented
   **three ways on identical data** — from scratch in NumPy (the derivation in Part 1,
   turned into code), and in **PyTorch** and **TensorFlow** — across four text encoders
   (word2vec, fastText, GloVe, BERT), then served through a **FastAPI** backend and a
   **React** UI that shows every model's prediction + confidence side by side.

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
- [Part 2 — Sentiment Classification App (Full-Stack)](#part-2--sentiment-classification-app-full-stack)
  - [What it does](#what-it-does)
  - [Project structure](#project-structure)
  - [1. Build the models (one command)](#1-build-the-models-one-command)
  - [2. How the encoders stay small](#2-how-the-encoders-stay-small)
  - [3. Run the app locally](#3-run-the-app-locally)
  - [Deploying](#deploying)
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

# Part 2 — Sentiment Classification App (Full-Stack)

Part 1 derives a vanilla RNN. Part 2 turns that derivation into a working app: the same
RNN is implemented from scratch in NumPy (plus PyTorch and TensorFlow versions), trained
to classify review sentiment, and served behind a small web UI.

## What it does

Given a product review, it predicts the sentiment — **negative / neutral / positive** —
and shows how twelve models compare on the same sentence: **four text encoders**
(word2vec · fastText · GloVe · BERT) × **three implementations** of the same vanilla RNN:

| Implementation | File | Notes |
|---|---|---|
| **PyTorch** | `model_artifacts_generation.py` | `nn.RNN`, 2 layers + dropout, last-real-word readout |
| **TensorFlow** | `model_artifacts_generation.py` | `SimpleRNN` → softmax |
| **Manual (NumPy)** | `manual_rnn.py` | from scratch: forward, BPTT, Adam, gradient clipping — the Part 1 derivation, applied to classification |

A **FastAPI** backend loads all the trained models and a **React** frontend sends a
review to it, then displays each model's predicted label, confidence, and full class
probabilities, plus a consensus vote across all models.

## Project structure

```
Vanilla-RNN/
├── code/
│   ├── model_building/                 # produces the models
│   │   ├── data_generation.py          # 1. download + split reviews -> data/raw/
│   │   ├── encoder.py                  # 2. build+trim encoders, encode splits
│   │   ├── model_artifacts_generation.py  # 3. train PyTorch + TF + manual RNNs
│   │   ├── manual_rnn.py               #    the from-scratch NumPy RNN (used by step 3)
│   │   └── run_pipeline.py             #    runs steps 1-3 end to end
│   └── backend/                        # serves the models
│       ├── predictor.py                # text -> per-model predictions (no web framework)
│       ├── app.py                      # FastAPI: POST /predict, GET /health, /info
│       └── requirements.txt
├── frontend/                           # Vite + React UI
└── data/                               # raw splits, trimmed encoders, embeddings, model artifacts
```

(See [`code/README.md`](code/README.md) for the per-file details.)

## 1. Build the models (one command)

```bash
cd code/model_building
python run_pipeline.py                  # data -> encoder -> model
# reuse existing data + encoders, just retrain:
python run_pipeline.py --skip data encoder
```

This writes 12 artifacts to `data/model_artifacts/` — per encoder:
`pytorch_<enc>.pt`, `tensorflow_<enc>.keras`, and `manual_<enc>.npz`.

## 2. How the encoders stay small

The pretrained encoders are gigabytes. `encoder.py` downloads each one, **trims it to our
dataset's vocabulary in memory, and saves only the small copy** (`data/text_encoder_small/`,
a few MB each) — the full files are never written to disk. BERT (`bert-base-uncased`) loads
straight from HuggingFace at runtime. So the repo only commits the small encoders + the
trained models; the full encoders and intermediate embeddings stay out of git.

## 3. Run the app locally

Two terminals, backend first:

```bash
# Terminal 1 — backend
cd code/backend
pip install -r requirements.txt          # fastapi + uvicorn (ML deps from the project venv)
uvicorn app:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev                              # open http://localhost:5173
```

The backend warms up all encoders + models on startup; the Vite dev server proxies
`/api/*` to it. Type a review, hit **Analyze sentiment**, and compare the models.

## Deploying (Streamlit + CI/CD)

The repo ships a Streamlit front end (`streamlit_app.py`) that reuses the same prediction
core as the FastAPI backend — `backend/predictor.py` has no web framework in it, so both
import it directly. Because the committed encoders are small and the artifacts are in the
repo, the app deploys from a normal GitHub clone with **no large-file hosting**.

**Try it locally:**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py            # http://localhost:8501
```

**Deploy to Streamlit Community Cloud (this is the CD):**

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo/branch
   and set **Main file path** = `streamlit_app.py`.
3. Deploy. From then on **every push to the branch auto-redeploys** the app — that's the
   continuous-delivery half, handled by Streamlit Cloud (no deploy keys needed).

**CI (the gate before deploy):** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs on every push/PR — it byte-compiles the code and verifies the committed model
artifacts the app needs are actually present, so a broken commit fails CI instead of
shipping a broken app.

**Fitting the free-tier memory budget.** Loading PyTorch + TensorFlow + BERT at once
exceeds Streamlit's free tier, so the deploy trims what it loads via env vars (set at the
top of `streamlit_app.py`, overridable in Streamlit **Settings → Secrets**):

| Variable | Default (deploy) | Effect |
|---|---|---|
| `VRNN_ENCODERS` | `word2vec,fasttext,glove` | skips BERT (the ~400 MB transformer) |
| `VRNN_FRAMEWORKS` | `pytorch,tensorflow,manual` | which models to run |

Imports are lazy, so a skipped piece is never loaded (and `transformers` isn't even needed
when BERT is off — it's in `requirements-dev.txt`, not the deploy `requirements.txt`). If
the app still OOMs, set `VRNN_FRAMEWORKS=pytorch,manual` in Secrets to drop TensorFlow — no
code change required.

---

## Reference

The architecture diagrams and the overall framing of the forward/backward passes follow
the **[DeepLearning.AI Sequence Models course](https://www.coursera.org/learn/nlp-sequence-models)**
on Coursera (taught by Andrew Ng). The from-scratch NumPy implementation and the
hand-worked gradient derivations in this repository are built on the notation and
intuition from that course.
