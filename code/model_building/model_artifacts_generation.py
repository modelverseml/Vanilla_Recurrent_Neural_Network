"""
Train a vanilla RNN to classify review sentiment (negative / neutral / positive).

We build the same model three ways -- PyTorch, TensorFlow, and a from-scratch
NumPy implementation (ManualRNN) -- train each on every encoder's data
(word2vec, fasttext, glove, bert), report the test accuracy, and save every
trained model into data/model_artifacts/ as its own file:

    pytorch_<enc>.pt      tensorflow_<enc>.keras      manual_<enc>.npz
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# NOTE: tensorflow is imported lazily (inside the TF functions) so this module
# can be imported without it -- the Streamlit deploy skips TF to save memory.

from encoder import load_embeddings, LABEL_TO_ID, WORD2VEC, FASTTEXT, GLOVE, BERT
from manual_rnn import ManualRNN

# where the trained models get saved
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_ROOT = REPO_ROOT / 'data' / 'model_artifacts'

ENCODERS = [WORD2VEC, FASTTEXT, GLOVE, BERT]

# ENCODERS = [FASTTEXT]
NUM_CLASSES = len(LABEL_TO_ID)  # 3: negative / neutral / positive

# training settings
HIDDEN_DIM = 256
NUM_LAYERS = 2      # how many RNN layers to stack (pytorch/tensorflow)
DROPOUT = 0.3       # applied between stacked RNN layers and in the classifier head
EPOCHS = 10
BATCH_SIZE = 64
MAX_WORDS = 30

# the from-scratch numpy RNN is a single-layer Elman cell, so a smaller hidden
# size keeps the pure-python BPTT fast while still training fine.
MANUAL_HIDDEN = 128

# fix the random state so that when we sweep learning rates the only thing
# changing is the lr, not the weight init or the data-shuffle order.
SEED = 42


def set_seed(seed=SEED):
    # re-seed every source of randomness so each training run starts identically
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # tensorflow not installed (e.g. lightweight deploy) -> skip

# learning rate per framework and per encoder, tune each one on its own.
# anything not listed here falls back to DEFAULT_LR.
DEFAULT_LR = 1e-3
LEARNING_RATES = {
    "pytorch": {
        WORD2VEC: 3e-3,
        FASTTEXT: 3e-3,
        GLOVE: 3e-3,
        BERT: 3e-3,
    },
    "tensorflow": {
        WORD2VEC: 3e-3,
        FASTTEXT: 3e-3,
        GLOVE: 3e-3,
        BERT: 3e-3,
    },
    "manual": {
        WORD2VEC: 5e-3,
        FASTTEXT: 5e-3,
        GLOVE: 5e-3,
        BERT: 5e-3,
    },
}


def get_lr(framework, encoder_name):
    # look up the learning rate for this framework + encoder, else default
    return LEARNING_RATES.get(framework, {}).get(encoder_name, DEFAULT_LR)


# ----------------------------- PyTorch -----------------------------

class VanillaRNNTorch(nn.Module):
    # plain (Elman) RNN -> take the last real word -> small MLP head -> 3 classes
    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        # dropout between RNN layers only kicks in with >1 layer
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # deeper classifier head instead of a single linear layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.rnn(x)      # out: (batch, seq_len, hidden)
        # titles are padded with zero vectors at the end, so the real words are
        # a prefix. a padded timestep is all-zeros -> build a mask of real words
        # and take the hidden state at the LAST real word, not at timestep -1
        # (which would be the state after feeding many zero pad vectors).
        mask = (x.abs().sum(dim=-1) > 0)          # (batch, seq_len) True for real words
        lengths = mask.sum(dim=1).clamp(min=1)    # at least 1 so empty titles don't break
        last_idx = lengths - 1                    # index of the last real word
        batch_idx = torch.arange(x.size(0), device=x.device)
        last = out[batch_idx, last_idx]           # (batch, hidden)
        return self.fc(last)                      # raw scores for each class


def evaluate_torch(model, X, y, device):
    # run the model and return the fraction of correct predictions
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float((preds == y).mean())


def train_torch(data, encoder_name):
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    X_test, y_test = data["test"]

    input_dim = X_train.shape[-1]  # 300 for w2v/fasttext/glove, 768 for bert
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed()  # same init + shuffle order for every lr we try
    model = VanillaRNNTorch(input_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
    lr = get_lr("pytorch", encoder_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"  [torch] lr={lr}")

    # feed the data in shuffled mini batches
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # remember the weights from the epoch with the best dev accuracy, so we test
    # and save that model instead of whatever the last epoch happened to land on
    best_dev_acc = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            # clip gradients -- vanilla RNNs are prone to exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        dev_acc = evaluate_torch(model, X_dev, y_dev, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            # copy to cpu so we keep the best weights even as training continues
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [torch] epoch {epoch + 1}/{EPOCHS}  dev_acc={dev_acc:.3f}")

    # restore the best-dev weights before testing/saving
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [torch] best dev_acc={best_dev_acc:.3f}")
    test_acc = evaluate_torch(model, X_test, y_test, device)

    # save the trained weights for this encoder
    out_file = ARTIFACT_ROOT / f"pytorch_{encoder_name}.pt"
    torch.save(model.state_dict(), out_file)
    print(f"  [torch] saved -> {out_file}")
    return test_acc


# ---------------------------- TensorFlow ----------------------------

def build_tf_model(input_dim, lr):
    import tensorflow as tf  # lazy: only needed when training the TF model
    # same shape of model: SimpleRNN -> Dense softmax over 3 classes
    model = tf.keras.Sequential([
        tf.keras.layers.Input((MAX_WORDS, input_dim)),
        tf.keras.layers.SimpleRNN(HIDDEN_DIM),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_tf(data, encoder_name):
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    X_test, y_test = data["test"]

    lr = get_lr("tensorflow", encoder_name)
    print(f"  [tf] lr={lr}")
    set_seed()  # same init + shuffle order for every lr we try
    model = build_tf_model(X_train.shape[-1], lr)
    model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,  # keep the output short, we print the accuracy ourselves
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # keras saves the whole model (architecture + weights) in one file
    out_file = ARTIFACT_ROOT / f"tensorflow_{encoder_name}.keras"
    model.save(out_file)
    print(f"  [tf] saved -> {out_file}")
    return float(test_acc)


# ------------------------------ manual ------------------------------

# the from-scratch numpy RNN lives in its own module (manual_rnn.py)
def train_manual(data, encoder_name):
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    X_test, y_test = data["test"]

    input_dim = X_train.shape[-1]
    lr = get_lr("manual", encoder_name)
    print(f"  [manual] lr={lr}")

    set_seed()  # same init + shuffle order as the other frameworks
    model = ManualRNN(input_dim, MANUAL_HIDDEN, NUM_CLASSES)
    model.fit(X_train, y_train, X_dev, y_dev,
              epochs=EPOCHS, batch_size=BATCH_SIZE, lr=lr)

    test_acc = model.accuracy(X_test, y_test)

    # save as a plain .npz of the weight matrices
    out_file = ARTIFACT_ROOT / f"manual_{encoder_name}.npz"
    model.save(out_file)
    print(f"  [manual] saved -> {out_file}")
    return test_acc


# ------------------------------- main -------------------------------

def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    results = {}  # encoder -> {framework: test accuracy}
    for encoder_name in ENCODERS:
        print(f"\n=== encoder: {encoder_name} ===")
        # load the pre-saved embeddings (run encoder.py first)
        data = load_embeddings(encoder_name)
        results[encoder_name] = {
            "pytorch": train_torch(data, encoder_name),
            "tensorflow": train_tf(data, encoder_name),
            "manual": train_manual(data, encoder_name),
        }

    # print a small table of test accuracies
    print("\n=== test accuracy ===")
    print(f"{'encoder':<10} {'pytorch':>8} {'tensorflow':>11} {'manual':>8}")
    for encoder_name, accs in results.items():
        def show(v):
            return f"{v:.3f}" if v is not None else "-"
        print(f"{encoder_name:<10} {show(accs['pytorch']):>8} "
              f"{show(accs['tensorflow']):>11} {show(accs['manual']):>8}")


if __name__ == "__main__":
    main()
