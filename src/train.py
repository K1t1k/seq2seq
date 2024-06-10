import os
import logging

import numpy as np
import torch
from torch import optim
from torch.utils import data

from torch import nn

from models.rnn import CharModel
from settings import Settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

# TODO работу с данными реализовать в классе Dataset
# TODO подавать предсказывать 1 символ по 1 символу
# TODO использовать hidden state

device = Settings.DEVICE
data_dir = os.path.join('../data', Settings.DATASET)
text_path = os.path.join(data_dir, 'chehov.txt')


raw_text = open(text_path, mode='r').read()
logging.debug('Length of text: {} characters'.format(len(raw_text)))

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars, n_vocab = len(raw_text), len(chars)

# prepare the dataset of input to output pairs encoded as integers
dataX, dataY = [], []
for i in range(0, n_chars - Settings.SEQ_SIZE, 1):
    seq_in = raw_text[i:i + Settings.SEQ_SIZE]
    seq_out = raw_text[i + Settings.SEQ_SIZE]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, Settings.SEQ_SIZE, 1).to(device)
X = X / float(n_vocab)
y = torch.tensor(dataY).to(device)
logging.debug(f"X.shape: {X.shape}, y.shape: {y.shape}")
loader_train = data.DataLoader(data.TensorDataset(X[:int(len(X) * .9)], y[:int(len(X) * .9)]), shuffle=True, batch_size=Settings.BATCH_SIZE)
loader_val = data.DataLoader(data.TensorDataset(X[int(len(X) * .9):], y[int(len(X) * .9):]), shuffle=True, batch_size=Settings.BATCH_SIZE)

model = CharModel(n_vocab=n_vocab, hidden_size=Settings.HIDDEN_SIZE, n_layer=Settings.N_LAYER).to(device)
unoptimized_model = model
model = torch.compile(model) # requires PyTorch 2.0
optimizer = optim.Adam(model.parameters(), lr=Settings.LR)
loss_fn = nn.CrossEntropyLoss()  # reduction="sum")


best_model = None
best_loss = np.inf
hidden_state = None
logging.debug("start train")
for epoch in range(Settings.N_EPOCHS):
    train_losses = []
    model.train()
    for X_batch, y_batch in loader_train:
        y_pred, hidden_state = model(X_batch, hidden_state)
        loss = loss_fn(y_pred, y_batch)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    logging.debug("Epoch %d: [train] Cross-entropy: %.4f" % (epoch, np.mean(train_losses, axis=0)))
    model.eval()
    val_losses = []
    loss = 0
    hidden_state = None
    with torch.no_grad():
        for X_batch, y_batch in loader_val:
            y_pred, hidden_state = model(X_batch, hidden_state)
            l = loss_fn(y_pred, y_batch)
            val_losses.append(l.item())
            loss += l
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        logging.debug("Epoch %d: [validation] Cross-entropy: %.4f" % (epoch, np.mean(val_losses, axis=0)))

torch.save([best_model, char_to_int], "single-char.pth")