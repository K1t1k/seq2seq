import os
import logging

import numpy as np
import torch
from torch import optim
from torch.utils import data

from torch import nn

from models.rnn import CharModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

N_EPOCHS = 50
BATCH_SIZE = 64
BUFFER_SIZE = 10000
SEQ_SIZE = 64
DATASET = 'shakespeare'

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

data_dir = os.path.join('../data', DATASET)
text_path = os.path.join(data_dir, 'input.txt')


raw_text = open(text_path, mode='r').read()
logging.debug('Length of text: {} characters'.format(len(raw_text)))


chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
 
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

logging.debug('{} unique characters'.format(len(chars)))
logging.debug('text_as_int length: {}'.format(len(raw_text)))

# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, n_chars - SEQ_SIZE, 1):
    seq_in = raw_text[i:i + SEQ_SIZE]
    seq_out = raw_text[i + SEQ_SIZE]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
logging.debug(f"Total Patterns: {n_patterns}")

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, SEQ_SIZE, 1).to(device)
X = X / float(n_vocab)
y = torch.tensor(dataY).to(device)
logging.debug(f"X.shape: {X.shape}, y.shape: {y.shape}")

model = CharModel(n_vocab=n_vocab).to(device)
unoptimized_model = model
model = torch.compile(model) # requires PyTorch 2.0

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=BATCH_SIZE)

best_model = None
best_loss = np.inf
logging.debug("start train")
for epoch in range(N_EPOCHS):
    train_losses = []
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    logging.debug("Epoch %d: [train] Cross-entropy: %.4f" % (epoch, np.mean(train_losses, axis=0)))
    # Validation
    # TODO добавить валидационную выборку
    model.eval()
    val_losses = []
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            l = loss_fn(y_pred, y_batch)
            val_losses.append(l.item())
            loss += l
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        logging.debug("Epoch %d: [validation] Cross-entropy: %.4f" % (epoch, np.mean(val_losses, axis=0)))

torch.save([best_model, char_to_int], "single-char.pth")