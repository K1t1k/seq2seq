import os
import logging

import torch
import numpy as np

from models.rnn import CharModel
from settings import Settings

# TODO реализовать предикт через консольную утилиту

device = Settings.DEVICE
data_dir = os.path.join('../data', Settings.DATASET)
text_path = os.path.join(data_dir, 'chehov.txt')


raw_text = open(text_path, mode='r').read()
logging.debug('Length of text: {} characters'.format(len(raw_text)))

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars, n_vocab = len(raw_text), len(chars)

model = CharModel(n_vocab=n_vocab)

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

# randomly generate a prompt

raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-Settings.SEQ_SIZE)
prompt = raw_text[start:start+Settings.SEQ_SIZE]
pattern = [char_to_int[c] for c in prompt]

model.to(device)
model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x.to(device))
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")