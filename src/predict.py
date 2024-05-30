import os
import logging

import torch
import numpy as np

from models.rnn import CharModel 
 
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
seq_length = 100

model = CharModel(n_vocab=n_vocab)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)
 
# randomly generate a prompt

raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
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