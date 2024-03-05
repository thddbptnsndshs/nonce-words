from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
import random
import logging
from torch.optim.lr_scheduler import LinearLR

from utils import to_matrix
from ld1nn import ld1nn

# with open('config.yaml') as cnf:
#     config = yaml.safe_load(cnf)

logger = logging.getLogger(__name__)


class CharModel(nn.Module):
    def __init__(self, charlist: List[str], config: Dict):
        super().__init__()
        self.charlist = charlist
        self.config = config
        self.char2id: Dict[str, int] = dict(zip(charlist, range(len(charlist))))
        self.embed = nn.Embedding(len(charlist), **config['model']['embedding'])
        self.lstm = nn.GRU(**config['model']['gru'])
        self.linear = nn.Linear(config['model']['gru']['hidden_size'], len(charlist))

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def get_next_tokens(self, prefix: str):
        prefix_ix = torch.as_tensor(to_matrix([prefix], self.char2id), dtype=torch.int64).to(self.config['device'])
        with torch.no_grad():
            probs = torch.softmax(self(prefix_ix)[0, -1], dim=-1).cpu().numpy()  # shape: [n_tokens]
        return dict(zip(self.charlist, probs))

    def generate(self, prefix='_', temperature=1.0, max_len=256):
        with torch.no_grad():
            while True:
                token_probs = self.get_next_tokens(prefix)
                tokens, probs = zip(*token_probs.items())
                if temperature == 0:
                    next_token = tokens[np.argmax(probs)]
                else:
                    probs = np.array([p ** (1. / temperature) for p in probs])
                    probs /= sum(probs)
                    next_token = np.random.choice(tokens, p=probs)

                prefix += next_token
                if next_token == '^' or len(prefix) > max_len: break
        return prefix
    
    def verbose_generate(self, prefix='_', temperature=1.0, max_len=256):
        x, y = [], []
        all_probs = []
        with torch.no_grad():
            while True:
                token_probs = self.get_next_tokens(prefix)
                tokens, probs = zip(*token_probs.items())
                if temperature == 0:
                    next_token = tokens[np.argmax(probs)]
                else:
                    probs = np.array([p ** (1. / temperature) for p in probs])
                    probs /= sum(probs)
                    next_token = np.random.choice(tokens, p=probs)
                    x.append(next_token)
                    y.append(probs[tokens.index(next_token)])
                    all_probs.append(probs)

                prefix += next_token
                if next_token == '^' or len(prefix) > max_len: break
        return prefix, x, y, all_probs


def configure_model_and_optim(charlist: List[str], config: Dict):
    model = CharModel(charlist, config)
    optim = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    model.train()
    model.to(config['device'])
    scheduler = LinearLR(optim, start_factor=0.5, total_iters=5)
    return model, optim, scheduler


def evaluate_on_prefixes(model, wordlist: List[str], config: Dict):
    output = []
    test_prefixes = ['_' + random.choice(wordlist)[:config['eval']['prefix_length']]
                     for _ in range(config['eval']['num_prefixes'])]
    for prefix in test_prefixes:
        for _ in range(config['eval']['n_trials']):
            generated_word = model.generate(prefix, **config['generate'])
            logging.info(generated_word)
            output.append(generated_word)

    return output

def evaluate_ld1nn(model, wordlist: List[str], config: Dict):
    generated_words = []
    wordlist_sample = random.sample(wordlist, config['eval']['ld1nn_sample_size']) # ld1nn takes too much time for the whole corpus
    for _ in range(config['eval']['ld1nn_sample_size']):
        generated_word = model.generate('_', **config['generate'])
        generated_words.append(generated_word[1:-1]) # remove BOS and EOS tokens
    return ld1nn(wordlist_sample, generated_words)
