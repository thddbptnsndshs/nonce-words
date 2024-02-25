from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
import random
import logging

from utils import to_matrix

with open('config.yaml') as cnf:
    config = yaml.safe_load(cnf)

logger = logging.getLogger(__name__)


class CharModel(nn.Module):
    def __init__(self, charlist: List[str]):
        super().__init__()
        self.charlist = charlist
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
        prefix_ix = torch.as_tensor(to_matrix([prefix], self.char2id), dtype=torch.int64).to(config['device'])
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


def configure_model_and_optim(charlist):
    model = CharModel(charlist)
    optim = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    model.train()
    model.to(config['device'])
    return model, optim


def evaluate_on_prefixes(model, wordlist: List[str]):
    test_prefixes = ['_' + random.choice(wordlist)[:config['eval']['prefix_length']]
                     for _ in range(config['eval']['num_prefixes'])]
    for prefix in test_prefixes:
        for _ in range(config['eval']['n_trials']):
            logging.info(model.generate(prefix, **config['generate']))
