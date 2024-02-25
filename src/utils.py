import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List
import re


def extract_lang_name(path):
    name: list = re.findall(r'vocab_txts/(.+?)_splits\.txt', path)
    if not name:
        return 'unk'
    else:
        return name[0]


def open_vocab_file(config):
    with open(config['data']['vocab_path'], 'r') as f:
        wordlist = f.read().split()
    if maxlen := config['data']['max_word_length']:
        if minlen := config['data']['min_word_length']:  # filter words by length
            wordlist = [word for word in wordlist if isinstance(word, str) and minlen < len(word) < maxlen]
        else:
            wordlist = [word for word in wordlist if isinstance(word, str) and len(word) < maxlen]
    if config['data']['frequency_sampling']:  # assumes that the words are ordered by descending frequency
        # zipf law distribution taken from
        # https://stackoverflow.com/questions/55518957/how-to-calculate-the-optimal-zipf-distribution-of-word-frequencies-in-a-text
        optimal_zipf = 1 / (np.array(list(range(1, (len(wordlist) + 1)))) * np.log(1.78 * len(wordlist)))  # 1.78
        optimal_zipf = np.round(optimal_zipf * len(wordlist)) + 1  # calculate frequency factor for sampling
        wordlist = sum([[word] * freq for word, freq in zip(wordlist, optimal_zipf)], [])  # concatenate words
        # multiplied by frequency factor
    return wordlist


def build_char_list(wordlist: List[str]):
    """
    Create list of characters to be used as tokens for the model
    :param wordlist: list of words in the vocabulary
    :return: list of token characters
    """
    charlist = set()
    for word in wordlist:
        charlist.update(word)
    charlist.add('_')  # begin char
    charlist.add('^')  # end character
    return sorted(charlist)


def to_matrix(lines, char2id, max_len=None, dtype=np.int64):
    """Casts a list of lines into torch-digestible matrix"""
    pad_begin = char2id['_']
    pad_end = char2id['^']

    max_len = (max_len or max(map(len, lines))) + 2
    lines_ix = np.full([len(lines), max_len], pad_end, dtype=dtype)
    lines_ix[:, 0] = pad_begin
    for i in range(len(lines)):
        line_ix = list(map(char2id.get, lines[i][:max_len]))
        lines_ix[i, 1:len(line_ix) + 1] = line_ix
    return lines_ix


def compute_mask(input_ix, char2id):
    eos_ix = char2id['^']
    """ compute a boolean mask that equals "1" until first EOS (including that EOS) """
    return F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True)


def loss(logits, answers, mask):
    """
    Modifies CrossEntropyLoss to take mask into account
    :param logits: model logits
    :param answers: targets
    :param mask: mask
    :return: masked loss, mean reduction
    """
    loss_fn = nn.CrossEntropyLoss(reduction='none')(logits, answers) * mask
    return loss_fn[loss_fn != 0.0].mean()
