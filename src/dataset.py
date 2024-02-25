from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from utils import open_vocab_file, build_char_list, to_matrix
import logging
import torch
from typing import Dict

logger = logging.getLogger(__name__)


class Words(Dataset):
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def __getitem__(self, idx):
        return self.wordlist[idx]

    def __len__(self):
        return len(self.wordlist)


def configure_dataloaders(config):
    wordlist = open_vocab_file(config)
    charlist = build_char_list(wordlist)
    input_length = max(32, len(max(wordlist, key=len)))
    logger.info(f'Number of unique characters: {len(charlist)}')
    logger.info(f'Max word length (32 if less): {input_length}')
    char2id: Dict[str, int] = dict(zip(charlist, range(len(charlist))))

    train, test = train_test_split(wordlist)
    train_dataset = Words(train)
    logging.info(f'length of train dataset: {len(train_dataset)}')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['train']['batch_size'],
                              collate_fn=lambda x: torch.tensor(to_matrix(x, char2id)),
                              shuffle=True)

    test_dataset = Words(test)
    logging.info(f'length of test dataset: {len(test_dataset)}')
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config['train']['batch_size'],
                             collate_fn=lambda x: torch.tensor(to_matrix(x, char2id)),
                             shuffle=True)
    return train_loader, test_loader, wordlist, charlist
