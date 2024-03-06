# Import the W&B Python Library and log into W&B
import os

import wandb
from sweep_train import update_config_and_train
import argparse


def objective(config):
    score = update_config_and_train(config)
    return score


def main():
    wandb.init(project="nonce-lstm-sweeps")
    return objective(wandb.config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default='config.yaml')
    arguments = parser.parse_args()
    wandb.login(key=os.environ['WANDB_TOKEN'])

    sweep_configuration = {
        "name": arguments.config_path.split('/')[-1].split('.')[0],
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "odds"},
        "parameters": {
            'config_path': {"values": [arguments.config_path]},
            'embedding_dim': {"values": [64, 256]},
            'hidden_size': {"values": [128, 512]},
            'lr': {"values": [0.00001, 0.0001, 0.001]},
            'num_layers': {'min': 2, 'max': 10},
            'min_word_length': {'min': 1, 'max': 5},
            'max_word_length': {'min': 7, 'max': 15},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="nonce-lstm-sweeps")

    wandb.agent(sweep_id, function=main, count=10)
