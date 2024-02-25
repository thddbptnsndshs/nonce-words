from tqdm import trange
import torch
from typing import Dict
from dataset import configure_dataloaders
from model import configure_model_and_optim, evaluate_on_prefixes
from utils import loss, compute_mask, extract_lang_name
import logging
import yaml
import numpy as np
import wandb
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default='config.yaml')
    args = parser.parse_args()
    with open(args.config_path) as cnf:
        config = yaml.safe_load(cnf)

    wandb.login(key='7a9437f52186ae00051016cfdf42d1ae9ac2a248')
    run = wandb.init(project=config['wandb']['project'], name=extract_lang_name(config['data']['vocab_path']))

    logging.basicConfig(filename=config['log_path'],
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    train_loader, test_loader, wordlist, charlist = configure_dataloaders(config)
    logging.info('train and test dataloaders ready')
    logging.info(f'train sample: {next(iter(train_loader))}')

    id2char: Dict[int, str] = dict(zip(range(len(charlist)), charlist))
    char2id: Dict[str, int] = dict(zip(charlist, range(len(charlist))))

    model, optim = configure_model_and_optim(charlist)
    logging.info('model and optimizer ready')

    th, vh = [], []
    for epoch in trange(config['train']['epochs']):
        train_history = []
        val_history = []
        val_perp_history = []
        for i, x in enumerate(train_loader):
            x = x.to(config['device'])
            logits = model(x[:, :-1]).permute(0, 2, 1)
            answers = x[:, 1:]
            mask = compute_mask(x[:, 1:], char2id)
            l_t = loss(logits, answers, mask)
            train_history.append(l_t.detach().cpu().numpy())
            optim.zero_grad()
            l_t.backward()
            optim.step()
        th.append((epoch, np.mean(train_history)))
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(config['device'])
                logits = model(x[:, :-1]).permute(0, 2, 1)
                answers = x[:, 1:]
                mask = compute_mask(x, char2id)[:, 1:]
                l_v = loss(logits, answers, mask)
                val_history.append(l_v.detach().cpu().numpy())
                perp = torch.exp(l_v)
                perp = perp.detach().cpu().numpy()
                val_perp_history.append(perp)
        vh.append((epoch, np.mean(val_history)))
        logging.info(f'perplexity on epoch {epoch}: {np.mean(val_perp_history)}')
        logging.info(f'train loss on epoch {epoch}: {np.mean(train_history)}')
        logging.info(f'val loss on epoch {epoch}: {np.mean(val_history)}')

        wandb.log({
            'val/perplexity': np.mean(val_perp_history),
            'train/loss': np.mean(train_history),
            'val/loss': np.mean(val_history),
        })

        logging.info('starting evaluation on prefixes')
        evaluate_on_prefixes(model, wordlist)

    model.eval()
    model_path = 'models/' + extract_lang_name(config['data']['vocab_path']) + '.pt'
    # torch.save(model, 'models/' + model_path)
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')

    artifact.add_file(model_path)
    run.log_artifact(artifact)
    run.finish()
