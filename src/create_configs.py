import yaml
import os

for file in os.listdir('data/vocab_txts'):
    print(file)
    with open('config.yaml') as cnf:
        config = yaml.safe_load(cnf)
    config['data']['vocab_path'] = './data/vocab_txts/' + file
    print('configs/' + file.split('_splits.')[0] + '.yaml')
    with open('configs/' + file.split('_splits.')[0] + '.yaml', 'w') as out:
        yaml.dump(config, out)