# Insights from nonce word generation with character-wise RNN

This is a repository with the source code for the training and inference of a character-wise RNN that generates nonce-words: words that comply with the phonological rules of some language but do not have a meaning, i.e. those are words that seem real but are not.

## Usage

All the training and hyperparameter tuning scripts log to wandb. In order to log in properly, set the environment variable with your wandb token first:

`export WANDB_TOKEN=<your_token>`

To train a model, specify a config file and run the `run_experiments.sh` script below. This will launch the `src/train.py` script with arguments coming from the config.

`bash run_experiments.sh <config_path>`

To tune hyperparameters, run `hyperparameter_search.sh`, which will launch `src/sweep_train.py` -- note that this is different from `src/train.py`.

`bash hyperparameter_search.sh <config_path>`

Since the model can be trained on many languages, there is also a script that edits the configs in bulk. Edit the `src/create_configs.py` file with your desired changes and run it as a Python script:

`python3 create_configs.py`

## Contents

- `paper/` contains the TeX source files for our paper on the project.
- `exp/` has the notebooks which were later refactored into python scripts.