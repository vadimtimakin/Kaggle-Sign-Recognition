import os
import time
import numpy as np

import json
import logging
import tg_logger

from config import config
from utils import set_seed
from train_functions import run

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(config.general.seed)

    if config.logging.telegram:
        with open('telegram_credits.json', 'r') as file:
            tokens = json.load(file)

        token = tokens["bot"]
        users = [tokens["user"]]

        logger = logging.getLogger('foo')
        logger.setLevel(logging.INFO)

        tg_logger.setup(logger, token=token, users=users)

    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints, exist_ok=True)
    
    if not os.path.exists(f'{config.paths.path_to_checkpoints}/oof'):
        os.makedirs(f'{config.paths.path_to_checkpoints}/oof', exist_ok=True)

    print('Have a nice training!')
    print(f'Testing {config.general.experiment_name} approach')
    print(f'Folds to train: {config.split.folds_to_train}')

    n_folds = config.split.n_splits
    src_path = config.paths.path_to_checkpoints
    scores = []
    final_score = 0
    start_time = time.time()

    for fold in config.split.folds_to_train:
        print(f'\nFold: {fold}')
        config.paths.path_to_checkpoints = os.path.join(src_path, f'fold_{fold}')
        cur_score, predictions, targets = run(config, fold)

        np.save(os.path.join(f'{src_path}/oof/preds_{fold}.npy'), predictions)
        np.save(os.path.join(f'{src_path}/oof/labels_{fold}.npy'), targets)

        final_score += cur_score / n_folds
        scores.append(cur_score)

        if config.logging.telegram:
            logger.info(f'Fold: {fold} | Score: {round(cur_score, 4)}')

    print()
    for fold in range(len(config.split.folds_to_train)):
        print(f'Fold: {fold} | Score: {round(scores[fold], 4)}')
        if config.logging.telegram:
            logger.info(f'Fold: {fold} | Score: {round(scores[fold], 4)}')

    cv = round(sum(scores) / len(scores), 4)
    t = int(time.time() - start_time)

    print('\nThe training has been finished!')
    print(f'CV: {cv}')
    print(f'Total Time: {t} s')

    with open('results.txt', 'a') as file:
        file.write(f'\n{config.general.experiment_name} | {cv} | {t} s\n')

    if config.logging.telegram:
        logger.info(f'\n{config.general.experiment_name} | {cv} | {t} s\n')