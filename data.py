import torch
import random
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold


class ISLDataset(Dataset):
    """The Isolated Sign Language Dataset."""
    
    def __init__(self, config, data, labels):
        self.data = data
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "features": torch.from_numpy(self.data[i]).float(),
            "labels": self.labels[i],
        }
    

def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d['labels'] = torch.LongTensor(d['labels'])
    return d
    

def get_data_loader(config, data, labels):
    """Gets a PyTorch Dataloader."""
    dataset = ISLDataset(config, data, labels)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=null_collate,
        **config.dataloader_params,
    )
    return data_loader


def get_fold_samples(config, current_fold):
    """Get a train and a val data for a single fold."""

    df = pd.read_csv(config.paths.path_to_csv)

    if config.paths.path_to_data.endswith('npy'):
        data = np.load(config.paths.path_to_data)
        labels =  np.load(config.paths.path_to_labels)
    else:
        with open(config.paths.path_to_data, 'rb') as file: data = pickle.load(file)
        with open(config.paths.path_to_labels, 'rb') as file: labels = pickle.load(file)
        
    # The dataframe is split in advantage
    if config.split.already_split:
        train_index = df.index[df["fold"] != current_fold]
        val_index = df.index[df["fold"] == current_fold]

        if type(data) is np.ndarray:
            train_data = data[train_index]
            train_targets = labels[train_index]
            val_data = data[val_index]
            val_targets = labels[val_index]
        else:
            train_data = [data[i] for i in train_index]
            train_targets = [labels[i] for i in train_index]
            val_data = [data[i] for i in val_index]
            val_targets = [labels[i] for i in val_index]

    # The dataframe isn't split in advantage
    else:
        groups = df["path"].map(lambda x: x.split("/")[1])
        kfold = StratifiedGroupKFold(n_splits=config.split.n_splits, shuffle=True, random_state=config.general.seed)
        
        for fold, (train_index, val_index) in enumerate(kfold.split(data, labels, groups)):
            if fold == current_fold:
                if type(data) is np.ndarray:
                    train_data = data[train_index]
                    train_targets = labels[train_index]
                    val_data = data[val_index]
                    val_targets = labels[val_index]
                else:
                    train_data = [data[i] for i in train_index]
                    train_targets = [labels[i] for i in train_index]
                    val_data = [data[i] for i in val_index]
                    val_targets = [labels[i] for i in val_index]
                break
    
    # The debug mode
    if config.training.debug:
        train_idx = random.sample([*range(len(train_data))], config.training.number_of_train_debug_samples)
        val_idx = random.sample([*range(len(val_data))], config.training.number_of_val_debug_samples)

        train_data = train_data[train_idx]
        train_targets = train_targets[train_idx]
        val_data = val_data[val_idx]
        val_targets = val_targets[val_idx]

    return train_data, train_targets, val_data, val_targets


def get_loaders(config, fold):
    """Get PyTorch Dataloaders."""

    train_data, train_targets, val_data, val_targets = get_fold_samples(config, fold)

    train_loader = get_data_loader(
        config=config,
        data=train_data,
        labels=train_targets,
    )

    val_loader = get_data_loader(
        config=config,
        data=val_data,
        labels=val_targets,
    )

    return train_loader, val_loader