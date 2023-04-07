import json
import torch
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist() 
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
SLIP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
]
SPOSE = (np.array([
    11,13,15,12,14,16,23,24,
])+489).tolist()


def do_hflip_hand(lhand, rhand):
    rhand[...,0] *= -1
    lhand[...,0] *= -1
    rhand, lhand = lhand, rhand
    return lhand, rhand


def do_hflip_eye(leye, reye):
    reye[...,0] *= -1
    leye[...,0] *= -1
    reye, leye = leye, reye
    return leye, reye


def do_hflip_spose(spose):
    spose[...,0] *= -1
    spose = spose[:,[3,4,5,0,1,2,7,6]]
    return spose


def do_hflip_slip(slip):
    slip[...,0] *= -1
    slip = slip[:,[10,9,8,7,6,5,4,3,2,1,0]+[19,18,17,16,15,14,13,12,11]]
    return slip


class ISLDataset(Dataset):
    """The Isolated Sign Language Dataset."""

    def __init__(self, df, config, is_train):
        self.df = df
        self.is_train = is_train
        self.config = config
        self.norm_ref = [500, 501, 512, 513, 159,  386, 13]
        self.fixed_frames = self.config.model.params.max_length
        self.dim = 2
        self.lh_idx_range = (468, 489)
        self.rh_idx_range = (522, 543)

    def __len__(self):
        return len(self.df)
    
    def load_relevant_data_subset(self, path):
        data_columns = ["x", "y"]
        data = pd.read_parquet(path, columns=data_columns)
        n_frames = int(len(data) / 543)
        data = data.values.reshape(n_frames, 543, len(data_columns))
        return data.astype(np.float32)

    def normalise(self, xyz):
        xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdim=True)
        xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdim=True)
        return xyz

    def preprocess(self, xyz):
        lhand = xyz[:,LHAND]
        rhand = xyz[:,RHAND]
        spose = xyz[:,SPOSE]
        leye = xyz[:,LEYE]
        reye = xyz[:,REYE]
        slip = xyz[:,SLIP]

        if self.is_train:
            if random.random() < 0.5:
                lhand, rhand = do_hflip_hand(lhand, rhand)
                spose = do_hflip_spose(spose)
                leye, reye = do_hflip_eye(leye, reye)
                slip = do_hflip_slip(slip)

        xyz = torch.cat([
            lhand,
            rhand,
            spose,
            leye,
            reye,
            slip,
        ],1)

        xyz[torch.isnan(xyz)] = 0
        xyz = xyz[:self.config.model.params.max_length]
        return xyz
    
    def prepare_frames(self, tensor):
        nan_frames = []
        for t in range(tensor.shape[0]):
            if np.all(np.isnan(tensor[t, self.lh_idx_range[0]:self.lh_idx_range[1], :])) and \
            np.all(np.isnan(tensor[t, self.rh_idx_range[0]:self.rh_idx_range[1], :])):
                nan_frames.append(t)

        if len(nan_frames)!=0:
            nan_mask = np.zeros((tensor.shape[0],), dtype=bool)
            nan_mask[nan_frames] = True
            new_tensor = tensor[~nan_mask]
        else:
            new_tensor = tensor

        tensor_frames = new_tensor.shape[0]

        if tensor_frames > self.fixed_frames:
            interval = np.linspace(0, tensor_frames-1, self.fixed_frames, dtype=int)
            new_tensor = np.array([new_tensor[i] for i in interval])
        else: 
            repetition = self.fixed_frames-tensor_frames
            for rep in range(repetition):
                new_tensor = np.concatenate([new_tensor, np.expand_dims(new_tensor[-1], axis=0)], axis=0)

        max_non_zero = self.fixed_frames*21*self.dim
        new_tensor = new_tensor.reshape((self.fixed_frames, self.dim*543))
        tensor = tensor.reshape((tensor.shape[0], self.dim*543))
        filled_tensor = np.where(np.isnan(new_tensor), np.zeros_like(new_tensor), new_tensor)

        right_hand_tensor = filled_tensor[:, self.rh_idx_range[0]*self.dim :self.rh_idx_range[1]*self.dim ]
        left_hand_tensor = filled_tensor[:, self.lh_idx_range[0]*self.dim :self.lh_idx_range[1]*self.dim ]
        count_right_nonzero = np.sum(np.count_nonzero(right_hand_tensor, axis=1))
        count_left_nonzero = np.sum(np.count_nonzero(left_hand_tensor, axis=1))

        if count_right_nonzero not in [max_non_zero, 0] or count_left_nonzero not in [max_non_zero, 0]:
            main_hand_tensor, start, end = (right_hand_tensor, self.rh_idx_range[0]*self.dim , self.rh_idx_range[1]*self.dim ) \
                                    if count_right_nonzero > count_left_nonzero \
                                    else (left_hand_tensor, self.lh_idx_range[0]*self.dim , self.lh_idx_range[1]*self.dim )
            all_indices = list(np.count_nonzero(main_hand_tensor, axis=1))
            zero_indices = [i for i, x in enumerate(all_indices) if x == 0]
            mean_frame = np.mean(np.where(np.isnan(tensor), np.zeros_like(tensor), tensor), axis=0, keepdims=True)[:, start:end]
            mean_tensor = None
            for frame_n in range(self.fixed_frames):
                if frame_n in zero_indices:
                    mean_tensor = np.concatenate([mean_tensor, mean_frame], axis=0) if frame_n != 0 else mean_frame
                else: 
                    main_tensor = np.expand_dims(main_hand_tensor[frame_n, :], axis=0)
                    mean_tensor = np.concatenate([mean_tensor, main_tensor], axis=0) if frame_n != 0 else main_tensor
            zero_tensor = np.zeros((self.fixed_frames,21*self.dim))
            new_right_tensor, new_left_tensor = (mean_tensor, zero_tensor) \
                                                if count_right_nonzero > count_left_nonzero \
                                                else (zero_tensor, mean_tensor)
            filled_tensor = np.concatenate([filled_tensor[:, :468*self.dim], new_left_tensor, filled_tensor[:, 489*self.dim:522*self.dim], new_right_tensor], -1)

        return filled_tensor
            
    def augment(
        self,
        xyz,
        scale  = (0.8,1.5),
        shift  = (-0.1,0.1),
        degree = (-15,15),
        p=0.5
    ):
        
        if random.random() < p:
            if scale is not None:
                scale = np.random.uniform(*scale)
                xyz = scale*xyz
    
        if random.random() < p:
            if shift is not None:
                shift = np.random.uniform(*shift)
                xyz = xyz + shift

        if random.random() < p:
            if degree is not None:
                degree = np.random.uniform(*degree)
                radian = degree / 180 * np.pi
                c = np.cos(radian)
                s = np.sin(radian)
                rotate = np.array([
                    [c,-s],
                    [s, c],
                ]).T
                xyz[...,:2] = xyz[...,:2] @rotate
            
        return xyz

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        pq_file = f'{self.config.paths.path_to_folder}{sample.path}'
        xyz = self.load_relevant_data_subset(pq_file)

        xyz = self.prepare_frames(xyz)
        xyz = torch.from_numpy(xyz).float()
        xyz = xyz.reshape((self.fixed_frames, 543, 2))
        xyz = self.normalise(xyz)
        xyz = self.preprocess(xyz)

        if self.is_train:
            xyz = self.augment(
                xyz=xyz,
                scale=self.config.augmentations.scale,
                shift=self.config.augmentations.shift,
                degree=self.config.augmentations.degree,
                p=self.config.augmentations.p,
            )

        return {
            "features": xyz,
            "labels": sample.label,
        }


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d['labels'] = torch.LongTensor(d['labels'])
    return d
    

def get_data_loader(df, config, is_train):
    """Gets a PyTorch Dataloader."""
    dataset = ISLDataset(df, config, is_train)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=null_collate,
        **config.dataloader_params,
    )
    return data_loader


def get_fold_samples(config, current_fold):
    """Get a train and a val data for a single fold."""

    df = pd.read_csv(config.paths.path_to_csv)
    label_map = json.load(open(config.paths.path_to_json, "r"))
    df['label'] = df['sign'].map(label_map)

    # df = df[df["participant_id"] != 29302]

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
                train_df = df.iloc[train_index]
                val_df = df.iloc[val_index]
                break
    
    # The debug mode
    if config.training.debug:
        train_idx = random.sample([*range(len(train_data))], config.training.number_of_train_debug_samples)
        val_idx = random.sample([*range(len(val_data))], config.training.number_of_val_debug_samples)

        train_data = train_data[train_idx]
        train_targets = train_targets[train_idx]
        val_data = val_data[val_idx]
        val_targets = val_targets[val_idx]

    return train_data, train_targets, train_df, val_data, val_targets, val_df


def get_loaders(config, fold):
    """Get PyTorch Dataloaders."""

    train_data, train_targets, train_df, val_data, val_targets, val_df = get_fold_samples(config, fold)

    train_loader = get_data_loader(
        df=train_df,
        config=config,
        is_train=True,
    )

    val_loader = get_data_loader(
        df=val_df,
        config=config,
        is_train=False,
    )

    return train_loader, val_loader