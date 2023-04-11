import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold


class ISLDataset(Dataset):
    """The Isolated Sign Language Dataset."""

    def __init__(self, df, config, is_train):
        self.df = df
        self.is_train = is_train
        self.config = config

        offset = (np.arange(1000) - self.config.model.params.max_length) // 2
        offset = np.clip(offset,0, 1000).tolist()
        self.offset = nn.Parameter(torch.LongTensor(offset),requires_grad=False)

        self.LHAND = np.arange(468, 489).tolist()
        self.RHAND = np.arange(522, 543).tolist() 
        self.REYE = [
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            246, 161, 160, 159, 158, 157, 173,
        ]
        self.LEYE = [
            263, 249, 390, 373, 374, 380, 381, 382, 362,
            466, 388, 387, 386, 385, 384, 398,
        ]
        self.SLIP = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            191, 80, 81, 82, 13, 312, 311, 310, 415,
        ]
        self.SPOSE = (np.array([11,13,15,12,14,16,23,24,])+489).tolist()
        self.TRIU = [
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
			14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28,
			29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
			45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
			58, 59, 60, 61, 62, 67, 68, 69, 70, 71, 72, 73, 74,
			75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92,
			93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 111,
			112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
			125, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
			145, 146, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
			166, 167, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
			188, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 221,
			222, 223, 224, 225, 226, 227, 228, 229, 230, 243, 244, 245, 246,
			247, 248, 249, 250, 251, 265, 266, 267, 268, 269, 270, 271, 272,
			287, 288, 289, 290, 291, 292, 293, 309, 310, 311, 312, 313, 314,
			331, 332, 333, 334, 335, 353, 354, 355, 356, 375, 376, 377, 397,
			398, 419,
		]

    def __len__(self):
        return len(self.df)
    
    def load_relevant_data_subset(self, path):
        data_columns = ["x", "y"]
        data = pd.read_parquet(path, columns=data_columns)
        n_frames = int(len(data) / 543)
        data = data.values.reshape(n_frames, 543, len(data_columns))
        return data.astype(np.float32)

    def normalise(self, xyz):
        xyz = xyz[:, :, :2]

        L = len(xyz)
        if L > self.config.model.params.max_length:
            i = self.offset[L]
            xyz = xyz[i:i+self.config.model.params.max_length]

        L = len(xyz)
        not_nan_xyz = xyz[~torch.isnan(xyz)]
        if len(not_nan_xyz) != 0:
            not_nan_xyz_mean = not_nan_xyz.mean(0, keepdim=True)
            not_nan_xyz_std  = not_nan_xyz.std(0, keepdim=True)
            xyz -= not_nan_xyz_mean
            xyz /= not_nan_xyz_std

        return xyz, L
    
    def do_hflip_hand(self, lhand, rhand):
        rhand[...,0] *= -1
        lhand[...,0] *= -1
        rhand, lhand = lhand, rhand
        return lhand, rhand

    def do_hflip_eye(self, leye, reye):
        reye[...,0] *= -1
        leye[...,0] *= -1
        reye, leye = leye, reye
        return leye, reye

    def do_hflip_spose(self, spose):
        spose[...,0] *= -1
        spose = spose[:,[3,4,5,0,1,2,7,6]]
        return spose

    def do_hflip_slip(self, slip):
        slip[...,0] *= -1
        slip = slip[:,[10,9,8,7,6,5,4,3,2,1,0]+[19,18,17,16,15,14,13,12,11]]
        return slip

    def preprocess(self, xyz, L):
        if self.is_train:
            xyz = self.augment(
                xyz=xyz,
                scale=self.config.augmentations.scale,
                shift=self.config.augmentations.shift,
                degree=self.config.augmentations.degree,
                p=self.config.augmentations.p,
            )

        lhand = xyz[:,self.LHAND]
        rhand = xyz[:,self.RHAND]
        spose = xyz[:,self.SPOSE]
        leye = xyz[:,self.LEYE]
        reye = xyz[:,self.REYE]
        slip = xyz[:,self.SLIP]

        if self.is_train:
            if random.random() < 0.5:
                lhand, rhand = self.do_hflip_hand(lhand, rhand)
                spose = self.do_hflip_spose(spose)
                leye, reye = self.do_hflip_eye(leye, reye)
                slip = self.do_hflip_slip(slip)

        lhand2 = lhand[:, :21, :2]
        ld = lhand2.reshape(-1, 21, 1, 2) - lhand2.reshape(-1, 1, 21, 2)
        ld = np.sqrt((ld ** 2).sum(-1))
        ld = ld.reshape(L, -1)
        ld = ld[:,self.TRIU]

        rhand2 = rhand[:, :21, :2]
        rd = rhand2.reshape(-1, 21, 1, 2) - rhand2.reshape(-1, 1, 21, 2)
        rd = np.sqrt((rd ** 2).sum(-1))
        rd = rd.reshape(L, -1)
        rd = rd[:,self.TRIU]

        led = leye.reshape(-1, 16, 1, 2) - leye.reshape(-1, 1, 16, 2)
        led = np.sqrt((led ** 2).sum(-1))
        led = led.reshape(L, -1)

        red = reye.reshape(-1, 16, 1, 2) - reye.reshape(-1, 1, 16, 2)
        red = np.sqrt((red ** 2).sum(-1))
        red = led.reshape(L, -1)

        lid = slip.reshape(-1, 20, 1, 2) - slip.reshape(-1, 1, 20, 2)
        lid = np.sqrt((lid ** 2).sum(-1))
        lid = lid.reshape(L, -1)

        pod = spose.reshape(-1, 8, 1, 2) - spose.reshape(-1, 1, 8, 2)
        pod = np.sqrt((pod ** 2).sum(-1))
        pod = pod.reshape(L, -1)

        xyz = torch.cat([
            lhand,
            rhand,
            spose,
            leye,
            reye,
            slip,
        ], 1).contiguous()
        dxyz = F.pad(xyz[:-1] - xyz[1:], [0, 0, 0, 0, 0, 1])

        xyz = torch.cat([
            xyz.reshape(L,-1),
            dxyz.reshape(L,-1),
            rd.reshape(L,-1),
            ld.reshape(L,-1),
            led.reshape(L,-1),
            red.reshape(L,-1),
            lid.reshape(L,-1),
            pod.reshape(L,-1),
        ], -1)

        xyz[torch.isnan(xyz)] = 0
        return xyz
            
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

        xyz = torch.from_numpy(xyz).float()
        xyz, L = self.normalise(xyz)
        xyz = self.preprocess(xyz, L)

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
                    if config.split.all_data_train:
                        train_data = data
                        train_targets = labels
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

        train_data = [data[i] for i in train_idx]
        train_targets = [labels[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        val_targets = [labels[i] for i in val_idx]

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