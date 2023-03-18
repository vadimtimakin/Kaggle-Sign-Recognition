import os

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import multiprocessing as mp

from config import config

ROWS_PER_FRAME = 543
label_map = json.load(open(config.paths.path_to_json, "r"))

# https://www.kaggle.com/competitions/asl-signs/discussion/391812#2168354
lipsUpperOuter =  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
lips = lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

class InputNet(nn.Module):
    def __init__(self, ):
        super().__init__()
  
    def forward(self, xyz):
        xyz = xyz[:,:,:2]
        xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdim=True)
        xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdim=True)

        lip = xyz[:, LIP]
        lhand = xyz[:, 468:489]
        rhand = xyz[:, 522:543]
        xyz = torch.cat([
            lip,
            lhand,
            rhand,
        ], 1)
        L = len(xyz)
        dxyz = np.pad(xyz[:-1] - xyz[1:],  [[0, 1], [0, 0], [0, 0]])

        lhand = xyz[:, :21, :2]
        ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
        ld = np.sqrt((ld ** 2).sum(-1))
        rhand = xyz[:, 21:42, :2]
        rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
        rd = np.sqrt((rd ** 2).sum(-1))

        x = np.concatenate([
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            rd.reshape(L, -1),
            ld.reshape(L, -1),
        ], -1)

        x[np.isnan(x)] = 0
        print(x.shape)
        return x


class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        x = x[:,:,:2]
        lips_x = x[:,lips,:].contiguous().view(-1, 43*2)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*2)
        pose_x = x[:,489:522,:].contiguous().view(-1, 33*2)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*2)
        
        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        
        x1m = torch.mean(lips_x, 0)
        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)
        
        x1s = torch.std(lips_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)
        
        xfeat = torch.cat([x1m,x2m,x3m,x4m, x1s,x2s,x3s,x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)
        
        return xfeat
    
feature_converter = InputNet()

def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def convert_row(row):
    x = load_relevant_data_subset(os.path.join(config.paths.path_to_folder, row[1].path))
    x = feature_converter(torch.tensor(x))
    return x, row[1].label


def convert_and_save_data():
    df = pd.read_csv(config.paths.path_to_csv)
    df['label'] = df['sign'].map(label_map)
    npdata = []
    nplabels = []
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata.append(x)
            nplabels.append(y)

    with open('./feature_data/feature_data.pickle', 'wb') as file:
        pickle.dump(npdata, file)

    with open('./feature_data/feature_labels.pickle', 'wb') as file:
        pickle.dump(nplabels, file)


if __name__ == "__main__":  
    convert_and_save_data()