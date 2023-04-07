import os

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import tensorflow as tf
import multiprocessing as mp

from config import config

ROWS_PER_FRAME = 543
label_map = json.load(open(config.paths.path_to_json, "r"))

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]


class InputNet(tf.keras.layers.Layer):
    def __init__(self, lh_idx_range=(468, 489), rh_idx_range=(522, 543)):
        super(InputNet, self).__init__()
        self.fixed_frames = 4
        self.dim = 2
        self.lh_idx_range = lh_idx_range
        self.rh_idx_range = rh_idx_range
        
    def call(self, tensor):
        nan_frames = []
        for t in range(tensor.shape[0]):
            if tf.math.reduce_all(tf.math.is_nan(tensor[t, self.lh_idx_range[0]:self.lh_idx_range[1], :])) and \
            tf.math.reduce_all(tf.math.is_nan(tensor[t, self.rh_idx_range[0]:self.rh_idx_range[1], :])):
                nan_frames.append(t)
                
        if len(nan_frames)!=0:
            nan_mask = tf.scatter_nd(indices=[[i] for i in nan_frames], updates=[True]*len(nan_frames), shape=(tensor.shape[0],))
            new_tensor = tf.boolean_mask(tensor, tf.logical_not(nan_mask), axis=0)
        else:
            new_tensor = tensor
        tensor_frames = new_tensor.shape[0]
        
        if tensor_frames > self.fixed_frames:
            interval = np.linspace(0, tensor_frames-1, self.fixed_frames, dtype=int)
            new_tensor = tf.concat([tf.expand_dims(new_tensor[i], axis=0) for i in interval], axis=0)
        else: 
            repetition = self.fixed_frames-tensor_frames
            for rep in range(repetition):
                new_tensor = tf.concat([new_tensor, tf.expand_dims(new_tensor[-1], axis=0)], axis=0)
        
        max_non_zero = self.fixed_frames*21*self.dim
        new_tensor = tf.reshape(new_tensor, (self.fixed_frames, self.dim*543))
        tensor = tf.reshape(tensor, (tensor.shape[0], self.dim*543))
        filled_tensor = tf.where(tf.math.is_nan(new_tensor), tf.zeros_like(new_tensor), new_tensor)
        
        right_hand_tensor = filled_tensor[:, self.rh_idx_range[0]*self.dim :self.rh_idx_range[1]*self.dim ]
        left_hand_tensor = filled_tensor[:, self.lh_idx_range[0]*self.dim :self.lh_idx_range[1]*self.dim ]
        count_right_nonzero = tf.reduce_sum(tf.math.count_nonzero(right_hand_tensor, axis=1))
        count_left_nonzero = tf.reduce_sum(tf.math.count_nonzero(left_hand_tensor, axis=1))
        
        if count_right_nonzero not in [max_non_zero, 0] or count_left_nonzero not in [max_non_zero, 0]:
            main_hand_tensor, start, end = (right_hand_tensor, self.rh_idx_range[0]*self.dim , self.rh_idx_range[1]*self.dim ) \
                                    if count_right_nonzero > count_left_nonzero \
                                    else (left_hand_tensor, self.lh_idx_range[0]*self.dim , self.lh_idx_range[1]*self.dim )
            all_indices = list(tf.math.count_nonzero(main_hand_tensor, axis=1))
            zero_indices = [i for i, x in enumerate(all_indices) if x == 0]
            mean_frame = tf.reduce_mean(tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor), axis=0, keepdims=True)[:, start:end]
            for frame_n in range(self.fixed_frames):
                if frame_n in zero_indices:
                    mean_tensor = tf.concat([mean_tensor, mean_frame], axis=0) if frame_n != 0 else mean_frame
                else: 
                    main_tensor = tf.expand_dims(main_hand_tensor[frame_n, :], axis=0)
                    mean_tensor = tf.concat([mean_tensor, main_tensor], axis=0) if frame_n != 0 else main_tensor
            zero_tensor = tf.zeros((self.fixed_frames,21*self.dim))
            new_right_tensor, new_left_tensor = (mean_tensor, zero_tensor) \
                                                if count_right_nonzero > count_left_nonzero \
                                                else (zero_tensor, mean_tensor)
            filled_tensor = tf.concat([filled_tensor[:, :468*self.dim], new_left_tensor, filled_tensor[:, 489*self.dim:522*self.dim], new_right_tensor], -1)
        
        return filled_tensor
    
    
feature_converter = InputNet()


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def convert_row(row):
    x = load_relevant_data_subset(os.path.join(config.paths.path_to_folder, row[1].path))
    x = feature_converter(x)
    return x, row[1].label


def convert_and_save_data():
    df = pd.read_csv(config.paths.path_to_csv)
    # df = df[df["participant_id"] != 29302]
    df['label'] = df['sign'].map(label_map)
    npdata = []
    nplabels = []
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata.append(x)
            nplabels.append(y)

    del df

    with open('./feature_data/feature_labels_drop.pickle', 'wb') as file:
        pickle.dump(nplabels, file)

    del nplabels

    with open('./feature_data/feature_data_drop.pickle', 'wb') as file:
        pickle.dump(npdata, file)


if __name__ == "__main__":   
    # convert_and_save_data()
    x = load_relevant_data_subset(os.path.join(config.paths.path_to_folder, '/home/toefl/K/asl-signs/train_landmark_files/53618/1001379621.parquet'))
    x = feature_converter(x)
    print(x.shape)