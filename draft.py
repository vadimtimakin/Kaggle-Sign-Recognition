import numpy as np
import torch
import torch.nn as nn


class ISLDataset(Dataset):
    """The Isolated Sign Language Dataset."""

    def preprocess(self, xyz):
        lhand = xyz[:,LHAND]
        rhand = xyz[:,RHAND]
        spose = xyz[:,SPOSE]
        leye = xyz[:,LEYE]
        reye = xyz[:,REYE]
        slip = xyz[:,SLIP]

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
            

    def __getitem__(self, xyz):
        NORM_REF = [500, 501, 512, 513, 159,  386, 13]
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

        K = xyz.shape[-1]
        ref = xyz[:, NORM_REF]
        xyz_flat = ref.reshape(-1,K)
        m = np.nanmean(xyz_flat,0).reshape(1,1,K)
        s = np.nanstd(xyz_flat, 0).mean() 
        xyz = xyz - m
        xyz = xyz / s

        lhand = xyz[:,LHAND]
        rhand = xyz[:,RHAND]
        spose = xyz[:,SPOSE]
        leye = xyz[:,LEYE]
        reye = xyz[:,REYE]
        slip = xyz[:,SLIP]

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


class InputNet(nn.Module):
    def __init__(self, ):
        super().__init__()
  
    def forward(self, xyz):
        NORM_REF = [500, 501, 512, 513, 159,  386, 13]
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
        
        K = xyz.shape[-1]
        ref = xyz[:, NORM_REF]
        xyz_flat = ref.reshape(-1,K)
        m = np.nanmean(xyz_flat,0).reshape(1,1,K)
        s = np.nanstd(xyz_flat, 0).mean() 
        xyz = xyz - m
        xyz = xyz / s

        lhand = xyz[:,LHAND]
        rhand = xyz[:,RHAND]
        spose = xyz[:,SPOSE]
        leye = xyz[:,LEYE]
        reye = xyz[:,REYE]
        slip = xyz[:,SLIP]

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