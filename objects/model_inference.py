import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import tensorflow as tf


class InputNet(tf.keras.layers.Layer):
    def __init__(self):
        super(InputNet, self).__init__()
        self.LHAND = np.array(tf.range(468, 489))
        self.RHAND = np.array(tf.range(522, 543))
        self.REYE = np.array([
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            246, 161, 160, 159, 158, 157, 173,
        ])
        self.LEYE = np.array([
            263, 249, 390, 373, 374, 380, 381, 382, 362,
            466, 388, 387, 386, 385, 384, 398,
        ])
        self.SLIP = np.array([
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            191, 80, 81, 82, 13, 312, 311, 310, 415,
        ])
        self.SPOSE = np.array((
            tf.constant([11, 13, 15, 12, 14, 16, 23, 24])
            + 489
        ))
        self.TRIU = np.array([
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
		])
        self.lhand = (468, 489)
        self.rhand = (522, 543)
        self.max_length = 100

    def call(self, xyz):
        xyz = xyz[:, :, :2]

        L = len(xyz)
        if len(xyz) > self.max_length:
            i = (L-self.max_length)//2
            xyz = xyz[i:i + self.max_length]

        L = len(xyz)
        not_nan_xyz = xyz[~tf.math.is_nan(xyz)]
        xyz -= tf.math.reduce_mean(not_nan_xyz, axis=0, keepdims=True)
        xyz /= tf.math.reduce_std(not_nan_xyz, axis=0, keepdims=True)

        lhand = tf.gather(xyz, self.LHAND, axis=1)
        rhand = tf.gather(xyz, self.RHAND, axis=1)
        spose = tf.gather(xyz, self.SPOSE, axis=1)
        leye = tf.gather(xyz, self.LEYE, axis=1)
        reye = tf.gather(xyz, self.REYE, axis=1)
        slip = tf.gather(xyz, self.SLIP, axis=1)

        lhand2 = xyz[:, self.lhand[0]:self.lhand[1],:2]
        rhand2 = xyz[:, self.rhand[0]:self.rhand[1],:2]

        ld = tf.reshape(lhand2,(-1,21,1,2))-tf.reshape(lhand2,(-1,1,21,2))
        ld = tf.math.sqrt(tf.reduce_sum((ld ** 2),-1))
        ld = tf.reshape(ld,(L, -1))
        ld = tf.gather(ld, self.TRIU, axis=1)

        rd = tf.reshape(rhand2,(-1,21,1,2))-tf.reshape(rhand2,(-1,1,21,2))
        rd = tf.math.sqrt(tf.reduce_sum((rd ** 2),-1))
        rd = tf.reshape(rd,(L, -1))
        rd = tf.gather(rd, self.TRIU, axis=1)

        xyz = tf.concat([
            lhand, rhand, spose, leye, reye, slip,
        ],axis=1)
        
        dxyz = tf.pad(xyz[:-1] - xyz[1:], [[0, 1], [0, 0], [0, 0]], mode="CONSTANT")
        xyz = tf.concat([
            tf.reshape(xyz,(L,-1)),
            tf.reshape(dxyz,(L,-1)),
            tf.reshape(rd,(L,-1)),
            tf.reshape(ld,(L,-1)),
        ], -1)

        xyz = tf.where(tf.math.is_nan(xyz), tf.zeros_like(xyz), xyz)
        return xyz
    

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
			nn.Hardswish(inplace=True),
			nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.mlp(x)


def positional_encoding(length, embed_dim):
    with torch.no_grad():
        dim = embed_dim // 2

        position = torch.arange(length).unsqueeze(1)
        dim = torch.arange(dim).unsqueeze(0) / dim

        angle = position * (1 / (10000**dim))

        pos_embed = torch.cat(
            [torch.sin(angle), torch.cos(angle)],
            dim=-1
        )
        return pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            batch_first,
        ):
        self.embed_dim= embed_dim
        self.num_head = num_head

        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )
    
    def forward(self, x):
        q = F.linear(x[:1], self.mha.in_proj_weight[:self.embed_dim], self.mha.in_proj_bias[:self.embed_dim])
        k = F.linear(x, self.mha.in_proj_weight[self.embed_dim:self.embed_dim * 2], 
                        self.mha.in_proj_bias[self.embed_dim:self.embed_dim * 2])
        v = F.linear(x, self.mha.in_proj_weight[self.embed_dim * 2:], self.mha.in_proj_bias[self.embed_dim * 2:]) 
        q = q.reshape(-1, self.num_head, self.embed_dim // self.num_head).permute(1, 0, 2)
        k = k.reshape(-1, self.num_head, self.embed_dim // self.num_head).permute(1, 2, 0)
        v = v.reshape(-1, self.num_head, self.embed_dim // self.num_head).permute(1, 0, 2)
        dot  = torch.matmul(q, k) * (1/(self.embed_dim // self.num_head)**0.5) 
        attn = F.softmax(dot, -1)
        out  = torch.matmul(attn, v)
        out  = out.permute(1, 0, 2).reshape(-1, self.embed_dim)
        out  = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)  
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
        embed_dim,
        num_head,
        out_dim,
        batch_first=True,
    ):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_head,batch_first)
        self.ffn   = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x): 
        x = x[:1] + self.attn((self.norm1(x)))
        x = x + self.ffn((self.norm2(x)))
        return x
    

class SingleNet(nn.Module):

    def __init__(self, fold, max_length, embed_dim, num_point, num_head, num_class, num_block):
        super().__init__()
        embed_dim = 256 if fold == 5 else embed_dim
        self.num_block = num_block
        self.embed_dim = embed_dim
        self.num_head  = num_head
        self.max_length = max_length
        self.num_point = num_point

        self.pos_embed = nn.Parameter(positional_encoding(max_length, self.embed_dim))

        self.cls_embed = nn.Parameter(torch.zeros((1, self.embed_dim), device='cuda'))
        
        self.x_embed = nn.Sequential(
            nn.Linear(num_point, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.Hardswish(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Hardswish(),
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                self.embed_dim,
                self.num_head,
                self.embed_dim,
                batch_first=False
            ) for _ in range(self.num_block)
        ])
        self.logit = ArcMarginProduct_subcenter(self.embed_dim, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xyz):
        with torch.no_grad():
            L = xyz.shape[0]
            x_embed = self.x_embed(xyz.flatten(1)) 
            x = torch.cat([
                self.cls_embed,
                x_embed[:L] + self.pos_embed[:L]
            ],0)
            x = self.encoder[0](x)
            cls = x[[0]]
            logit = self.logit(cls)
            pred = self.softmax(logit)
            return pred