import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import tensorflow as tf


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
    

class HardSwish(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        return x * F.relu6(x+3) * 0.16666667
    

def pack_seq(
    seq,
):
    length = [len(s) for s in seq]
    batch_size = len(seq)
    num_landmark=seq[0].shape[1]

    x = torch.zeros((batch_size, max(length), num_landmark, 3)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, max(length))).to(seq[0].device)
    for b in range(batch_size):
        L = length[b]
        x[b, :L] = seq[b][:L]
        x_mask[b, L:] = 1
    x_mask = (x_mask>0.5)
    x = x.reshape(batch_size,-1,num_landmark*3)
    return x, x_mask


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim , hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.mlp(x)


def positional_encoding(length, embed_dim):
    dim = embed_dim//2

    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)

    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)

    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


class InputNet(tf.keras.layers.Layer):
    def init(self):
        super(InputNet, self).__init__()

    def call(self, xyz):
        xyz = xyz[:, :, :2]
        xyz = xyz[:45]
        xyz = xyz - tf.math.reduce_mean(tf.boolean_mask(xyz, ~tf.math.is_nan(xyz)), axis=0, keepdims=True) #noramlisation to common maen
        xyz = xyz / tf.math.reduce_std(tf.boolean_mask(xyz, ~tf.math.is_nan(xyz)), axis=0, keepdims=True)

        LIP = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])

        lip = tf.gather(xyz, LIP, axis=1)
        lhand = xyz[:, 468:489]
        rhand = xyz[:, 522:543]
        xyz = tf.concat([  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ], 1)

        L = len(xyz)
        dxyz = tf.pad(xyz[:-1] - xyz[1:],  [[0, 1], [0, 0], [0, 0]])

        lhand = xyz[:, :21, :2]
        ld = lhand[:, :, tf.newaxis, :] - lhand[:, tf.newaxis, :, :]
        ld = tf.sqrt(tf.reduce_sum(ld ** 2, axis=-1))
        rhand = xyz[:, 21:42, :2]
        rd = rhand[:, :, tf.newaxis, :] - rhand[:, tf.newaxis, :, :]
        rd = tf.sqrt(tf.reduce_sum(rd ** 2, axis=-1))

        x = tf.concat([
            tf.reshape(xyz, (L, -1)),
            tf.reshape(dxyz, (L, -1)),
            tf.reshape(rd, (L, -1)),
            tf.reshape(ld, (L, -1)),
        ], -1)
        
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        return x

#overwrite the model used in training ....

# use fix dimension
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
    #https://github.com/pytorch/text/blob/60907bf3394a97eb45056a237ca0d647a6e03216/torchtext/modules/multiheadattention.py#L5
    def forward(self, x):
        q = F.linear(x[:1], self.mha.in_proj_weight[:self.embed_dim], self.mha.in_proj_bias[:self.embed_dim]) #since we need only cls
        k = F.linear(x, self.mha.in_proj_weight[self.embed_dim:self.embed_dim * 2], 
                        self.mha.in_proj_bias[self.embed_dim:self.embed_dim * 2])
        v = F.linear(x, self.mha.in_proj_weight[self.embed_dim * 2:], self.mha.in_proj_bias[self.embed_dim * 2:]) 
        q = q.reshape(-1, self.num_head, self.embed_dim / self.num_head).permute(1, 0, 2)
        k = k.reshape(-1, self.num_head, self.embed_dim / self.num_head).permute(1, 2, 0)
        v = v.reshape(-1, self.num_head, self.embed_dim / self.num_head).permute(1, 0, 2)
        dot  = torch.matmul(q, k) * (1/(self.embed_dim / self.num_head)**0.5) # H L L
        attn = F.softmax(dot, -1)  #   L L
        out  = torch.matmul(attn, v)  #   L H dim
        out  = out.permute(1, 0, 2).reshape(-1, self.embed_dim)
        out  = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)  
        return out

# remove mask
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

    def __init__(self, max_length, embed_dim, num_point, num_head, num_class, num_block):
        super().__init__()
        self.num_block = num_block
        self.embed_dim = embed_dim
        self.num_head  = num_head
        self.max_length = max_length
        self.num_point = num_point

        pos_embed = positional_encoding(max_length, self.embed_dim)
        self.pos_embed = nn.Parameter(pos_embed)

        self.cls_embed = nn.Parameter(torch.zeros((1, self.embed_dim)))
        self.x_embed = nn.Sequential(
            nn.Linear(num_point * 2, embed_dim * 3),
            nn.LayerNorm(embed_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                self.embed_dim,
                self.num_head,
                self.embed_dim,
                batch_first=False
            ) for i in range(self.num_block)
        ])
        self.logit = ArcMarginProduct_subcenter(self.embed_dim, num_class)

    def forward(self, xyz):
        with torch.no_grad():
            L = xyz.shape[0]
            x_embed = self.x_embed(xyz.flatten(1)) 
            x = x_embed[:L] + self.pos_embed[:L]
            x = torch.cat([
                self.cls_embed,
                x
            ],0)

            x = self.encoder[0](x)
            cls = x[[0]]
            logit = self.logit(cls)
            return logit