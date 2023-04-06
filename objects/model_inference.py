import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import tensorflow as tf


class InputNet(tf.keras.Model):
    def __init__(self):
        super(InputNet, self).__init__()
        self.NORM_REF = np.array([500, 501, 512, 513, 159, 386, 13])
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

    def call(self, xyz):
        xyz = xyz[:60]
        xyz = xyz[:, :, :2]
        
        xyz = xyz - tf.math.reduce_mean(tf.boolean_mask(xyz, ~tf.math.is_nan(xyz)), axis=0, keepdims=True)
        xyz = xyz / tf.math.reduce_std(tf.boolean_mask(xyz, ~tf.math.is_nan(xyz)), axis=0, keepdims=True)
        # K = xyz.shape[-1]
        # ref = tf.gather(xyz, self.NORM_REF, axis=1)
        # xyz_flat = tf.reshape(ref, (-1, K))
        # m = tf.math.reduce_mean(xyz_flat, axis=0, keepdims=True)
        # s = tf.math.reduce_mean(tf.math.reduce_std(xyz_flat, axis=0))
        # xyz = xyz - m
        # xyz = xyz / s

        lhand = tf.gather(xyz, self.LHAND, axis=1)
        rhand = tf.gather(xyz, self.RHAND, axis=1)
        spose = tf.gather(xyz, self.SPOSE, axis=1)
        leye = tf.gather(xyz, self.LEYE, axis=1)
        reye = tf.gather(xyz, self.REYE, axis=1)
        slip = tf.gather(xyz, self.SLIP, axis=1)

        xyz = tf.concat(
            [lhand, rhand, spose, leye, reye, slip],
            axis=1,
        )

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

    def __init__(self, max_length, embed_dim, num_point, num_head, num_class, num_block):
        super().__init__()
        self.num_block = num_block
        self.embed_dim = embed_dim
        self.num_head  = num_head
        self.max_length = max_length
        self.num_point = num_point

        self.pos_embed = nn.Parameter(positional_encoding(max_length, self.embed_dim))

        self.cls_embed = nn.Parameter(torch.zeros((1, self.embed_dim), device='cuda'))
        self.x_embed = nn.Sequential(
            nn.Linear(num_point * 2, embed_dim * 3),
            nn.LayerNorm(embed_dim * 3),
            nn.Hardswish(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.Hardswish(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 2, embed_dim),
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

    def forward(self, xyz):
        with torch.no_grad():
            L = xyz.shape[0]
            # xyz = xyz.reshape(xyz.shape[0], xyz.shape[1] // 2, 2)
            x_embed = self.x_embed(xyz.flatten(1)) 
            x = torch.cat([
                self.cls_embed,
                x_embed[:L] + self.pos_embed[:L]
            ],0)
            x = self.encoder[0](x)
            cls = x[[0]]
            logit = self.logit(cls)
            return logit