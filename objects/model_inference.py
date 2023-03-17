#pytorch model

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
    

#num_landmark = 543
max_length = 80
num_class  = 250
num_point  = 82  # LIP, LHAND, RHAND

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
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.mlp(x)


#https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            batch_first,
        ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )

    def forward(self, x, x_mask):
        out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
        return out


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

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x

class Net(nn.Module):

    def __init__(self, num_class=num_class):
        super().__init__()
        self.output_type = ['inference', 'loss']

        num_block = 1
        embed_dim = 1024
        num_head  = 8

        pos_embed = positional_encoding(max_length, embed_dim)
        # self.register_buffer('pos_embed', pos_embed)
        self.pos_embed = nn.Parameter(pos_embed)

        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))
        self.x_embed = nn.Sequential(
            nn.Linear(num_point * 3, embed_dim, bias=False),
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_head,
                embed_dim,
            ) for i in range(num_block)
        ])
        self.logit = nn.Linear(embed_dim, num_class)

    def forward(self, batch):
        length = [len(x) for x in batch['xyz']]
        xyz = batch['xyz']

        x, x_mask = pack_seq(xyz)
        B,L,_ = x.shape
        x = self.x_embed(x)
        x = x + self.pos_embed[:L].unsqueeze(0)

        x = torch.cat([
            self.cls_embed.unsqueeze(0).repeat(B,1,1),
            x
        ],1)
        x_mask = torch.cat([
            torch.zeros(B,1).to(x_mask),
            x_mask
        ],1)


        #x = F.dropout(x,p=0.25,training=self.training)
        for block in self.encoder:
            x = block(x,x_mask)

        cls = x[:,0]
        cls = F.dropout(cls,p=0.4,training=self.training)
        logit = self.logit(cls)

        output = {}
        if 'loss' in self.output_type:
            output['label_loss'] = F.cross_entropy(logit, batch['label'])

        if 'inference' in self.output_type:
            output['sign'] = torch.softmax(logit,-1)

        return output


def pre_process(xyz):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common mean
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    
    LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]

    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat([ #(none, 82, 3)
        lip,
        lhand,
        rhand,
    ],1)
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_length]
    return xyz

#pytorch model for tflite conversion

#simplfiy for one video input 
max_length = 96  #reduce this if gets out of memory error

class InputNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.max_length = 60 
  
    def forward(self, xyz):
        xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdim=True) #noramlisation to common maen
        xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdim=True)

        LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        #LHAND = np.arange(468, 489).tolist()
        #RHAND = np.arange(522, 543).tolist()

        lip = xyz[:, LIP]
        lhand = xyz[:, 468:489]
        rhand = xyz[:, 522:543]
        xyz = torch.cat([  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ], 1)
        xyz[torch.isnan(xyz)] = 0
        x = xyz[:self.max_length]
        return x


#overwrite the model used in training ....

# use fix dimension
class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            batch_first,
        ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
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
        # out,_ = self.mha(x,x,x,need_weights=False)
        # out,_ = F.multi_head_attention_forward(
        #     x, x, x,
        #     self.mha.embed_dim,
        #     self.mha.num_heads,
        #     self.mha.in_proj_weight,
        #     self.mha.in_proj_bias,
        #     self.mha.bias_k,
        #     self.mha.bias_v,
        #     self.mha.add_zero_attn,
        #     0,#self.mha.dropout,
        #     self.mha.out_proj.weight,
        #     self.mha.out_proj.bias,
        #     training=False,
        #     key_padding_mask=None,
        #     need_weights=False,
        #     attn_mask=None,
        #     average_attn_weights=False
        # )
 
        #qkv = F.linear(x, self.mha.in_proj_weight, self.mha.in_proj_bias)
        #qkv = qkv.reshape(-1,3,1024)
        #q,k,v = qkv[[0],0], qkv[:,1],  qkv[:,2]

        q = F.linear(x[:1], self.mha.in_proj_weight[:1024], self.mha.in_proj_bias[:1024]) #since we need only cls
        k = F.linear(x, self.mha.in_proj_weight[1024:2048], self.mha.in_proj_bias[1024:2048])
        v = F.linear(x, self.mha.in_proj_weight[2048:], self.mha.in_proj_bias[2048:]) 
        q = q.reshape(-1, 8, 128).permute(1, 0, 2)
        k = k.reshape(-1, 8, 128).permute(1, 2, 0)
        v = v.reshape(-1, 8, 128).permute(1, 0, 2)
        dot  = torch.matmul(q, k) * (1/128**0.5) # H L L
        attn = F.softmax(dot, -1)  #   L L
        out  = torch.matmul(attn, v)  #   L H dim
        out  = out.permute(1, 0, 2).reshape(-1, 1024)
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
            nn.Linear(num_point * 2, self.embed_dim, bias=False),
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                self.embed_dim,
                self.num_head,
                self.embed_dim,
                batch_first=False
            ) for i in range(self.num_block)
        ])
        self.logit = ArcMarginProduct(self.embed_dim, num_class)

    def forward(self, xyz):
        L = xyz.shape[0]
        x_embed = self.x_embed(xyz.flatten(1)) 
        x = x_embed[:L] + self.pos_embed[:L]
        x = torch.cat([
            self.cls_embed,
            x
        ],0)
        #x = x.unsqueeze(1)

        #for block in self.encoder: x = block(x) #remove tflite loop
        x = self.encoder[0](x)
        cls = x[[0]]
        logit = self.logit(cls)
        return logit
    

