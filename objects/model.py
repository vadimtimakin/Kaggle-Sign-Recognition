import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


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


def positional_encoding(length, embed_dim):
    dim = embed_dim//2
    position = np.arange(length)[:, np.newaxis]
    dim = np.arange(dim)[np.newaxis, :]/dim
    angle = 1 / (10000**dim)
    angle = position * angle
    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


def pack_seq(
    seq, max_length,
):
    # seq = [s.reshape(s.shape[0], s.shape[1] // 2, 2) for s in seq]
    length = [min(len(s), max_length)  for s in seq]
    batch_size = len(seq)
    K = seq[0].shape[1]
    L = max(length)

    x = torch.zeros((batch_size, L, K, 2)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, L)).to(seq[0].device)
    for b in range(batch_size):
        l = length[b]
        x[b, :l] = seq[b][:l]
        x_mask[b, l:] = 1
    x_mask = (x_mask>0.5)
    x = x.reshape(batch_size,-1,K*2)
    return x, x_mask


class BasedPartyNet(nn.Module):

    def __init__(self, max_length, embed_dim, num_point, num_head, num_class, num_block):
        super().__init__()
        self.output_type = ['inference', 'loss']

        pos_embed = positional_encoding(max_length, embed_dim)
        self.max_length = max_length

        self.pos_embed = nn.Parameter(pos_embed)
        self.embed_dim = embed_dim

        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))
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
                embed_dim,
                num_head,
                embed_dim,
            ) for i in range(num_block)
        ])
        self.logit = ArcMarginProduct_subcenter(self.embed_dim, num_class)

    def forward(self, inputs):
        x, x_mask = pack_seq(inputs, self.max_length)

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

        for block in self.encoder:
            x = block(x,x_mask)

        cls = x[:,0]
        cls = F.dropout(cls,p=0.4,training=self.training)
        logit = self.logit(cls)

        return logit