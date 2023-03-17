import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# class ArcMarginProduct(nn.Module):
#     def __init__(
#         self,
#         config,
#         in_features,
#         out_features,
#         scale=30.0,
#         margin=0.50,
#         easy_margin=False,
#         ls_eps=0.0,
#     ):
#         super(ArcMarginProduct, self).__init__()
#         self.config = config

#         self.in_features = in_features
#         self.out_features = out_features
#         self.scale = scale
#         self.margin = margin
#         self.ls_eps = ls_eps  # label smoothing
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)
#         self.th = math.cos(math.pi - margin)
#         self.mm = math.sin(math.pi - margin) * margin

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#         one_hot = torch.zeros(cosine.size(), device=self.config.training.device)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         if self.ls_eps > 0:
#             one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.scale

#         return output


# class Model(nn.Module):
#     def __init__(self, cfg):
#         super(Model, self).__init__()
#         self.cfg = cfg
#         self.lin_bn_mish = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("lin_mish1", lin_bn_mish(472, 512)),
#                     ("lin_mish2", lin_bn_mish(512, 256)),
#                     ("lin_mish3", lin_bn_mish(256, 256)),
#                     ("lin_mish4", lin_bn_mish(256, 128)),
#                 ]
#             )
#         )

#         self.final = ArcMarginProduct(
#             128,
#             self.cfg.target_size,
#             scale=self.cfg.scale,
#             margin=self.cfg.margin,
#             easy_margin=False,
#             ls_eps=0.0,
#         )
#         self.fc_probs = nn.Linear(128, self.cfg.target_size)

#     def forward(self, x, label):
#         feature = self.lin_bn_mish(x)
#         if self.cfg.arcface:
#             arcface = self.final(feature, label)
#             probs = self.fc_probs(feature)
#             return probs, arcface
#         else:
#             probs = self.fc_probs(feature)
#             return probs


# def lin_bn_mish(input_dim, output_dim):
#     return nn.Sequential(
#         OrderedDict(
#             [
#                 ("lin", nn.Linear(input_dim, output_dim, bias=False)),
#                 ("bn", nn.BatchNorm1d(output_dim)),
#                 ("dropout", nn.Dropout(0.2)),
#                 ("relu", nn.Mish()),
#             ]
#         )
#     )


# class ASLLinearModel(torch.nn.Module):
#     def __init__(
#         self,
#         config,
#         in_features: int,
#         first_out_features: int,
#         num_classes: int,
#         num_blocks: int,
#         drop_rate: float,
#     ):
#         super(ASLLinearModel, self).__init__()
#         self.config = config

#         blocks = []
#         out_features = first_out_features
#         for idx in range(num_blocks):
#             if idx == num_blocks - 1:
#                 out_features = num_classes

#             blocks.append(self._make_block(in_features, out_features, drop_rate))

#             in_features = out_features
#             out_features = out_features // 2

#         self.model = nn.Sequential(*blocks)

#     def _make_block(self, in_features, out_features, drop_rate):
#         return nn.Sequential(
#             nn.Linear(in_features, out_features),
#             nn.BatchNorm1d(out_features),
#             nn.ReLU(),
#             nn.Dropout(drop_rate),
#         )

#     def forward(self, x):
#         return self.model(x)


class ASLModel(nn.Module):
    def __init__(self, p, in_features, n_class):
        super(ASLModel, self).__init__()
        self.dropout = nn.Dropout(p)
        self.layer0 = nn.Linear(in_features, 1024)
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.Linear(512, n_class)
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class ASLLinearModel(nn.Module):
    def __init__(
        self,
        config,
        in_features: int,
        first_out_features: int,
        num_classes: int,
        num_blocks: int,
        drop_rate: float,
    ):
        super().__init__()
        self.config = config

        blocks = []
        out_features = first_out_features
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                out_features = num_classes

            blocks.append(self._make_block(in_features, out_features, drop_rate))

            in_features = out_features
            out_features = out_features // 2

        self.model = nn.Sequential(*blocks)

    def _make_block(self, in_features, out_features, drop_rate):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.model(x)