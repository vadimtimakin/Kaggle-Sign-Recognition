import torch
import numpy as np

a = torch.ones([23, 82, 2])
b = torch.min(a, dim=1).values
c = a.permute(1, 0, 2) - b
print(c.shape)
