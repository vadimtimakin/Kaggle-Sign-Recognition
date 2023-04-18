import torch
a = torch.zeros([64, 101, 512])
b = torch.zeros([64, 101, 1024])
print(a[:, 0].shape)
c = torch.cat([a, b], axis=2)
print(c.shape)