import torch
xyz = torch.zeros([64, 100, 512])
xyz = torch.cat([
    torch.zeros(64, 1, 512),
    xyz
],1)
print(xyz.shape)