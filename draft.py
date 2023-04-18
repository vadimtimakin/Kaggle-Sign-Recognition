import torch
def gem(x, p=3, eps=1e-6):
    return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
a = torch.zeros([64, 101, 512])
b = torch.zeros([64, 101, 1024])
print(a[:, 0].shape)
c = torch.cat([a, b], axis=2)
print(gem(a).shape)
print(torch.nn.functional.avg_pool2d(a, (101, 1)).shape)