import torch

a = torch.ones((64, 101, 512))
print(torch.cat((a, a),dim =2).shape)