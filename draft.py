import torch
import numpy as np

a = np.ones([100, 1210])
del a[5]
print(a.shape)
