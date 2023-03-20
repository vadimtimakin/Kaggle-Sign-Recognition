import numpy as np
from objects.model_inference import InputNet

a = np.random.random((120, 534, 3))
model = InputNet()
x = model(a)
print(x.shape)