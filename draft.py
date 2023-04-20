import torch.nn as nn

cls = nn.Sequential(
    nn.Linear(512, 512 * 2),
    nn.LayerNorm(512 * 2),
    nn.Hardswish(),
    nn.Dropout(0.4),
    nn.Linear(512 * 2, 512),
    nn.LayerNorm(512),
    nn.Hardswish(),
)

print(cls[3].p)