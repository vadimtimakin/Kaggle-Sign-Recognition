# Google - Isolated Sign Language Recognition
In this repository you can find the solution and code for [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/discussion) competition (top-50).

## General pipeline

## Validation

## Metric learning

## Modeling

## Preprocessing

## Augmentations

## Training setup

## Model optimization
My model was initially written on PyTorch and converted to TFLite using `PyTorch → torch.jit.script() → ONNX → TFLite` pipeline before submission. Preprocessing part for the inference model  was rewritten on TensorFlow due to the RAM leakage in PyTorch version. I also used `fp16` inference for a submission. In some of my submission I replaced `Swish` activation functions to `ReLU`, it slightly decreased a score, but gave a noticable advantage in model's speed.

## Some other things that didn't work
- Dropping messy participant
- Voting ensemble
- Max-prob ensemble

## Didn't have time to try
- Using pretrained models
- Pretrain on another datasets
- External data
- Graph networks
- CNNs

## Bonus
I also shared some thoughts on model generalization and possible score plateau [here](https://www.kaggle.com/competitions/asl-signs/discussion/406457).
