# Google - Isolated Sign Language Recognition
In this repository you can find the solution and code for [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/discussion) competition (top-50).

## General pipeline

## Validation

## Metric learning

## Modeling

## Preprocessing
My preprocessing pipeline can be described as follows:
1. Drop Z-coordinates.
2. Sequences with length higher than 384 are center cropped to 384, sequences with lower length aren't changed.
3. Apply mean-std normalization.
4. Take only crucial keypoints.
5. Hand distances are generated.
6. Motion features are generated.

Advanced normalization (moving coordinate system to the center and rotating, and scaling it), joint angles, OX-angles, External distances and using constant number of frames didn't improve the score.

Leaving only one hand leads to a higher score, but the same perfomance could be achieved with the hflip augmentation, so I decided to choose the second option in order to make room for TTA (which wasn't used in the final submission due to the time limit).

## Augmentations
My solution uses `Shift, Scale, Rotate and Hflips` augmentations. Besides them, I checked another augmentations like Random Frame Drops, Interpolation, Noise, Mix Up and Local Affines, but they didn't improved the score. 

## Training setup
The model was trained for `100 epochs` with `Mixed Precision`, `batchsize = 64` (other values showed worse perfomance) and `Ranger` optimizer (learning rate schedule: 1e-6 → 1e-4 → 1e-7).

I also tested another optimizers like MADGRAD, AdamW, Lion, Ranger optimizer and SAM, but the best perfomance was achieved with Ranger. 

Besides that, I tried to split the training into several stages such as encoder pretrain, arcface tuning and late dropout, but none of them led to a higher score.

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
