# Google - Isolated Sign Language Recognition
In this repository you can find the solution and code for [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/discussion) competition (top-50).

## General pipeline
<a href="https://github.com/t0efL/Kaggle-Sign-Recognition/blob/main/Pipeline.jpg"><img alt="Pipeline" src="https://github.com/t0efL/Kaggle-Sign-Recognition/blob/main/Pipeline.jpg"></a>

## Validation
Using StratifiedGroupKFold (group by participant_id) was crucial in the beginning of competition in order to prevent overfitting, however later, after adding some regularization to the training process, simple StratifiedKFold or Random Seed Ensemble trained on the full data gave the same results. Final solution uses 5 of 5 folds of `StratifiedGroupKFold` split, has approximate size of `35 MB` and runs in around `50 minutes`.

## Metric learning
The main advantage I got was from treating this problem as a metric learning problem (since there are 250 classes). I changed CrossEntropyLoss to `ArcFaceLoss` (later improved with `label smoothing = 0.2`) and added `ArcMarginProduct` after the encoder block resulting to a significantly higher model perfomance. Since the classification was based on several body parts I decided to use `Sub Center ArcMarginProduct block`, which also led to a better score.

Inspired by this, I spent more time on this idea and tested ArcFaceLoss based on FocalLoss and combination of ArcFaceLoss and CrossEntropy but it didn't bring better results.

I also tried using modern approaches like AdaCos, FaceCos, MagFace, Multi-Similarity loss functions and others. However, they didn't improve model's perfomance. I think the reason of this lies in fact that the main advantage of those loss functions comes from dealing with few number of samples or class imbalance using adaptive margin algorithms which wasn't relevant in this competition since all classes were balanced and presented with enough samples.

## Modeling
I used `Transformer` as the main model in this competition. I was able to achieve the best perfomance using `small number of encoder blocks (1)` and `big number of heads (16)`. I also added `MLP embedding` before encoder block and `Sub Center ArcMarginProduct layer` after it. As an activation function I used `Swish`. The perfomanced was achieved with `dropout = 0.4`.

I did some experiments with LSTM and attention modifications but those didn't succeed.

Besided that, I observed better results with higher number of MLP layers, but didn't use it since it led to more memory consumption and slower speed.

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
