# CMI - Detect Behavior with Sensor Data (Kaggle Gold Medal, lightsource part)

Full kaggle writeup: [link](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/11th-place-solution)

## Features

**Used:**
- IMU: base (`acc_x/y/z`, `rot_w/x/y/z`)
- From public solutions: `acc_mag`, `rot_angle`, `acc_mag_jerk`, `rot_angle_vel`, `linear_acc_x/y/z`, `linear_acc_mag`, `linear_acc_mag_jerk`, `angular_vel_x/y/z`, `angular_distance`
- TOF: raw 320 features
- THM: raw 5 features

**Unsuccessful experiments:**
- IMU: 6d quat representations (https://arxiv.org/pdf/1812.07035), sliding-window statistics, integrals (except linear velocity), euler angles, gravity-based features, positional features
- TOF/THM: inter-sensor differences, statistics, center of mass, gradients
- Demographics: failed due to limited number of subjects

## Preprocessing

**Used:**
- No normalization (better than per-sequence scaling and StandardScaler)
- TOF: replaced -1 with 255, then divided by 255
- Leftside padding and truncation, seq_len=120
 
**Unsuccessful experiments:**
- Filtering methods (firwin, wavelet, savgol, butterworth, median) degraded performance
- Kalman filtering was neutral
- Left-to-right handedness transformations were not useful (did not increase cv and lb, although for private it would probably be important)

## Modeling

**Two models:**
- IMU only
- IMU + TOF + THM (model selected at inference based on TOF availability)

**Branches:**
- acc: `acc_x/y/z`
- rot: `rot_w/x/y/z`
- fe1: hand-crafted features (jerks, velocities, angles, distance,
etc.)
- fe2: lag/lead diff, cumsum
- full: all IMU features
- thm: all THM features (only in IMU+TOF+THM model)
- tof1-tof5: TOF features from individual sensors (only in IMU+TOF+THM model)

**Extractors:**
- Public_SingleSensor - based on the most popular public model
- FilterNetFeatureExtractor (only IMU only) - modified from https://github.com/WhistleLabs/FilterNet
- ConvTran_SingleSensor_NoTranLol modified from https://github.com/Navidfoumani/ConvTran
- ResNet1D
- Public2_SingleSensor (only IMU only) - based on another public model that uses an extractor for each channel
- MultiResidualBiGRU - modified from https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410

**Aggregation:**
- Branch level: concat (better than MHA, comparable to GAT, GCN worse)
- Extractor level: multihead attention

**Heads:**
- gesture
- sequence type
- orientation
- additional gesture heads directly over extractors

**Unsuccessful experiments:**
- Architectures: inceptiontime, efficientnet, harmamba, transformers (husformer, medformer, squeezeformer, timemil, etc.), moderntcn, wavenet, 2D/3D extractors for TOF
- Using spectrograms, scalograms, line plots
- Heads: gesture start prediction, full behavior mask, demographic features

## Augmentations

**Used:**
- mixup and variants (cutmix, channelmix, zebra)
- channel masking
- jitter

**Unsuccessful experiments:**
- time shift, stretch, warp
- rotations, left/right reflections
- low pass filter augmentation

## Training

**Used:**
- Optimizers: AdamW, MuonWithAuxAdam (slightly better in some cases)

**Unsuccessful experiments:**
- TorchJD for multi-task learning
- Metric learning components to improve the separation of similar classes, also did not provide any gain (arcface, tripletloss, ...)
- EMA
