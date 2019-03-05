
# FP-GAN TensorFlow
This repository contains a re-implementation of the model proposed by Lee et al. (2018) in their paper "Unpaired image-to-image translation using cycle-consistent adversarial networks".

As an extension, we added a feature preserving cost function. We preserve eye gaze directions, as well as regional landmarks when training the image translation networks.

A report explaining this method in detail will be uploaded soon.

## Quick Start

1. Download the datasets
2. Create a config file for the FP-GAN
3. (optional) Train the feature consistency models for eye gaze and/or landmarks consistency
3. Train an FP-GAN model
4. Translate images
5. (optional) Create a config file for a gaze estimator. Then, train a gaze estimation network and run inference.


## Folder Structure
The ```src``` folder contains the following sub-folders.

* ```input```: dataset classes and preprocessing scripts
* ```models```: eye gaze estimation models, GAN models (Generator, Discriminator, Feature Models) and scripts for model export / inference
* ```run```: scripts for running model training, inference, testing, visualisations
* ```util```: various helper scripts
* ```visualisations```: various visualisation scripts

## References

* LEE , K., KIM , H., AND SUH , C. 2018. Simulated+unsupervised learning with adaptive data
generation and bidirectional mappings. In International Conference on Learning Represen-
tations.
* ZHU, J., PARK, T., ISOLA, P., AND EFROS, A. A. 2017. Unpaired image-to-image translation
using cycle-consistent adversarial networks. CoRR abs/1703.10593.
* CycleGAN paper: https://arxiv.org/abs/1703.10593
* Official CycleGAN source code in Torch: https://github.com/junyanz/CycleGAN
