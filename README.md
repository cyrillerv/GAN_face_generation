# Face Generation using Generative Adversarial Networks (GANs)

The goal of this project is to train a **Generative Adversarial Network (GAN)** capable of generating realistic human faces of people who do not exist in real life.

---

## Dataset

- **CelebA** dataset from Kaggle  
- ~200,000 face images  
- ~10,000 different celebrities  

---

## Tools

- Python
- PyTorch

---

## Challenges Encountered

During training, the **discriminator quickly became too strong**, leading to the following issue:
- It was able to identify fake images almost perfectly
- The generator received little to no meaningful gradient
- Training stagnated

---

## Implemented Solutions

To stabilize the training process, several techniques were applied:

- **Reduced discriminator learning rate**
- **Label smoothing**:
  - Real images → target label `0.9`
  - Fake images → target label `0.1`
  - Prevents the discriminator from becoming overly confident
- **Noise injection on real images**:
  - Gaussian noise added to real images
  - Noise magnitude gradually reduced over epochs

These adjustments significantly improved training stability and generator learning.

---

## Features

- Automatic dataset download
- GAN training with PyTorch
- Model checkpoint saved at each epoch
- Training can be resumed at any time from the last checkpoint
- Image generation and visualization during training
- Export of trained generator model

---

## Installation

```bash
git clone https://github.com/username/gan-face-generation.git
cd gan-face-generation
pip install -r requirements.txt
