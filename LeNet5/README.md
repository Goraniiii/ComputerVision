# LeNet-5 Classifier for POC Dataset

## Project Overview
This project implements and evaluates a LeNet-5–based classifier on a medical image dataset.

The goal is to benchmark the original LeNet-5 architecture against a set of modern deep learning methods and investigate how well classical CNNs perform on medical imaging tasks.

**Model presentation slide**: [introcv_lenet_5_presentation.pdf](https://github.com/Goraniiii/ComputerVision/blob/main/LeNet5/introcv_lenet_5_presentation.pdf)

---
## Dataset
A medical image dataset consisting of four histological categories.
All images are RGB with resolution 224×224.

- Chorionic_villi - Training: 1,391 | Testing: 390
- Decidual_tissue - Training: 926 | Testing: 349
- Hemorrhage - Training: 1,138 | Testing: 421
- Trophoblastic_tissue - Training: 700 | Testing: 351


## Experiments

1. **Original LeNet-5**  
   - input: 32*32 grayscale(1 channel)
   - Apply the architecture and method of the paper as it is.
   - Activation: tanh
   - Out: RBF
   - Loss: MSE

2. **LeNet-5 + Modern methods**  
   - input: 32*32 grayscale(1 channel)
   - Apply the original architecture but with modern methods.
   - Activation: ReLU
   - criterion: Cross Entropy

3. **Extended LeNet-5 for High-resolution RGB Images**  
   - input: 224*224 rgb(3 channel)
   - Architecture modified to accept 3-channel images
   - Activation: ReLU
   - criterion: Cross Entropy

---

## Code Organization

- **practice.ipynb**  
  - Main notebook for experiments and visualization.  

- **model.py**  
  - LeNet-5 architecture implementation using pytorch.  

- **util.py**  
  - Utility functions for training, preprocessing, and other functions.

- **dataset.py**
  - Custom dataset implementation using torch.util.data
 


---

## Notes
- For experiments 1 and 2, all images are internally converted from RGB → grayscale using Otsu thresholding before being resized to 32×32.
- Experiment 3 keeps the original RGB channels and uses 224×224 resolution.
- This project focuses on comparing classical CNN behavior with modern training practices on medical imaging tasks.
