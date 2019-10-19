# AirportScan

Threats detection using 3D body scan images at an airport is a vital process for
the safety of an air travel.  Training a 3D Convolutional Neural Network (CNN)
requires an extensive computation power and a large number of training examples.
With our limited number of examples, we utilized a multi-view 2D CNN (MVCNN)
approach using a pre-trained VGG-16 model on the ImageNet dataset. In this
paper, we propose two architectures of MVCNNs to accurately detect threats from
projected 2D body scan images:
1. fine-tuning later parts of the VGG-16 feature extractions
2. training on the fully connected layers of concatenated output from transfer 
learning.  

Our models capture information from eight different angles around a body scan by
- element-wise max pooling in fine-tuning MVCNN, and
- concatenating the output from each angle and train for more layers in
transfer learning MVCNN.  

Both architectures yielded an accuracy higher than 95%
on the test set. The predictions from the transfer learning MVCNN are slightly
better than the fine-tuning MVCNN.

The summary and visualization of the project can be found in
[poster.pdf](poster.pdf). The dataset is provided by TSA and is a part of a
[kaggle
competition](https://www.kaggle.com/c/passenger-screening-algorithm-challenge)

This work is a collaboration with Chayawan Jaikla (@cjaikla)
