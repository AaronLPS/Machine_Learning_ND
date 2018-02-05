# README

## Software and libraries used in ths Project
### Software/Platform:
- Pycharm, Jupyter notebook, Anaconda, AWS EC2, 
### Libraries:
- Tensorflow, Tensorboard, keras, 
- numpy, Pandas, matplotlib, PIL
- glob, os

## Datasets
Datasets[for tain/test] were downloaded from: https://www.kaggle.com/c/plant-seedlings-classification/data
Evaluation datasets[Images From The Wild (15MB)] were downloaded from
https://vision.eng.au.dk/?download=/data/WeedData/ImagesFromTheWild.zip

## Quick Start
1. Read __Capstone_Report-V1-2.pdf__ to see the introduction and the implementation report
2. Run __seedlingClassificationCNN.py__ to train the model. During the refinement, a series of model architectures have been implemented and trained here. In my case, I was using AWS EC2 to train those models.
3. Run __seedlingClassificationPredict.py__ to test the model. Model evaluation was also implemented in this file.

## Description of other files
4. Run __./src/Train_data_visualization.ipynb__ to find the visualization of training dataset
5. Run __Model Evaluation.ipynb__ to find information about the evaluation datasets
6. Run __Model_Visualisation.ipynb__ to see the architecture of models listed at capstone report(Refinement Section).
7. Run __Benchmark_CNN_keras_model_V1-2.ipynb__ to train the benchmark model(Benchmark model can also be trained at __seedlingClassificationCNN.py__ using GPU). It is the first try while I started this project at my PC.
