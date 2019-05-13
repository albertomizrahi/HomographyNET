# COS 529 Research Project
**Author:** Alberto Mizrahi
Spring 2019

A description of the relevant files in this repo:
- **Generating Homography Data.ipynb:** This Jupyter notebook describes and illustrates the process of generating a training homography example from a given image.
- **Test Homography Data.ipynb:** This Jupyter notebook takes two images, estimates a homography between them and compares it with the ground truth homography.
- **hom_RANSAC.py**: implements homography estimation using ORB feature detection and description + RANSAC.
- **hom_constants.py:** stores the various constants used for training and testing including path to the image datasets, number of epochs, batch size, etc.
- **hom_predict.py**: this program takes as input the architecture and weights of the trained model, generataes examples from the test dataset and determines how well the model. In particular, it calculates the mean average corner error and prediction time from all the predictions made by the mode. It also does the same thing for the ORB+RANSAC method.
- **hom_train.py:** trains the regression HomographyNET architecture.

For precomputed weights, download the zip file from [https://www.dropbox.com/sh/fxz42ptms7xsye7/AAA7oyL0FSjOT6k9RrjOJ-H-a?dl=0](https://www.dropbox.com/sh/fxz42ptms7xsye7/AAA7oyL0FSjOT6k9RrjOJ-H-a?dl=0).

All the files in the zip file are prefixed by _X_Y, where X is the number of epochs the model was run and Y is the number of training examples. Furthermore, there are three types of files:

 - **hom_net_reg_X_Y.json:** this file describes the architecture of the model.
 -  **hom_net_reg_X_Y.h5:** this file contains the weights that were obtained after training the model.
 - **examples_visualizations_X_Y.pickle:** contains a list of the image IDs that were used to train the model.

In particular, the first two files are needed to load the model into Keras.

Finally, the set of files for the model that had the best accuracy is _35_100000.

