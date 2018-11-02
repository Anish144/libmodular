# libmodular

A library to learn modules in a modular neural network done for UCL's CSML course thesis, supervised by Dr. David Barber

## How does it work

Uses a variational dropout scheme to learn globally useful modules. Within these, a controller conditionally executes to ensure the relevant inputs are passed onto each module. The entire scheme is trained via the pathwise gradient estimator.

## Relevant Files

cifar10.py runs the experiments on the default dataset of cifar10, the dataset can be changed inside the arguments dictionary in the file. The modular network itself is contained in libmodular/modular.py and the controller is in libmodular/layers.py
