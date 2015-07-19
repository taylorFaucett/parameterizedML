# parameterizedML

This script utilizes [Theano](http://deeplearning.net/software/theano/), [Pylearn2](http://deeplearning.net/software/pylearn2/) and alexjc's [scikit-neuralnetwork](https://github.com/aigamedev/scikit-neuralnetwork) to create a fixed and parameterized machine learning scheme. Datasets are generated for multiple gaussian shaped signals and a uniform (i.e. flat) background. trainFixed uses a regression neural network (NN) to learn for N gaussians at fixed means (mu) which can map a 1D array to signal/background values of 1 or 0. trainParam trains for all N gaussians simultaneously and then uses then performs predictions for these gaussian signals parameterized by a secondary input (alpha).

## Outputs

### Fixed Training
Under fixed training, signal data is generated according to a gaussian distribution with background data generated according to a uniform/flat distribution.

![Distribution Map](/plots/images/modelPlot.png)

Theano/Pylearn2's NN learns from these inputs and maps testdata to signal/background for fixed mean values (mu=-2, -1,0, +1, +2).

![Fixed Training](/plots/images/fixedTraining.png)

### Parameterized Training
using those same sets of training gaussians

![Complete Gaussian Set](/plots/images/paramTraining.png)

Changing the input to be parameterized in terms of a variable (alpha) such that input = (x, alpha), the NN can now train across the parameterized input such that output for untrainined mu values (e.g. mu=-0.5 and mu=+0.5) can interpolate from other trained gaussians (e.g. mu=-1.0, mu=0.0, mu=+1.0) in those untrained regions.

#### Parameterized Training (Complete Set)
![Parameterized Training (Complete Set)](/plots/images/paramTraining_complete.png)
