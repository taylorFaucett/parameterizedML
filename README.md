# parameterizedML

This script utilizes [Theano](http://deeplearning.net/software/theano/), [Pylearn2](http://deeplearning.net/software/pylearn2/) and alexjc's [scikit-neuralnetwork](https://github.com/aigamedev/scikit-neuralnetwork) to create a fixed and parameterized machine learning scheme. Datasets are generated for multiple signals and a uniform (i.e. flat) background. A regression neural network (NN) is used to learn for N signals at fixed means which are mapped as a 1D array to signal/background values of 1 or 0. Second, parameterized training is done for those same N signals simultaneously and then predictions are made for the signals parameterized by a secondary input (alpha).

## Gaussians on a flat background

### Fixed Training
Initially, signal is generated for a toy model of gaussians and a flat background

![Distribution Map](/gaussian/plots/images/modelPlot.png)

Theano/Pylearn2's NN learns from these inputs and maps testdata to signal/background for fixed mean values (mu=-2, -1,0, +1, +2).

![Fixed Training](/gaussian/plots/images/fixedTraining.png)

### Parameterized Training
using those same sets of training gaussians

![Complete Gaussian Set](/plots/images/paramTraining.png)

Changing the input to be parameterized in terms of a variable (alpha) such that input = (x, alpha), the NN can now train across the parameterized input such that output for untrainined mu values (e.g. mu=-1.5, -0.5, 0.5, 1.5) can interpolate from trained gaussians in those untrained regions.

![Parameterized Training (Complete Set)](/plots/images/paramTraining_complete.png)

## X->ttbar data

Running the same analysis on real world data

![Histogram 400](/mwwbb/plots/images/histograms/histo_400.png)

for 12 energy samples

| ![Histogram 400](/mwwbb/plots/images/histograms/histo_400.png) | ![Histogram 500](/mwwbb/plots/images/histograms/histo_500.png) | ![Histogram 600](/mwwbb/plots/images/histograms/histo_600.png) |
| ![Histogram 700](/mwwbb/plots/images/histograms/histo_700.png) | ![Histogram 800](/mwwbb/plots/images/histograms/histo_800.png) | ![Histogram 900](/mwwbb/plots/images/histograms/histo_900.png) |
| ![Histogram 1000](/mwwbb/plots/images/histograms/histo_1000.png) | ![Histogram 1100](/mwwbb/plots/images/histograms/histo_1100.png) | ![Histogram 1200](/mwwbb/plots/images/histograms/histo_1200.png) |
| ![Histogram 1300](/mwwbb/plots/images/histograms/histo_1300.png) | ![Histogram 1400](/mwwbb/plots/images/histograms/histo_1400.png) | ![Histogram 1500](/mwwbb/plots/images/histograms/histo_1500.png) |

### Fixed Training

![X->ttbar](/mwwbb/plots/images/fixedTraining.png)

### Parameterized Training
For fixed training at mu=500, 1000, 1500 
![X->ttbar](/mwwbb/plots/images/paramTraining.png)
we can interpolate at energies not trained for (i.e. mu=750, 1250)
![X->ttbar](/mwwbb/plots/images/paramTraining_complete.png)
