# parameterizedML

## Toy Model - Gaussian signals on a flat background

### Histogram for 5 gaussian signals and a flat background
* Number of data points per signal/background: 50,000
* Signal: Gaussians at mu=-2,-1,0,1,2
* Background: Flat
* Bins: 25
* Bin Width: 0.4
[Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/histogram_gaussian.png)

### Fixed Training
Each gaussian (i.e. mu=-2, -1, 0, 1, 2) is trained with a seperate NN and then prediction outputs are plotted vs input values between [-5, 5].
[Fixed Training](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/fixedTraining.png)

### Parameterized Training
A NN is trained for gaussians at mu=-2, -1, 0, 1, 2 and predictions are made at mu=-1.5, -1, -0.5, 0, 0.5, 1, 1.5. Thus, predictions at half-odd integer values (i.e. mu=-1.5, -0.5, 0.5, 1.5) are interpolations based on training at integer values. 
[Parameterized Training](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/paramTraining_complete.png)

### Parameterized Training
ROC curves and AUC values for the same set of predictions made in the parameterized training set (i.e. training at mu=-2, -1, 0, 1, 2 with predictions at mu=-1.5, -1, -0.5, 0, 0.5, 1, 1.5)
[Parameterized ROC Curve](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/ROC_parameterized.png)

---

## Xttbar - mWWbb

### Histogram for 5 signals and corresponding background
* Signal at mx = 500 GeV, 750, 1000, 1250, 1500
* Bin Width: 50 GeV
[Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/signal_background_histogram.png)

### Fixed and Parameterized Training
A seperate NN is trained to seperate signal and background at each of the 5 fixed mx values (mx = 500, 750, 1000, 1250, 1500) and plotted (plot lines). Next, a seperate NN is trained with for all signals and backgrounds but each time excluding one of the mx values so that this mx value can be interpolated from neighboring signals (circle markers). For example, the NN excluding mx=500 GeV (i.e. trained with mx=750 GeV, 1000, 1250, 1500) is used to predict the outputs for a signal at mx=500 GeV. Similarly, the NN trained at mx=500 GeV, 1000, 1250, 1500 (i.e. excluding mx=750 GeV) is used to predict the outputs for a signal at mx=750 GeV.   
[Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/parameterized_vs_fixed_output_plot.png)

### Fixed and Parameterized ROC/AUC
ROC curves and AUC values for the fixed and parameterized training performed in the previous section.
[Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/parameterized_vs_fixed_ROC_plot.png)

### Fixed and Parameterized Output Distribution
NN output for the fixed and parameterized training in the previous section is plotted as a histogram to check the way in which output is parsed between signal (1) and background (0) scores. 
[Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/parameterized_vs_fixed_output_histogram.png)

