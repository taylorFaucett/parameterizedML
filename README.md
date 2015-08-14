# parameterizedML

## Toy Model - Gaussian signals on a flat background

### Histogram for 5 gaussian signals and a flat background
* Number of data points per signal/background: 50,000
* Signal: Gaussians at mu=-2,-1,0,1,2
* Background: Flat
* Bins: 25
* Bin Width: 0.4

![Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/histogram_gaussian.png)

### Fixed Training
Each gaussian (i.e. mu=-2, -1, 0, 1, 2) is trained with a seperate NN and then prediction outputs are plotted vs input values between [-5, 5].

![Fixed Training](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/fixedTraining.png)

### Parameterized Training
A NN is trained for gaussians at mu=-2, -1, 0, 1, 2 and predictions are made at mu=-1.5, -1, -0.5, 0, 0.5, 1, 1.5. Thus, predictions at half-odd integer values (i.e. mu=-1.5, -0.5, 0.5, 1.5) are interpolations based on training at integer values. 

![Parameterized Training](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/paramTraining_complete.png)

### Parameterized Training
ROC curves and AUC values for the same set of predictions made in the parameterized training set (i.e. training at mu=-2, -1, 0, 1, 2 with predictions at mu=-1.5, -1, -0.5, 0, 0.5, 1, 1.5)

![Parameterized ROC Curve](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/gaussian/plots/images/ROC_parameterized.png)

---

## X->ttbar - mWWbb

### Histogram for 5 signals and corresponding background
* Signal at mx = 500 GeV, 750, 1000, 1250, 1500
* Bin Width: 50 GeV

![Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/signal_background_histogram.png)

### Fixed and Parameterized Training
A seperate NN is trained to seperate signal and background at each of the 5 fixed mx values (mx = 500, 750, 1000, 1250, 1500) and plotted (plot lines). Next, a seperate NN is trained with for all signals and backgrounds but each time excluding one of the mx values so that this mx value can be interpolated from neighboring signals (circle markers). For example, the NN excluding mx=500 GeV (i.e. trained with mx=750 GeV, 1000, 1250, 1500) is used to predict the outputs for a signal at mx=500 GeV. Similarly, the NN trained at mx=500 GeV, 1000, 1250, 1500 (i.e. excluding mx=750 GeV) is used to predict the outputs for a signal at mx=750 GeV.   

![Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/parameterized_vs_fixed_output_plot.png)

### Fixed and Parameterized ROC/AUC
ROC curves and AUC values for the fixed and parameterized training performed in the previous section.

![Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/parameterized_vs_fixed_ROC_plot.png)

### Fixed and Parameterized Output Distribution
NN output for the fixed and parameterized training in the previous section is plotted as a histogram to check the way in which output is parsed between signal (1) and background (0) scores. 

![Signal and Background histogram](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb/plots/images/parameterized_vs_fixed_output_histogram.png)

## X->ttbar and W->jj as a function of the jet energy scale
2 mass input values (mWWbb and mjj) are used to train a NN along with jes values of 0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250. 

### Mass values for WWbb

<table style="width:100%">
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_0.750.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_0.900.png"></td>    
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_0.950.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_0.975.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_1.000.png"></td>    
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_1.025.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_1.050.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_1.100.png"></td>    
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mWWbb_histogram_1.250.png"></td>
  </tr>
</table>

### Mass values for jj

<table style="width:100%">
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_0.750.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_0.900.png"></td>    
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_0.950.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_0.975.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_1.000.png"></td>    
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_1.025.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_1.050.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_1.100.png"></td>    
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/histograms/images/mjj_histogram_1.250.png"></td>
  </tr>
</table>

### Fixed Training Output
With 2 input mass values, plotting the NN output requires a 3D plot. This is done using a color-coded heatmap.

<table style="width:100%">
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_0.750.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_0.900.png"></td>   
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_0.950.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_0.975.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_1.000.png"></td>   
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_1.025.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_1.050.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_1.100.png"></td>   
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_plot_surface_1.250.png"></td>
  </tr>
</table>

Changes in the output as a function of the jet energy scale is subtle but can be easier seen when animated

<img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/fixed/fixed_output_animation.gif">

### Parameterized Training Output
The same inputs, but interpolated through the parameterized method, yield similar results

<table style="width:100%">
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_0.750.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_0.900.png"></td>   
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_0.950.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_0.975.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_1.000.png"></td>   
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_1.025.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_1.050.png"></td>
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_1.100.png"></td>   
    <td><img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/param_output_plot_surface_1.250.png"></td>
  </tr>
</table>


<img src="https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/output_heat_map/images/parameterized/parameterized_output_animation.gif">

### Fixed and parameterized ROC/AUC plots
The Receiver Operating Characteristic can be plotted for both the fixed training and interpolated/parameterized training method

#### Fixed ROC Plot
![Fixed ROC Plot](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/images/fixed_ROC_plot.png)

#### Parameterized ROC Plot
![Parameterized ROC Plot](https://raw.githubusercontent.com/tfaucett/parameterizedML/master/mwwbb_jes/plots/images/parameterized_ROC_plot.png)
