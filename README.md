# parameterizedML

## Gaussian
<table style="width:100%">
<tr>
<td align=center> <h4>Figure</h4> </td>
<td align=center> <h4>Parameters</h4> </td>
</tr>
<tr>
<td width=65%, align=center>
<h4>Histogram of Signal/Background</h4>
<img src="/gaussian/plots/images/histogram_gaussian.png"><a href="/gaussian/plots/histogram_gaussian.pdf">[Download PDF]</a></td>
<td>
<ul>
<li> # of data points per signal/background: 10,000</li>
<li> Signal: Gaussians at mu=-2,-1,0,1,2</li>
<li> Background: Flat </li>
<li> Bins: 100 </li>
<li> Bin Width: 0.1 </li>
</ul>
</td>
</tr>
<tr>
<td width=65%, align=center>
<h4>Fixed Training</h4>
<img src="/gaussian/plots/images/fixedTraining.png"><a href="/gaussian/plots/fixedTraining.pdf">[Download PDF]</a></td>
<td>
<ul>
<li>Fixed training at mu=-2, -1, 0, 1, 2</li>
</ul>
</td>
</tr>
<tr>
<td width=65%, align=center>
<h4>Parameterized Training</h4>
<img src="/gaussian/plots/images/paramTraining_complete.png"><a href="/gaussian/plots/paramTraining_complete.pdf">[Download PDF]</a></td>
<td>
<ul>
<li>Predictions made at mu=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2</li>
<li>Predictions at trained values of mu: mu=-2,-1,0,1,2</li>
<li>Predictions at intermediate values of mu: mu=-1.5,-0.5,0.5,1.5</li>
</ul>
</td>
</tr>
<tr>
<td width=65%, align=center>
<h4>Fixed Training ROC Curve</h4>
<img src="/gaussian/plots/images/ROC_Fixed.png"><a href="/gaussian/plots/ROC_Fixed.pdf">[Download PDF]</a></td>
<td>
<ul>
<li>ROC Curve for fixed training at mu=-2,-1,0,1,2 </li>
</ul>
</td>
</tr>
</table>

---

## Xttbar - mWWbb

<table style="width:100%">
<tr>
<td align=center> <h4>Figure</h4> </td>
<td align=center> <h4>Parameters</h4> </td>
</tr>
<tr>
<td width=65%, align=center><img src="/mwwbb/plots/images/mWWbb_histogram.png"><a href="/mwwbb/plots/mWWbb_histogram.pdf">[Download PDF]</a></td>
<td>
<ul>
<li> Signal at mx=500, 750, 1000, 1250, 1500</li>
<li> Bin Width: 50 GeV </li>
</ul>
</td>
</tr>
<tr>
<td width=65%, align=center><img src="/mwwbb/plots/images/fixedTraining.png"><a href="/mwwbb/plots/fixedTraining.pdf">[Download PDF]</a></td>
<td>
<ul>
<li>Fixed training at mu=1000 GeV</li>
</ul>
</td>
</tr>
<tr>
<td width=65%, align=center><img src="/mwwbb/plots/images/paramTraining_complete.png"><a href="/mwwbb/plots/paramTraining_complete.pdf">[Download PDF]</a></td>
<td>
<ul>
<li>Fixed training plot at mu=1000 GeV (red x)</li>
<li>Fixed training at mu=500, 750, 1250, 1500 (circles)</li>
<li>Prediction at mu=1000 based on fixed training at mu=500, 750, 1250, 1500 (red dots)</li>
</ul>
</td>
</tr>
<tr>
<td width=65%, align=center> 
<img src="/mwwbb/plots/images/ROC_parameterized.png"><a href="/mwwbb/plots/ROC_parameterized.pdf">[Download PDF]</a></td>
<td>
<ul>
<li>ROC curve for fixed training plot at mu=1000 GeV (red x)</li>
<li>ROC curve for fixed training at mu=500, 750, 1250, 1500 (circles)</li>
<li>ROC curve at predicted value of mu=1000 GeV based on fixed training at mu=500, 750, 1250, 1500 (red dots)</li>
</ul>
</td>
</tr>
</table>