'''
author Taylor Faucett <tfaucett@uci.edu>

This script utilizes Theano/Pylearn2 and SKLearn-NeuralNetwork to create a fixed
and parameterized machine learning scheme. Data taken from an X->ttbar selection
is imported as signal and a uniform (i.e. flat) background is generated and
appended. mwwbb_fixed uses a regression NN to learn for n signals at fixed means
(mx) which can map a 1D array to signal/background values of 1 or 0. trainParam
trains for all n signals simultaneously and then trains for these signals with a
parameter by a secondary input (alpha).
'''

import ROOT
import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer

def file_runner(directory):
    print 'Entering file_runner'
    files = glob.iglob(directory+'*.root')
    for data in files:
    	file_generate(data)

def flat_bkg(bkgNum, low, high):
    print 'Entering flat_bkg'
    print 'Genterating a flat background with %s data points betwee %s-%s' %(bkgNum, low, high)
    mx_values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    w = ROOT.RooWorkspace('w')

    w.factory('Uniform::f(x[%s,%s])' %(low, high))

    # Define variables
    x      = w.var('x')
    bkgpdf = w.pdf('f')

    # Fill traindata, testdata and testdata1
    print 'Generating background data'
    bkg_values = bkgpdf.generate(ROOT.RooArgSet(x), bkgNum)
    bkg_data = np.zeros((bkgNum, 3))
    for j in range(len(mx_values)):
        for i in range(bkgNum):
            bkg_data[i, 0] = bkg_values.get(i).getRealValue('x')
            bkg_data[i, 1] = mx_values[j]
            bkg_data[i, 2] = 0.000000

        np.savetxt('data/flat_bkg/bkg_mx_%s.dat' %mx_values[j], bkg_data, fmt='%f')

def file_generate(root_file):
    print 'Entering file_generate'
    print 'Generating data using values from: %s' %root_file
    mwwbb = root_export(root_file,'xtt','mwwbb')
    mx = root_export(root_file,'xtt','mx')
    mjj = root_export(root_file, 'xtt', 'mjj')
    target = root_export(root_file,'xtt','target')
    jes = root_export(root_file, 'xtt', 'jes')
    size = len(mwwbb)
    data = np.zeros((size, 5))
    if target[0] == 0.000000:
        label = 'bkg'
    elif target[0] == 1.000000:
        label = 'sig'
    for i in range(size):
        data[i, 0] = mwwbb[i]
        data[i, 1] = mjj[i]
        data[i, 2] = jes[i]
        data[i, 3] = target[i]
        data[i, 4] = mx[i]
    np.savetxt('data/root_export/%s_mx_%0.0f_jes_%0.3f.dat' %(label, mx[0], jes[0]), data, fmt='%f')

def root_export(root_file, tree, leaf):
    print 'Entering root_export'
    print 'Extracting data from file: %s' %root_file
    print 'Extracting data from TTree: %s' %tree
    print 'Extracting data from TLeaf: %s' %leaf
    f = ROOT.TFile(root_file)
    t = f.Get(tree)
    l = t.GetLeaf(leaf)
    size = t.GetEntriesFast()
    entries = []
    for n in range(size):
        t.Scan(leaf,'','',1,n)
        val = l.GetValue(0)
        entries.append(val)
    return entries

def file_concatenater():
    print 'Entering file_concatenater'
    sig_files = glob.iglob('data/root_export/sig_mx_*.dat')
    bkg_files = glob.iglob('data/root_export/bkg_mx_*.dat')
    #flt_dat = glob.iglob('data/flat_bkg/*.dat')
    for signal, background in zip(sig_files, bkg_files):
        sig = np.loadtxt(signal)
        bkg = np.loadtxt(background)
        #flt = np.loadtxt(flat)
        data_complete = np.concatenate((sig, bkg), axis=0)
        np.savetxt('data/concatenated/ttbar_mx_%0.3f.dat' %sig[0,2], data_complete, fmt='%f')
        #data_complete = np.concatenate((sig, flt), axis=0)
        #np.savetxt('data/concatenated/flat_mx_%0.0f.dat' %sig[0,1], data_complete, fmt='%f')


''' NN Training '''

def mwwbb_fixed():
    print 'Entering mwwbb_fixed'
    files = glob.iglob('data/concatenated/*.dat')
    for dat in files:
        data = np.loadtxt(dat)
        print 'Processing fixed training on mu=%s' %data[0,2]
        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network',
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    #n_stable=1,
                    #f_stable=100,
                    n_iter=10,
                    #learning_momentum=0.1,
                    batch_size=10,
                    learning_rule="nesterov",
                    #valid_size=0.05,
                    verbose=True,
                    #debug=True
                    ))])
        print nn
    	traindata = data[:,0:3]
    	targetdata = data[:,3]
        print traindata
        print targetdata
    	nn.fit(traindata, targetdata)

    	fit_score = nn.score(traindata, targetdata)
    	print 'Fixed training score = %s' %fit_score
    	outputs = nn.predict(traindata)
    	outputs = outputs.reshape((1,len(outputs)))
    	fixed_plot = np.vstack((traindata[:,0], traindata[:,1], outputs)).T
    	np.savetxt('data/plot_data/fixed_plot_%0.3f.dat' %data[0,2], fixed_plot, fmt='%f')

    	actual = targetdata
    	predictions = outputs[0]
    	fpr, tpr, thresholds = roc_curve(actual, predictions)
    	ROC_plot = np.vstack((fpr, tpr)).T
    	roc_auc = [auc(fpr, tpr)]

    	np.savetxt('data/plot_data/ROC/fixed_ROC_plot_%0.3f.dat' %data[0,2], ROC_plot, fmt='%f')
    	np.savetxt('data/plot_data/ROC/fixed_ROC_AUC_%0.3f.dat' %data[0,2], roc_auc)

    	pickle.dump(nn, open('data/pickle/fixed_%0.3f.pkl' %data[0,2], 'wb'))


def mwwbb_parameterized():
    print 'Entering mwwbb_parameterized'
    mwwbb_complete = np.concatenate((
                        np.loadtxt('data/concatenated/ttbar_mx_0.750.dat'),
                        #np.loadtxt('data/concatenated/ttbar_mx_0.90.dat'),
                        #np.loadtxt('data/concatenated/ttbar_mx_0.95.dat'),
                        #np.loadtxt('data/concatenated/ttbar_mx_0.97.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1.000.dat'),
                        #np.loadtxt('data/concatenated/ttbar_mx_1.02.dat'),
                        #np.loadtxt('data/concatenated/ttbar_mx_1.05.dat'),
                        #np.loadtxt('data/concatenated/ttbar_mx_1.10.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1.250.dat')),
                        axis=0)

    print 'Parameterized training on all signals'
    traindata = mwwbb_complete[:, 0:3]      
    targetdata     = mwwbb_complete[:, 3]

    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network',
            Regressor(
                layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                learning_rate=0.01,
                n_iter=10,
                #n_stable=1,
                #f_stable=0.001,
                #learning_momentum=0.1,
                batch_size=10,
                learning_rule="nesterov",
                #valid_size=0.05,
                verbose=True,
                #debug=True
                ))])

    nn.fit(traindata, targetdata)
    fit_score = nn.score(traindata, targetdata)
    print 'Parameterized training score = %s' %fit_score
    outputs = nn.predict(traindata)
    outputs = outputs.reshape((1, len(outputs)))
    param_plot = np.vstack((traindata[:,0], traindata[:,1], outputs)).T
    np.savetxt('data/plot_data/param.dat', param_plot, fmt='%f')
    pickle.dump(nn, open('data/pickle/param.pkl', 'wb'))

        

def scikitlearnFunc(mx, mjj, alpha):
    nn = pickle.load(open('data/pickle/param.pkl','rb'))
    traindata = np.array((mx, mjj, alpha), ndmin=2)
    outputs   = nn.predict(traindata)

    #print 'x,alpha,output =', x, alpha, outputs[0]
    #plt.plot(x, outputs, 'ro', alpha=0.5)
    return outputs[[0]]

def mwwbbParameterizedRunner():
    print 'Entering mwwbbParameterizedRunner'
    alpha = [0.75, 0.95, 1.00, 1.05, 1.25]
    size = 1000

    mx_75_raw = np.loadtxt('data/concatenated/ttbar_mx_0.750.dat')
    mx_90_raw = np.loadtxt('data/concatenated/ttbar_mx_0.900.dat')
    mx_95_raw = np.loadtxt('data/concatenated/ttbar_mx_0.950.dat')
    mx_97_raw = np.loadtxt('data/concatenated/ttbar_mx_0.975.dat')
    mx_100_raw = np.loadtxt('data/concatenated/ttbar_mx_1.000.dat')
    mx_102_raw = np.loadtxt('data/concatenated/ttbar_mx_1.025.dat')
    mx_105_raw = np.loadtxt('data/concatenated/ttbar_mx_1.050.dat')
    mx_110_raw = np.loadtxt('data/concatenated/ttbar_mx_1.100.dat')
    mx_125_raw = np.loadtxt('data/concatenated/ttbar_mx_1.250.dat')

    mx_75 = np.concatenate((mx_75_raw[:size, :],mx_75_raw[-size:, :]), axis=0)
    mx_90 = np.concatenate((mx_90_raw[:size, :],mx_90_raw[-size:, :]), axis=0)
    mx_95 = np.concatenate((mx_95_raw[:size, :],mx_95_raw[-size:, :]), axis=0)
    mx_97 = np.concatenate((mx_97_raw[:size, :],mx_97_raw[-size:, :]), axis=0)
    mx_100 = np.concatenate((mx_100_raw[:size, :],mx_100_raw[-size:, :]), axis=0)
    mx_102 = np.concatenate((mx_102_raw[:size, :],mx_102_raw[-size:, :]), axis=0)
    mx_105 = np.concatenate((mx_105_raw[:size, :],mx_105_raw[-size:, :]), axis=0)
    mx_110 = np.concatenate((mx_110_raw[:size, :],mx_110_raw[-size:, :]), axis=0)
    mx_125 = np.concatenate((mx_125_raw[:size, :],mx_125_raw[-size:, :]), axis=0)

    input_list = [mx_75[:,0:2],
                    mx_90[:,0:2],
                    mx_95[:,0:2],
                    mx_97[:,0:2],
                    mx_100[:,0:2],
                    mx_102[:,0:2],
                    mx_105[:,0:2],
                    mx_110[:,0:2],
                    mx_125[:,0:2]]

    actual_list = [mx_75[:,3],
                    mx_90[:,3],
                    mx_95[:,3],
                    mx_97[:,3],
                    mx_100[:,3],
                    mx_102[:,3],
                    mx_105[:,3],
                    mx_110[:,3],
                    mx_125[:,3]]
    input1 = input_list[0]
    input1 = input1[:,0]
    input2 = input_list[0]
    input2 = input2[:,1]

    print "Running on %s alpha values: %s" %(len(alpha), alpha)

    for a in range(len(alpha)):
        print 'working on alpha=%s' %alpha[a]
        inputs = []
        predictions = []
        for x in range(0, 2*size, 1):
            outputs = scikitlearnFunc(input1[x]/1.,input2[x]/1., alpha[a])
            inputs.append(input1[x]/1.)
            predictions.append(outputs[0][0])
            print 'Percent remaining: %0.3f' %(x/(2.0*size))
        data = np.vstack((inputs, predictions)).T
        np.savetxt('data/plot_data/param_alpha_%s.dat' %alpha[a], data, fmt='%f')
        actual = actual_list[a]
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        roc_auc = [auc(fpr, tpr)]

        roc_data = np.vstack((fpr, tpr)).T
        np.savetxt('data/plot_data/ROC/param_alpha_ROC_%s.dat' %alpha[a], roc_data, fmt='%f')
        np.savetxt('data/plot_data/ROC/param_alpha_ROC_AUC_%s.dat' %alpha[a], roc_auc)

def fixVSparam():
    print 'Entering fixVSparam'
    alpha = [750, 1000, 1250]
    size = 10000

    mx_750_raw = np.loadtxt('data/concatenated/ttbar_mx_750.dat')
    mx_1000_raw = np.loadtxt('data/concatenated/ttbar_mx_1000.dat')
    mx_1250_raw = np.loadtxt('data/concatenated/ttbar_mx_1250.dat')

    mx_750_sig = mx_750_raw[:size, :]
    mx_750_bkg = mx_750_raw[-size:, :]

    mx_1000_sig = mx_1000_raw[:size, :]
    mx_1000_bkg = mx_1000_raw[-size:, :]

    mx_1250_sig = mx_1250_raw[:size, :]
    mx_1250_bkg = mx_1250_raw[-size:, :]

    mx_750 = np.concatenate((mx_750_sig, mx_750_bkg), axis=0)
    mx_1000 = np.concatenate((mx_1000_sig, mx_1000_bkg), axis=0)
    mx_1250 = np.concatenate((mx_1250_sig, mx_1250_bkg), axis=0)

    input_list = [mx_750[:,0],
                    mx_1000[:,0],
                    mx_1250[:,0]]

    actual_list = [mx_750[:,2],
                    mx_1000[:,2],
                    mx_1250[:,2]]


    mwwbb_complete = np.concatenate((
                        np.loadtxt('data/concatenated/ttbar_mx_750.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1250.dat')),
                        axis=0)

    traindata      = mwwbb_complete[:,0:2]
    targetdata     = mwwbb_complete[:,2]

    print "Running on %s alpha values: %s" %(len(alpha), alpha)

    for a in range(len(alpha)):
        print 'working on alpha=%s' %alpha[a]
        predictions = []
        input = input_list[a]
        for x in range(0,2*size, 1):
            outputs = scikitlearnFunc(input[x]/1., alpha[a], alpha[a])
            predictions.append(outputs[0])
            print "%s - Percent Complete: %s" %(alpha[a], (x/(2.0*size))*100.0)
        actual = actual_list[a]
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        roc_auc = [auc(fpr, tpr)]
        roc_data = np.vstack((fpr, tpr)).T
        np.savetxt('data/plot_data/fixVSparam_%s.dat' %alpha[a], roc_data, fmt='%f')
        np.savetxt('data/plot_data/fixVSparam_ROC_AUC_%s.dat' %alpha[a], roc_auc)





''' Plots '''


def plt_histogram():
    print 'Entering plt_histogram'
    sig_dat = glob.iglob('data/root_export/sig_mx_*.dat')
    bkg_dat = glob.iglob('data/root_export/bkg_mx_*.dat')
    for signal, background in zip(sig_dat, bkg_dat):
        bin_size   = 50
        sig = np.loadtxt(signal)
        bkg = np.loadtxt(background)
        print 'Processing histogram for mu=%s' %sig[0,4]
        n, bins, patches = plt.hist([sig[:,0],bkg[:,0]],
                            bins=range(0,3000, bin_size), normed=True,
                            histtype='step', alpha=0.75, linewidth=2, 
                            label=['Signal', 'Background'], rasterized=True)
        plt.setp(patches)
        plt.title('jes=%s' %sig[0,4])
        plt.ylabel('Fraction of events$/%0.0f$ GeV' %bin_size)
        plt.xlabel('m$_{WWbb}$ [GeV]')
        plt.grid(True)
        plt.legend(loc='upper right')
        #plt.xlim([0, 3000])
        #plt.ylim([0, 35000])
        plt.savefig('plots/histograms/mWWbb_%s_histogram.pdf' %sig[0,4], dpi=400)
        plt.savefig('plots/histograms/images/mWWbb_%s_histogram.png' %sig[0,4])
        plt.clf()


        bin_size   = 15
        n, bins, patches = plt.hist([sig[:,1],bkg[:,1]],
                            bins=range(0,1000, bin_size), normed=True,
                            histtype='step', alpha=0.75, linewidth=2, 
                            label=['Signal', 'Background'], rasterized=True)
        plt.setp(patches)
        plt.title('jes=%s' %sig[0,4])
        plt.ylabel('Fraction of events$/%0.0f$ GeV' %bin_size)
        plt.xlabel('m$_{jj}$ [GeV]')
        plt.grid(True)
        plt.legend(loc='upper right')
        #plt.xlim([0, 3000])
        #plt.ylim([0, 35000])
        plt.savefig('plots/histograms/mjj_%s_histogram.pdf' %sig[0,4], dpi=400)
        plt.savefig('plots/histograms/images/mjj_%s_histogram.png' %sig[0,4])
        plt.clf()  


def fixed_plot(): 
    files = glob.iglob('data/plot_data/fixed_plot_*.dat')
    jes = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]
    i=0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for dat in files:
        data = np.loadtxt(dat)
        for c, m, zl, zh in [('r', 'o', -50, 25), ('b', '^', -30, -5)]:
            xs = data[:,0]
            ys = data[:,1]
            zs = data[:,2]
            ax.plot(xs, ys, zs, c=c, marker=m)
        ax.set_xlabel('$m_{WWbb}$')
        ax.set_ylabel('$m_{jj}$')
        ax.set_zlabel('NN output')
        data = np.loadtxt(dat)
        i=i+1
        plt.savefig('plots/fixedTraining_%0.3f.pdf' %jes[i], dpi=400)
        plt.savefig('plots/images/fixedTraining_%0.3f.png' %jes[i])
        plt.clf()

def fixed_ROC_plot():
    print "Entering fixed_ROC_plot"
    jes = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]
    ROC_files = glob.iglob('data/plot_data/ROC/fixed_ROC_plot_*.dat')
    AUC_files = glob.iglob('data/plot_data/ROC/fixed_ROC_AUC_*.dat')
    i=0
    for (ROC, AUC) in zip(ROC_files, AUC_files):
        ROC_data = np.loadtxt(ROC)
        AUC = np.loadtxt(AUC)
        plt.plot(ROC_data[:,0], ROC_data[:,1], '-', 
                    label='jes=%0.3f (AUC=%0.3f)' %(jes[i], AUC), 
                    rasterized=True)
        i=i+1
    plt.plot([0,1],[0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixed_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_ROC_plot.png')
    plt.clf()


def fixVSparam_plot():
    print 'Entering fixVSparam_plot'
    param_files = [np.loadtxt('data/plot_data/fixVSparam_750.dat'),
    			np.loadtxt('data/plot_data/fixVSparam_1000.dat'),
    			np.loadtxt('data/plot_data/fixVSparam_1250.dat')
    			]
    fixed_files = [np.loadtxt('data/plot_data/fixed_ROC_750.dat'),
    			np.loadtxt('data/plot_data/fixed_ROC_1000.dat'),
    			np.loadtxt('data/plot_data/fixed_ROC_1250.dat')
    			]
    AUC_param = [np.loadtxt('data/plot_data/fixVSparam_ROC_AUC_750.dat'),
    				np.loadtxt('data/plot_data/fixVSparam_ROC_AUC_1000.dat'),
    				np.loadtxt('data/plot_data/fixVSparam_ROC_AUC_1250.dat')]
    AUC_fixed = [#np.loadtxt('data/plot_data/fixed_ROC_AUC_500.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_750.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_1000.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_1250.dat'),
    				#np.loadtxt('data/plot_data/fixed_ROC_AUC_1500.dat')
    				]
    param_markers = ['go', 'ro', 'co']
    fixed_markers = ['g.', 'r.', 'c.']
    mx = [750, 1000, 1250]
    for i in range(len(param_files)):
    	plt.plot(param_files[i][:,0], param_files[i][:,1], param_markers[i], 
    				alpha=0.5, markevery=200, 
    				label='$\mu=$%s (AUC=%0.3f)' %(mx[i], AUC_param[i]), rasterized=True)
    for i in range(len(fixed_files)):
    	plt.plot(fixed_files[i][:,0], fixed_files[i][:,1], fixed_markers[i], 
    				alpha=1, markevery=1000, 
    				label='$\mu=$%s (AUC=%0.3f)' %(mx[i], AUC_fixed[i]),  rasterized=True)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixVSparam_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixVSparam_plot.png')
    plt.clf()

def param_ROC_plot():
    print 'Entering param_ROC_plot'
    param_files = glob.iglob('data/plot_data/ROC/param_alpha_ROC_*.dat')

    fixed_files = ['data/plot_data/ROC/fixed_ROC_plot_0.750.dat',
                    'data/plot_data/ROC/fixed_ROC_plot_1.000.dat',
                    'data/plot_data/ROC/fixed_ROC_plot_1.250.dat']

    AUC_param = glob.iglob('data/plot_data/ROC/param_alpha_ROC_AUC_*.dat')

    AUC_fixed = glob.iglob('data/plot_data/ROC/fixed_ROC_AUC_*.dat')

    jes = [0.75, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.100, 1.25]
    marker = ['o', 'x', '^', 'v', 's', 'p']
    i = 0
    for file, AUC_val in zip(param_files, AUC_param):
        data = np.loadtxt(file)
        AUC = np.loadtxt(AUC_val)
    	plt.plot(data[:,0], data[:,1],
                    marker[i], 
    				alpha=0.2, 
                    markevery=10, 
                    label='jes$_p$=%s (AUC=%0.3f)' %(jes[i+2], AUC),  
    				rasterized=True)
        i=i+1
    i=0
    for file, AUC_val in zip(fixed_files, AUC_fixed):
        data = np.loadtxt(file)
        AUC = np.loadtxt(AUC_val)
    	plt.plot(data[:,0], data[:,1],
                    '-', 
    				alpha=1, 
                    markevery=100, 
                    label='jes$_f$=%s (AUC=%0.3f)' %(jes[i], AUC),  
    				rasterized=True)
        i=i+1
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/param_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/param_ROC_plot.png')	
    plt.clf()

def param_plot(): 
    print 'Entering fixed_plot'
    fixed_files = [#np.loadtxt('data/plot_data/fixed_500.dat'),
    			np.loadtxt('data/plot_data/fixed_750.dat'),
    			np.loadtxt('data/plot_data/fixed_1000.dat'),
    			np.loadtxt('data/plot_data/fixed_1250.dat'),
    			#np.loadtxt('data/plot_data/fixed_1500.dat')
    			]
    param_files = [np.loadtxt('data/plot_data/param_alpha_500.dat'),
    			np.loadtxt('data/plot_data/param_alpha_750.dat'),
    			np.loadtxt('data/plot_data/param_alpha_1000.dat'),
    			np.loadtxt('data/plot_data/param_alpha_1250.dat'),
    			np.loadtxt('data/plot_data/param_alpha_1500.dat')
    			]
    mx = [500, 750, 1000, 1250, 1500]
    fixed_colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    param_colors = ['DarkBlue', 'DarkGreen', 'DarkRed', 'DarkCyan', 'DarkMagenta']
    for i in range(len(fixed_files)):
    	plt.plot(fixed_files[i][:,0], fixed_files[i][:,1], 
    				'.', color=fixed_colors[i+1], alpha=0.5, markevery = 1, 
    				label='$\mu=$%s' %mx[i+1], rasterized=True)
    for i in range(len(param_files)):
    	plt.plot(param_files[i][:,0], param_files[i][:,1], 
    				'o', color=param_colors[i], alpha=0.5, markevery = 1, 
    				label='$\mu=$%s' %mx[i], rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('m$_{WWbb}$ [GeV]')
    plt.xlim([250,3000])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/paramTraining_complete.pdf', dpi=400)
    plt.savefig('plots/images/paramTraining_complete.png')
    plt.clf()


if __name__ == '__main__':
    '''File Runners'''
    #file_runner('data/root_files/')
    #flat_bkg(10000,0,5000)
    #file_concatenater()
    
    ''' NN Training '''
    #mwwbb_fixed()
    #mwwbb_parameterized()
    #mwwbbParameterizedRunner()
    #fixVSparam()

    
    '''Plotters'''
    #plt_histogram()
    #fixed_plot()
    #fixed_ROC_plot()
    #param_plot()
    param_ROC_plot()
    #fixVSparam_plot()