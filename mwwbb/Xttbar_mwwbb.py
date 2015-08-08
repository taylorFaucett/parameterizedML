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
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer

def file_runner():
    print 'Entering file_runner'
    sig_files = glob.iglob('data/root_files/xttbar_*.root')
    bkg_files = glob.iglob('data/root_files/smttbar.root')
    for data in sig_files:
    	file_generate(data, 1.000000)
    for data in bkg_files:
        file_generate(data, 0.000000)

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

def file_generate(root_file, target):
    print 'Entering file_generate'
    print 'Generating data using values from: %s' %root_file
    signal = root_export(root_file,'xtt','mwwbb')
    mx = root_export(root_file,'xtt','mx')
    #target = root_export(root_file,'xtt','target')
    size = len(signal)
    data = np.zeros((size, 3))
    if target<0.5:
        label = 'bkg'
    elif target>0.5:
        label = 'sig'
    for i in range(size):
        data[i, 0] = signal[i]
        data[i, 1] = mx[i]
        data[i, 2] = target
    np.savetxt('data/root_export/%s_mx_%0.0f.dat' %(label, mx[0]), data, fmt='%f')

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
    sig_dat = glob.iglob('data/root_export/sig_mx_*.dat')
    bkg_dat = glob.iglob('data/root_export/bkg_mx_*.dat')
    flt_dat = glob.iglob('data/flat_bkg/*.dat')
    for signal, background, flat in zip(sig_dat, bkg_dat, flt_dat):
        sig = np.loadtxt(signal)
        bkg = np.loadtxt(background)
        flt = np.loadtxt(flat)
        data_complete = np.concatenate((sig, bkg), axis=0)
        np.savetxt('data/concatenated/ttbar_mx_%0.0f.dat' %sig[0,1], data_complete, fmt='%f')
        data_complete = np.concatenate((sig, flt), axis=0)
        np.savetxt('data/concatenated/flat_mx_%0.0f.dat' %sig[0,1], data_complete, fmt='%f')


''' NN Training '''

def mwwbb_fixed():
    print 'Entering mwwbb_fixed'
    files = ['data/concatenated/ttbar_mx_500.dat',
    			'data/concatenated/ttbar_mx_750.dat',
    			'data/concatenated/ttbar_mx_1000.dat',
    			'data/concatenated/ttbar_mx_1250.dat',
    			'data/concatenated/ttbar_mx_1500.dat']
    mx = [500, 750, 1000, 1250, 1500]
    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network',
            Regressor(
                layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                learning_rate=0.01,
                #n_stable=1,
                #f_stable=100,
                n_iter=20,
                #learning_momentum=0.1,
                batch_size=10,
                learning_rule="nesterov",
                #valid_size=0.05,
                #verbose=True,
                #debug=True
                ))])
    print nn
    for i in range(len(files)):
    	print 'Processing fixed training on mu=%s' %mx[i]
    	data = np.loadtxt(files[i])
    	traindata = data[:,0:2]
    	targetdata = data[:,2]
    	nn.fit(traindata, targetdata)

    	fit_score = nn.score(traindata, targetdata)
    	print 'score = %s' %fit_score
    	outputs = nn.predict(traindata)
    	outputs = outputs.reshape((1,len(outputs)))
    	fixed_plot = np.vstack((traindata[:,0], outputs)).T
    	np.savetxt('data/plot_data/fixed_%s.dat' %mx[i], fixed_plot, fmt='%f')

    	actual = targetdata
    	predictions = outputs[0]
    	fpr, tpr, thresholds = roc_curve(actual, predictions)
    	ROC_plot = np.vstack((fpr, tpr)).T
    	roc_auc = [auc(fpr, tpr)]
    	np.savetxt('data/plot_data/fixed_ROC_%s.dat' %mx[i], ROC_plot, fmt='%f')
    	np.savetxt('data/plot_data/fixed_ROC_AUC_%s.dat' %mx[i], roc_auc)

    	pickle.dump(nn, open('data/pickle/fixed_%s.pkl' %mx[i], 'wb'))


def mwwbb_parameterized():
    print 'Entering mwwbb_parameterized'
    mwwbb_complete750 = np.concatenate((
                        np.loadtxt('data/concatenated/ttbar_mx_500.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1250.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1500.dat')),
                        axis=0)

    mwwbb_complete1000 = np.concatenate((
                    np.loadtxt('data/concatenated/ttbar_mx_500.dat'),
                    np.loadtxt('data/concatenated/ttbar_mx_750.dat'),
                    np.loadtxt('data/concatenated/ttbar_mx_1250.dat'),
                    np.loadtxt('data/concatenated/ttbar_mx_1500.dat')),
                    axis=0)

    mwwbb_complete1250 = np.concatenate((
                np.loadtxt('data/concatenated/ttbar_mx_500.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_750.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1500.dat')),
                axis=0)

    train_list = [mwwbb_complete750[:,0:2],
                    mwwbb_complete1000[:,0:2],
                    mwwbb_complete1250[:,0:2]
                    ]

    target_list = [mwwbb_complete750[:,2],
                    mwwbb_complete1000[:,2],
                    mwwbb_complete1250[:,2]
                    ]

    mx = [750, 1000, 1250]

    for i in range(len(mx)):
    	print 'Parameterized training on all signals except for mu=%s' %mx[i]
        traindata      = train_list[i]
        targetdata     = target_list[i]

        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network',
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    n_iter=20,
                    #n_stable=1,
                    #f_stable=0.001,
                    #learning_momentum=0.1,
                    batch_size=5,
                    learning_rule="nesterov",
                    #valid_size=0.05,
                    #verbose=True,
                    #debug=True
                    ))])
        print nn

        nn.fit(traindata, targetdata)
        fit_score = nn.score(traindata, targetdata)
        print 'score = %s' %fit_score
        outputs = nn.predict(traindata)
        outputs = outputs.reshape((1, len(outputs)))
        param_plot = np.vstack((traindata[:,0], outputs)).T
        np.savetxt('data/plot_data/param_%s.dat' %mx[i], param_plot, fmt='%f')
        pickle.dump(nn, open('data/pickle/param_%s.pkl' %mx[i], 'wb'))

        

def scikitlearnFunc(x, alpha, mx):
    nn = pickle.load(open('data/pickle/param_%s.pkl' %mx,'rb'))
    traindata = np.array((x, alpha), ndmin=2)
    outputs   = nn.predict(traindata)

    #print 'x,alpha,output =', x, alpha, outputs[0]
    #plt.plot(x, outputs, 'ro', alpha=0.5)
    return outputs[[0]]

def mwwbbParameterizedRunner():
    print 'Entering mwwbbParameterizedRunner'
    alpha = [500, 750, 1000, 1250, 1500]
    size = 2000

    mx_500_raw = np.loadtxt('data/concatenated/ttbar_mx_500.dat')
    mx_750_raw = np.loadtxt('data/concatenated/ttbar_mx_750.dat')
    mx_1000_raw = np.loadtxt('data/concatenated/ttbar_mx_1000.dat')
    mx_1250_raw = np.loadtxt('data/concatenated/ttbar_mx_1250.dat')
    mx_1500_raw = np.loadtxt('data/concatenated/ttbar_mx_1500.dat')

    mx_500_sig = mx_500_raw[:size, :]
    mx_500_bkg = mx_500_raw[-size:, :]

    mx_750_sig = mx_750_raw[:size, :]
    mx_750_bkg = mx_750_raw[-size:, :]

    mx_1000_sig = mx_1000_raw[:size, :]
    mx_1000_bkg = mx_1000_raw[-size:, :]

    mx_1250_sig = mx_1250_raw[:size, :]
    mx_1250_bkg = mx_1250_raw[-size:, :]

    mx_1500_sig = mx_1500_raw[:size, :]
    mx_1500_bkg = mx_1500_raw[-size:, :]

    mx_500 = np.concatenate((mx_500_sig, mx_500_bkg), axis=0)
    mx_750 = np.concatenate((mx_750_sig, mx_750_bkg), axis=0)
    mx_1000 = np.concatenate((mx_1000_sig, mx_1000_bkg), axis=0)
    mx_1250 = np.concatenate((mx_1250_sig, mx_1250_bkg), axis=0)
    mx_1500 = np.concatenate((mx_1500_sig, mx_1500_bkg), axis=0)

    input_list = [mx_500[:,0],
                    mx_750[:,0],
                    mx_1000[:,0],
                    mx_1250[:,0],
                    mx_1500[:,0]]

    actual_list = [mx_500[:,2],
                    mx_750[:,2],
                    mx_1000[:,2],
                    mx_1250[:,2],
                    mx_1500[:,2]]


    mwwbb_complete = np.concatenate((
                        np.loadtxt('data/concatenated/ttbar_mx_500.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1500.dat')),
                        axis=0)

    traindata      = mwwbb_complete[:,0:2]
    targetdata     = mwwbb_complete[:,2]

    print "Running on %s alpha values: %s" %(len(alpha), alpha)

    for a in range(len(alpha)):
        print 'working on alpha=%s' %alpha[a]
        inputs = []
        predictions = []
        input = input_list[a]
        for x in range(0,2*size, 1):
            outputs = scikitlearnFunc(input[x]/1., alpha[a], 1000)
            inputs.append(input[x]/1.)
            predictions.append(outputs[0][0])
        data = np.vstack((inputs, predictions)).T
        np.savetxt('data/plot_data/param_alpha_%s.dat' %alpha[a], data, fmt='%f')
        actual = actual_list[a]
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        roc_auc = [auc(fpr, tpr)]

        roc_data = np.vstack((fpr, tpr)).T
        np.savetxt('data/plot_data/param_alpha_ROC_%s.dat' %alpha[a], roc_data, fmt='%f')
        np.savetxt('data/plot_data/param_alpha_ROC_AUC_%s.dat' %alpha[a], roc_auc)

def fixVSparam():
    print 'Entering fixVSparam'
    alpha = [750, 1000, 1250]
    size = 2000

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
    bin_size   = 50
    #sig_dat = glob.iglob('data/root_export/sig_mx_*.dat')
    #bkg_dat = glob.iglob('data/root_export/bkg_mx_*.dat')
    plt_color=['black', # background
        'blue', # mu=500
        'green', # mu=750
        'red', # mu=1000
        'cyan', # mu=1250
        'magenta' # mu=1500
        ]
    data_list = ['data/root_export/bkg_mx_500.dat',
                'data/root_export/sig_mx_500.dat',
                'data/root_export/sig_mx_750.dat',
                'data/root_export/sig_mx_1000.dat',
                'data/root_export/sig_mx_1250.dat',
                'data/root_export/sig_mx_1500.dat']

    histtype_list = ['stepfilled',
                        'step',
                        'step',
                        'step',
                        'step',
                        'step']

    label_count = 0
    for i in range(len(data_list)):
        data = np.loadtxt(data_list[i])
        label = ['Background', 
                    '$\mu=500\,$ GeV',
                    '$\mu=750\,$ GeV', 
                    '$\mu=1000\,$ GeV', 
                    '$\mu=1250\,$ GeV',
                    '$\mu=1500\,$ GeV']
        n, bins, patches = plt.hist([data[:,0] ],
                            bins=range(0,3000, bin_size), normed=True,
                            histtype=histtype_list[i], alpha=0.75, linewidth=2, 
                            label=[label[label_count]], color=plt_color[i],
                            rasterized=True)
        label_count = label_count + 1
        plt.setp(patches)
    #plt.title('m$_{WWbb} =$ %s GeV' %sig[0,1])
    plt.ylabel('Fraction of events$/%0.0f$ GeV' %bin_size)
    plt.xlabel('m$_{WWbb}$ [GeV]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlim([0, 3000])
    #plt.ylim([0, 35000])
    plt.savefig('plots/mWWbb_histogram.pdf', dpi=400)
    plt.savefig('plots/images/mWWbb_histogram.png')
    plt.clf()



def fixed_plot(): 
    print 'Entering fixed_plot'
    files = [#np.loadtxt('data/plot_data/fixed_500.dat'),
    			np.loadtxt('data/plot_data/fixed_750.dat'),
    			np.loadtxt('data/plot_data/fixed_1000.dat'),
    			np.loadtxt('data/plot_data/fixed_1250.dat'),
    			#np.loadtxt('data/plot_data/fixed_1500.dat')
    			]
    mx = [500, 750, 1000, 1250, 1500]
    plt_marker = ['b.', 'g.', 'r.', 'c.', 'm.']
    for i in range(len(files)):
    	plt.plot(files[i][:,0], files[i][:,1], 
    				plt_marker[i+1], alpha=0.5, markevery = 50, 
    				label='$\mu=$%s' %mx[i], rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('m$_{WWbb}$ [GeV]')
    plt.xlim([250,3000])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixedTraining.pdf', dpi=400)
    plt.savefig('plots/images/fixedTraining.png')
    plt.clf()

def fixed_ROC_plot():
    print "Entering fixed_ROC_plot"
    files = [np.loadtxt('data/plot_data/fixed_ROC_500.dat'),
    			np.loadtxt('data/plot_data/fixed_ROC_750.dat'),
    			np.loadtxt('data/plot_data/fixed_ROC_1000.dat'),
    			np.loadtxt('data/plot_data/fixed_ROC_1250.dat'),
    			np.loadtxt('data/plot_data/fixed_ROC_1500.dat')]
    AUC = [np.loadtxt('data/plot_data/fixed_ROC_AUC_500.dat'),
    		np.loadtxt('data/plot_data/fixed_ROC_AUC_750.dat'),
    		np.loadtxt('data/plot_data/fixed_ROC_AUC_1000.dat'),
    		np.loadtxt('data/plot_data/fixed_ROC_AUC_1250.dat'),
    		np.loadtxt('data/plot_data/fixed_ROC_AUC_1500.dat')]
    mx = [500, 750, 1000, 1250, 1500]
    plt_color = ['blue', 'green', 'red', 'cyan', 'magenta']
    plt_marker = ['.', '.', '.', '.', '.']
    for i in range(len(files)):
    	plt.plot(files[i][:,0], files[i][:,1],
    				marker=plt_marker[i], color=plt_color[i], alpha=0.5, 
    				markevery = 1000, label='$\mu=$%s (AUC=%0.2f)' %(mx[i], AUC[i]), rasterized=True)
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
    				alpha=0.5, markevery=1, 
    				label='$\mu=$%s (AUC=%0.2f)' %(mx[i], AUC_param[i]), rasterized=True)
    for i in range(len(fixed_files)):
    	plt.plot(fixed_files[i][:,0], fixed_files[i][:,1], fixed_markers[i], 
    				alpha=0.5, markevery=1000, 
    				label='$\mu=$%s (AUC=%0.2f)' %(mx[i], AUC_fixed[i]),  rasterized=True)
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
    param_files = [#np.loadtxt('data/plot_data/param_alpha_ROC_500.dat'),
    			np.loadtxt('data/plot_data/param_alpha_ROC_750.dat'),
    			np.loadtxt('data/plot_data/param_alpha_ROC_1000.dat'),
    			np.loadtxt('data/plot_data/param_alpha_ROC_1250.dat'),
    			#np.loadtxt('data/plot_data/param_alpha_ROC_1500.dat')
    			]
    fixed_files = [np.loadtxt('data/plot_data/fixed_ROC_500.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_750.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_1000.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_1250.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_1500.dat')
    				]
    AUC_param = [#np.loadtxt('data/plot_data/param_alpha_ROC_AUC_500.dat'),
    				np.loadtxt('data/plot_data/param_alpha_ROC_AUC_750.dat'),
    				np.loadtxt('data/plot_data/param_alpha_ROC_AUC_1000.dat'),
    				np.loadtxt('data/plot_data/param_alpha_ROC_AUC_1250.dat'),
    				#np.loadtxt('data/plot_data/param_alpha_ROC_AUC_1500.dat')
    				]

    AUC_fixed = [np.loadtxt('data/plot_data/fixed_ROC_AUC_500.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_750.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_1000.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_1250.dat'),
    				np.loadtxt('data/plot_data/fixed_ROC_AUC_1500.dat')
    				]
    param_markers = ['bo', 'go', 'ro', 'co', 'mo']
    fixed_markers = ['b.', 'g.', 'r.', 'c.', 'm.']
    mx = [500,750, 1000, 1250,1500]

    for i in range(len(param_files)):
    	plt.plot(param_files[i][:,0], param_files[i][:,1], param_markers[i+1], 
    				alpha=0.5, markevery=100, label='$\mu=$%s (AUC=%0.2f)' %(mx[i], AUC_param[i]),  
    				rasterized=True)
    for i in range(len(fixed_files)):
    	plt.plot(fixed_files[i][:,0], fixed_files[i][:,1], fixed_markers[i], 
    				alpha=0.5, markevery=100, label='$\mu=$%s (AUC=%0.2f)' %(mx[i], AUC_fixed[i]),  
    				rasterized=True)
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
    #file_runner()
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
    fixVSparam_plot()
    param_ROC_plot()
    #param_plot()
