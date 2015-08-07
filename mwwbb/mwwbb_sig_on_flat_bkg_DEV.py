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
    root_files = glob.iglob('data/root_files_lg/*.root')
    for data in root_files:
        file_generate(data)
    for data in root_files:
        file_generate(data)

def flat_bkg(bkgNum, low, high):
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

def root_export(root_file, tree, leaf):
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

def file_generate(root_file):
    signal = root_export(root_file,'xtt','mwwbb')
    mx = root_export(root_file,'xtt','mx')
    #target = root_export(root_file,'xtt','target')
    target = 1.000000
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

def file_concatenater():
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

def plt_histogram():
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


def mwwbb_fixed(iterations):
    #conc_files = glob.iglob('data/concatenated/ttbar_mx_*.dat')
    conc_files = ['data/concatenated/ttbar_mx_750.dat',
                    'data/concatenated/ttbar_mx_1000.dat',
                    'data/concatenated/ttbar_mx_1250.dat']
    plt_marker =['g+', 'r+', 'c+']
    counter = 0
    for dat_file in conc_files:
        data = np.loadtxt(dat_file)
        traindata = data[:, 0:2]
        targetdata = data[:, 2]
        print 'Working on mu=%s' %data[0,1]
        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network',
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    #n_stable=1,
                    #f_stable=100,
                    n_iter=iterations,
                    #learning_momentum=0.1,
                    batch_size=10,
                    learning_rule="nesterov",
                    #valid_size=0.05,
                    verbose=True,
                    #debug=True
                    ))])

        nn.fit(traindata, targetdata)

        fit_score = nn.score(traindata, targetdata)
        print 'score = %s' %fit_score
        outputs = nn.predict(traindata)

        output_reshape = outputs.reshape((1,len(outputs)))
        actual = targetdata
        predictions = output_reshape[0]
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        roc_auc = auc(fpr, tpr)

        fig1 = plt.figure(1)
        plt.plot(traindata[:, 0], outputs, plt_marker[counter], alpha=1, label='$\mu=$%0.0f GeV' %data[0,1], rasterized=True)
        plt.ylabel('NN output')
        plt.xlabel('m$_{WWbb}$ [GeV]')
        plt.xlim([250, 3000])
        plt.ylim([-0.1, 1.1])
        plt.legend(loc='lower right')
        plt.grid(True)
        #plt.suptitle('Theano NN fixed training for m$_{WWbb}$ input', fontsize=14, fontweight='bold')

        fig2 = plt.figure(2)
        ROC_plot_fixed(data[0,1], fpr, tpr, roc_auc, plt_marker[counter])
        counter = counter+1
    fig1.savefig('plots/fixedTraining.pdf', dpi=400)
    fig2.savefig('plots/ROC_fixed.pdf', dpi=400)
    fig1.savefig('plots/images/fixedTraining.png')
    fig2.savefig('plots/images/ROC_fixed.png')
    #plt.clf()

    pickle.dump(nn, open('data/pickle/fixed.pkl', 'wb'))


def mwwbb_parameterized(iterations):
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

    mx_list = [750, 1000, 1250]

    for i in range(3):
        traindata      = train_list[i]
        targetdata     = target_list[i]

        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network',
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    n_iter=iterations,
                    #n_stable=1,
                    #f_stable=0.001,
                    #learning_momentum=0.1,
                    batch_size=10,
                    learning_rule="nesterov",
                    #valid_size=0.05,
                    verbose=True,
                    #debug=True
                    ))])
        print nn

        nn.fit(traindata, targetdata)

        fit_score = nn.score(traindata, targetdata)
        print 'score = %s' %fit_score

        outputs = nn.predict(traindata)

        # Plot settings
        plt.plot(traindata[:, 0], outputs, 'o', alpha=1, rasterized=True)
        plt.ylabel('NN output')
        plt.xlabel('m$_{WWbb}$ [GeV]')
        plt.xlim([250, 3000])
        plt.ylim([-0.1, 1.1])
        plt.grid(True)
        #plt.suptitle('Theano NN fixed training for m$_{WWbb}$ input',
            #fontsize=14, fontweight='bold')
        #plt.savefig('plots/paramTraining.pdf', dpi=400)
        #plt.savefig('plots/images/paramTraining.png')
        plt.clf()

        pickle.dump(nn, open('data/pickle/param_%s.pkl' %mx_list[i], 'wb'))

def scikitlearnFunc(x, alpha, mx):
    nn = pickle.load(open('data/pickle/param_%s.pkl' %mx,'rb'))
    traindata = np.array((x, alpha), ndmin=2)
    outputs   = nn.predict(traindata)

    #print 'x,alpha,output =', x, alpha, outputs[0]
    #plt.plot(x, outputs, 'ro', alpha=0.5)
    return outputs[[0]]

def mwwbbParameterizedRunner():
    plt_marker=['bo', 'go', 'r.', 'co', 'mo']
    plt_marker = ['o', 'o', 'o', 'o', 'o']
    plt_color = ['blue', 'DarkGreen', 'DarkRed', 'DarkCyan', 'magenta']
    alpha = [500, 750, 1000, 1250, 1500]
    size = 20000

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
        predictions = []
        input = input_list[a]
        for x in range(0,2*size, 1):
            outputs = scikitlearnFunc(input[x]/1., alpha[a], 1000)
            predictions.append(outputs[0])
            fig1 = plt.figure(1)
            plt.plot(input[x]/1., outputs[0], marker=plt_marker[a], color=plt_color[a], alpha=1, rasterized=True)
            print "%s - Percent Complete: %s" %(alpha[a], (x/(2.0*size))*100.0)
        actual = actual_list[a]
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        roc_auc = auc(fpr, tpr)
        fig2 = plt.figure(2)
        ROC_plot_param(alpha[a], fpr, tpr, roc_auc, plt_marker[a], plt_color[a])
    for i in range(len(alpha)):
        fig1 = plt.figure(1)
        plt.plot(-4,0, plt_marker[i], alpha=1, label="$\mu=$%s GeV" %alpha[i], rasterized=True)
    #plt.legend(bbox_to_anchor=(0.6, .4), loc=2, borderaxespad=0)
    fig1 = plt.figure(1)
    plt.legend(loc='lower right')
    plt.ylabel('NN output')
    plt.xlabel('m$_{WWbb}$ [GeV]')
    plt.xlim([250, 3000])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    #plt.suptitle('Theano NN regression output for parameterized m$_{WWbb}$ input',
               #fontsize=12, fontweight='bold')

    fig1.savefig('plots/paramTraining_complete.pdf', dpi=400)
    fig1.savefig('plots/images/paramTraining_complete.png')
    fig2 = plt.figure(2)
    fig2.savefig('plots/ROC_parameterized.pdf', dpi=400)
    fig2.savefig('plots/images/ROC_parameterized.png')

def ROC_plot_fixed(mx, fpr, tpr, roc_auc, plt_marker):
    print "Plotting ROC curve for mx=%s" %mx
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, plt_marker, label='AUC (mx=%s) = %0.2f' %(mx, roc_auc), 
                alpha=1, rasterized=True)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.grid(True)

def ROC_plot_param(mx, fpr, tpr, roc_auc, plt_marker, plt_color):
    print "Plotting ROC curve for mx=%s" %mx
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, marker=plt_marker, color=plt_color, label='AUC (mx=%s) = %0.2f' %(mx, roc_auc), alpha=1, rasterized=True)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.grid(True)

def fixVSparam():
    plt_marker = ['o', 'o', 'o']
    plt_color = ['DarkGreen', 'DarkRed', 'DarkCyan']
    alpha = [750, 1000, 1250]
    size = 20000

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
        roc_auc = auc(fpr, tpr)
        fig2 = plt.figure(2)
        ROC_plot_param(alpha[a], fpr, tpr, roc_auc, plt_marker[a], plt_color[a])
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig2.savefig('plots/fixVSparam.pdf', dpi=400)
    fig2.savefig('plots/images/fixVSparam.png')
    fig1.clf()
    fig2.clf()
    plt.clf()

if __name__ == '__main__':
    #file_runner()
    #flat_bkg(10000,0,5000)
    #plt_histogram()
    #file_concatenater()
    mwwbb_parameterized(100)

    mwwbb_fixed(100)
    fixVSparam()

    mwwbb_fixed(100)
    mwwbbParameterizedRunner()