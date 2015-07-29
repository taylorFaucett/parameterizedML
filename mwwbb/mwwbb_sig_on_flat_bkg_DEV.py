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
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer



# Plot marks (color circles - i.e. bo = blue circle, go = green circle)
plt_marker=['bo', 'go', 'ro', 'co', 'mo', 'yo', 'bo', 'wo']

mx_values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

sig_files = ['xttbar_14tev_mx400_jes1.0.root',
                'xttbar_14tev_mx500_jes1.0.root',
                'xttbar_14tev_mx600_jes1.0.root',
                'xttbar_14tev_mx700_jes1.0.root',
                'xttbar_14tev_mx800_jes1.0.root',
                'xttbar_14tev_mx900_jes1.0.root',
                'xttbar_14tev_mx1000_jes1.0.root',
                'xttbar_14tev_mx1100_jes1.0.root',
                'xttbar_14tev_mx1200_jes1.0.root',
                'xttbar_14tev_mx1300_jes1.0.root',
                'xttbar_14tev_mx1400_jes1.0.root',
                'xttbar_14tev_mx1500_jes1.0.root']

bkg_files = ['smttbar_14tev_mx400_jes1.0.root',
                'smttbar_14tev_mx500_jes1.0.root',
                'smttbar_14tev_mx600_jes1.0.root',
                'smttbar_14tev_mx700_jes1.0.root',
                'smttbar_14tev_mx800_jes1.0.root',
                'smttbar_14tev_mx900_jes1.0.root',
                'smttbar_14tev_mx1000_jes1.0.root',
                'smttbar_14tev_mx1100_jes1.0.root',
                'smttbar_14tev_mx1200_jes1.0.root',
                'smttbar_14tev_mx1300_jes1.0.root',
                'smttbar_14tev_mx1400_jes1.0.root',
                'smttbar_14tev_mx1500_jes1.0.root'
                ]

mx_text    = ['400', '500', '600', '700',
                '800', '900', '1000', '1100',
                '1200', '1300', '1400', '1500']

sig_dat = ['data/root_export/sig_mx_400.dat',
            'data/root_export/sig_mx_500.dat',
            'data/root_export/sig_mx_600.dat',
            'data/root_export/sig_mx_700.dat',
            'data/root_export/sig_mx_800.dat',
            'data/root_export/sig_mx_900.dat',
            'data/root_export/sig_mx_1000.dat',
            'data/root_export/sig_mx_1100.dat',
            'data/root_export/sig_mx_1200.dat',
            'data/root_export/sig_mx_1300.dat',
            'data/root_export/sig_mx_1400.dat',
            'data/root_export/sig_mx_1500.dat'
            ]

bkg_dat = ['data/root_export/bkg_mx_400.dat',
            'data/root_export/bkg_mx_500.dat',
            'data/root_export/bkg_mx_600.dat',
            'data/root_export/bkg_mx_700.dat',
            'data/root_export/bkg_mx_800.dat',
            'data/root_export/bkg_mx_900.dat',
            'data/root_export/bkg_mx_1000.dat',
            'data/root_export/bkg_mx_1100.dat',
            'data/root_export/bkg_mx_1200.dat',
            'data/root_export/bkg_mx_1300.dat',
            'data/root_export/bkg_mx_1400.dat',
            'data/root_export/bkg_mx_1500.dat']

flat_dat = ['data/flat_bkg/flat_bkg_mx_400.dat',
            'data/flat_bkg/flat_bkg_mx_500.dat',
            'data/flat_bkg/flat_bkg_mx_600.dat',
            'data/flat_bkg/flat_bkg_mx_700.dat',
            'data/flat_bkg/flat_bkg_mx_800.dat',
            'data/flat_bkg/flat_bkg_mx_900.dat',
            'data/flat_bkg/flat_bkg_mx_1000.dat',
            'data/flat_bkg/flat_bkg_mx_1100.dat',
            'data/flat_bkg/flat_bkg_mx_1200.dat',
            'data/flat_bkg/flat_bkg_mx_1300.dat',
            'data/flat_bkg/flat_bkg_mx_1400.dat',
            'data/flat_bkg/flat_bkg_mx_1500.dat']

conc_dat = ['data/concatenated/ttbar_mx_400.dat',
            'data/concatenated/ttbar_mx_500.dat',
            'data/concatenated/ttbar_mx_600.dat',
            'data/concatenated/ttbar_mx_700.dat',
            'data/concatenated/ttbar_mx_800.dat',
            'data/concatenated/ttbar_mx_900.dat',
            'data/concatenated/ttbar_mx_1000.dat',
            'data/concatenated/ttbar_mx_1100.dat',
            'data/concatenated/ttbar_mx_1200.dat',
            'data/concatenated/ttbar_mx_1300.dat',
            'data/concatenated/ttbar_mx_1400.dat',
            'data/concatenated/ttbar_mx_1500.dat']


def file_runner():
    directory = 'data/root_files/'
    for i in range(len(sig_files)):
        file_generate(directory+sig_files[i])
    for i in range(len(bkg_files)):
        file_generate(directory+bkg_files[i])

def flat_bkg(bkgNum, low, high):
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

        np.savetxt('data/flat_bkg/flat_bkg_mx_%s.dat' %mx_values[j], bkg_data, fmt='%f')

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

def file_generate(File):
    signal = root_export(File,'xtt','mwwbb')
    mx = root_export(File,'xtt','mx')
    target = root_export(File,'xtt','target')

    size = len(signal)
    data = np.zeros((size, 3))
    if target[0]<0.5:
        label = 'bkg'
    elif target[0]>0.5:
        label = 'sig'
    for i in range(size):
        data[i, 0] = signal[i]
        data[i, 1] = mx[i]
        data[i, 2] = target[i]
    np.savetxt('data/root_export/%s_mx_%0.0f.dat' %(label, mx[0]), data, fmt='%f')

def file_concatenater():
    for i in range(len(sig_dat)):
        sig = np.loadtxt(sig_dat[i])
        bkg = np.loadtxt(bkg_dat[i])
        data_complete = np.concatenate((sig, bkg),axis=0)
        np.savetxt('data/concatenated/ttbar_mx_%s.dat' %mx_text[i], data_complete, fmt='%f')
    for i in range(len(sig_dat)):
        sig = np.loadtxt(sig_dat[i])
        flat = np.loadtxt(flat_dat[i])
        data_complete = np.concatenate((sig, flat),axis=0)
        np.savetxt('data/concatenated/flat_mx_%s.dat' %mx_text[i], data_complete, fmt='%f')

def plt_histogram():
    bin_size   = 50
    for i in range(len(sig_dat)):
        sig = np.loadtxt(sig_dat[i])
        bkg = np.loadtxt(bkg_dat[i])
        n, bins, patches = plt.hist([sig[:,0], bkg[:,0] ],
                            bins=range(0,4000, bin_size), histtype='stepfilled',
                            alpha=0.5, label=['Signal', 'Background'])
        plt.setp(patches)
        plt.title('m$_{WWbb} =$ %s GeV/c$^2$' %mx_values[i])
        plt.ylabel('Number of events$/%s$ GeV/c$^2$' %bin_size)
        plt.xlabel('m$_{WWbb}$ [GeV/c$^2$]')
        plt.grid(True)
        plt.legend()
        plt.xlim([0, 4000])
        plt.ylim([0, 350])
        plt.savefig('plots/histograms/histo_%s.pdf'%mx_text[i])
        plt.savefig('plots/images/histograms/histo_%s.png'%mx_text[i])
        plt.clf()


def mwwbb_fixed(iterations):
    for i in range(len(conc_dat)):
        data = np.loadtxt(conc_dat[i])
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
                    #verbose=True,
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
        plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5, label='$\mu=$%s' % mx_values[i])
        plt.ylabel('NN_output( m$_{WWbb}$ )')
        plt.xlabel('m$_{WWbb}$ [GeV/c$^2$]')
        plt.xlim([0, 4000])
        plt.ylim([-0.1, 1.1])
        plt.legend()
        plt.grid(True)
        plt.suptitle('Theano NN fixed training for m$_{WWbb}$ input', fontsize=14, fontweight='bold')

        fig2 = plt.figure(2)
        ROC_plot(mx_values[i], fpr, tpr, roc_auc)
    fig1.savefig('plots/fixedTraining.pdf')
    fig2.savefig('plots/ROC_fixed.pdf')
    fig1.savefig('plots/images/fixedTraining.png')
    fig2.savefig('plots/images/ROC_fixed.png')
    plt.clf()

    pickle.dump(nn, open('data/pickle/fixed.pkl', 'wb'))


def mwwbb_parameterized(iterations):
    mwwbb_complete = np.concatenate((
                        #np.loadtxt(conc_dat[0]),
                        np.loadtxt(conc_dat[1]),
                        #np.loadtxt(conc_dat[2]),
                        #np.loadtxt(conc_dat[3]),
                        #np.loadtxt(conc_dat[4]),
                        #np.loadtxt(conc_dat[5]),
                        np.loadtxt(conc_dat[6]),
                        #np.loadtxt(conc_dat[7]),
                        #np.loadtxt(conc_dat[8]),
                        #np.loadtxt(conc_dat[9]),
                        #np.loadtxt(conc_dat[10]),
                        np.loadtxt(conc_dat[11])),
                        axis=0)

    traindata      = mwwbb_complete[:,0:2]
    targetdata     = mwwbb_complete[:,2]

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
                #verbose=True,
                #debug=True
                ))])
    print nn

    nn.fit(traindata, targetdata)

    fit_score = nn.score(traindata, targetdata)
    print 'score = %s' %fit_score

    outputs = nn.predict(traindata)

    # Plot settings
    plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5)
    plt.ylabel('NN_output( m$_{WWbb}$ )')
    plt.xlabel('m$_{WWbb}$ [GeV/c$^2$]')
    plt.xlim([0, 4000])
    plt.ylim([-0.1, 1.1])
    #plt.axhline(y=0, color = 'black', linewidth = 2, alpha=0.75)
    #plt.axhline(y=1, color = 'black', linewidth = 2, alpha=0.75)
    plt.grid(True)
    plt.suptitle('Theano NN fixed training for m$_{WWbb}$ input',
        fontsize=14, fontweight='bold')
    plt.savefig('plots/paramTraining.pdf')
    plt.savefig('plots/images/paramTraining.png')
    #plt.show()
    plt.clf()

    pickle.dump(nn, open('data/pickle/param.pkl', 'wb'))

def scikitlearnFunc(x, alpha):
    nn = pickle.load(open('data/pickle/param.pkl','rb'))
    #print "inouttest input was", x
    traindata = np.array((x, alpha), ndmin=2)
    outputs   = nn.predict(traindata)
    #print outputs

    #print 'x,alpha,output =', x, alpha, outputs[0]
    #plt.plot(x, outputs, 'ro', alpha=0.5)
    #plt.show()
    return outputs[[0]]

def mwwbbParameterizedRunner():
    alpha = [500, 750, 1000, 1250, 1500]
    print "Running on %s alpha values: %s" %(len(alpha), alpha)
    for a in range(len(alpha)):
        print 'working on alpha=%s' %alpha[a]
        for x in range(0,4000, 10):
            outputs = scikitlearnFunc(x/1., alpha[a])
            plt.plot(x/1., outputs[0], plt_marker[a], alpha=0.5)
    for i in range(len(alpha)):
        plt.plot(-4,0, plt_marker[i], alpha=0.5, label="$\mu=$%s GeV/c$^2$" %alpha[i])
    plt.legend(bbox_to_anchor=(0.6, .4), loc=2, borderaxespad=0)
    plt.ylabel('NN_output( m$_{WWbb}$ )')
    plt.xlabel('m$_{WWbb}$ [GeV/c$^2$]')
    plt.xlim([0, 4000])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.suptitle('Theano NN regression output for parameterized m$_{WWbb}$ input',
               fontsize=12, fontweight='bold')

    plt.savefig('plots/paramTraining_complete.pdf')
    plt.savefig('plots/images/paramTraining_complete.png')
    #plt.show()

def ROC_plot(mx, fpr, tpr, roc_auc):
    print "Plotting ROC curve for mx=%s" %mx
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC (mx=%s) = %0.2f' %(mx, roc_auc), linewidth=1.5)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    #plt.savefig('plots/ROC/ROC_fixed_mx_%s.pdf' %mx)
    #plt.savefig('plots/ROC/images/ROC_fixed_mx_%s.png' %mx)
    #plt.clf()
    #plt.show()


if __name__ == '__main__':
    #file_runner()
    #flat_bkg(1000,0,5000)
    #plt_histogram()
    #file_concatenater()
    mwwbb_fixed(100)
    #mwwbb_parameterized(1000)
    #mwwbbParameterizedRunner()
