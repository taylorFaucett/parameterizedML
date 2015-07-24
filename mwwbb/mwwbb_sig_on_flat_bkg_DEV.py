'''
author Taylor Faucett <tfaucett@uci.edu>

This script utilizes Theano/Pylearn2 and SKLearn-NeuralNetwork to create a fixed and parameterized
machine learning scheme. Data taken from an X->ttbar selection is imported as
signal and a uniform (i.e. flat) background is generated and appended. mwwbb_fixed uses a regression NN
to learn for n signals at fixed means (mx) which can map a 1D array to signal/background 
values of 1 or 0. trainParam trains for all n signals simultaneously and then trains for 
these signals with a parameter by a secondary input (alpha).
'''

import ROOT
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer

# Plot marks (color circles - i.e. bo = blue circle, go = green circle)
plt_marker=['bo', 'go', 'ro', 'co', 'mo', 'yo', 'bo', 'wo']

def bkg_merge():
    bkgNum = 2000
    mwwbb_400 = np.loadtxt('data/mwwbb_raw/mwwbb_400.dat')
    mwwbb_500 = np.loadtxt('data/mwwbb_raw/mwwbb_500.dat')
    mwwbb_600 = np.loadtxt('data/mwwbb_raw/mwwbb_600.dat')
    mwwbb_700 = np.loadtxt('data/mwwbb_raw/mwwbb_700.dat')
    mwwbb_800 = np.loadtxt('data/mwwbb_raw/mwwbb_800.dat')
    mwwbb_900 = np.loadtxt('data/mwwbb_raw/mwwbb_900.dat')
    mwwbb_1000 = np.loadtxt('data/mwwbb_raw/mwwbb_1000.dat')
    mwwbb_1100 = np.loadtxt('data/mwwbb_raw/mwwbb_1100.dat')
    mwwbb_1200 = np.loadtxt('data/mwwbb_raw/mwwbb_1200.dat')
    mwwbb_1300 = np.loadtxt('data/mwwbb_raw/mwwbb_1300.dat')
    mwwbb_1400 = np.loadtxt('data/mwwbb_raw/mwwbb_1400.dat')
    mwwbb_1500 = np.loadtxt('data/mwwbb_raw/mwwbb_1500.dat')

    mwwbb_raw = [mwwbb_400, mwwbb_500, mwwbb_600, mwwbb_700, mwwbb_800, mwwbb_900, mwwbb_1000, mwwbb_1100, mwwbb_1200, mwwbb_1300, mwwbb_1400, mwwbb_1500]
    mx_values = [400.000000, 500.000000, 600.000000, 700.000000, 800.000000, 900.000000, 1000.000000, 1100.000000, 1200.000000, 1300.000000, 1400.000000, 1500.000000]
    mx_text = ['400', '500', '600', '700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']

    w = ROOT.RooWorkspace('w')
    w.factory('Uniform::e(x[0,7000])')

    # Define variables
    x      = w.var('x')
    bkgpdf = w.pdf('e')

    # Fill traindata, testdata and testdata1
    print 'Generating background data and concatenting with signal data'
    bkg_values = bkgpdf.generate(ROOT.RooArgSet(x), bkgNum*12)
    for j in range(len(mwwbb_raw)):
        bkgdata = np.zeros((bkgNum, 3))
        for i in range(bkgNum):
            bkgdata[i, 0] = bkg_values.get(i).getRealValue('x')
            bkgdata[i, 1] = mx_values[j]
            bkgdata[i, 2] = 0
        conc = np.concatenate((mwwbb_raw[j],bkgdata))
        np.savetxt('data/mwwbb/mwwbb_%s.dat' %mx_text[j], conc, fmt='%f')


def param_merge():
    mwwbb_400 = np.loadtxt('data/mwwbb/mwwbb_400.dat')
    mwwbb_500 = np.loadtxt('data/mwwbb/mwwbb_500.dat')
    mwwbb_600 = np.loadtxt('data/mwwbb/mwwbb_600.dat')
    mwwbb_700 = np.loadtxt('data/mwwbb/mwwbb_700.dat')
    mwwbb_800 = np.loadtxt('data/mwwbb/mwwbb_800.dat')
    mwwbb_900 = np.loadtxt('data/mwwbb/mwwbb_900.dat')
    mwwbb_1000 = np.loadtxt('data/mwwbb/mwwbb_1000.dat')
    mwwbb_1100 = np.loadtxt('data/mwwbb/mwwbb_1100.dat')
    mwwbb_1200 = np.loadtxt('data/mwwbb/mwwbb_1200.dat')
    mwwbb_1300 = np.loadtxt('data/mwwbb/mwwbb_1300.dat')
    mwwbb_1400 = np.loadtxt('data/mwwbb/mwwbb_1400.dat')
    mwwbb_1500 = np.loadtxt('data/mwwbb/mwwbb_1500.dat')

    mwwbb_raw = [mwwbb_400, mwwbb_500, mwwbb_600, mwwbb_700, mwwbb_800, mwwbb_900, mwwbb_1000, mwwbb_1100, mwwbb_1200, mwwbb_1300, mwwbb_1400, mwwbb_1500]
    mx_values = [400.000000, 500.000000, 600.000000, 700.000000, 800.000000, 900.000000, 1000.000000, 1100.000000, 1200.000000, 1300.000000, 1400.000000, 1500.000000]
    mx_text = ['400', '500', '600', '700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']

    data_complete = np.concatenate((
                        #mwwbb_400, 
                        mwwbb_500, 
                        #mwwbb_600,
                        #mwwbb_700,
                        #mwwbb_800,
                        #mwwbb_900,
                        mwwbb_1000,
                        #mwwbb_1100,
                        #mwwbb_1200,
                        #mwwbb_1300,
                        #mwwbb_1400,
                        mwwbb_1500), 
                        axis=0)
    #print data_complete
    np.savetxt('data/mwwbb/mwwbb_complete.dat', data_complete, fmt='%f')

def plt_histogram():
    mwwbb_400 = np.loadtxt('data/mwwbb_raw/mwwbb_400.dat')
    mwwbb_500 = np.loadtxt('data/mwwbb_raw/mwwbb_500.dat')
    mwwbb_600 = np.loadtxt('data/mwwbb_raw/mwwbb_600.dat')
    mwwbb_700 = np.loadtxt('data/mwwbb_raw/mwwbb_700.dat')
    mwwbb_800 = np.loadtxt('data/mwwbb_raw/mwwbb_800.dat')
    mwwbb_900 = np.loadtxt('data/mwwbb_raw/mwwbb_900.dat')
    mwwbb_1000 = np.loadtxt('data/mwwbb_raw/mwwbb_1000.dat')
    mwwbb_1100 = np.loadtxt('data/mwwbb_raw/mwwbb_1100.dat')
    mwwbb_1200 = np.loadtxt('data/mwwbb_raw/mwwbb_1200.dat')
    mwwbb_1300 = np.loadtxt('data/mwwbb_raw/mwwbb_1300.dat')
    mwwbb_1400 = np.loadtxt('data/mwwbb_raw/mwwbb_1400.dat')
    mwwbb_1500 = np.loadtxt('data/mwwbb_raw/mwwbb_1500.dat')

    mwwbb_raw = [mwwbb_400, mwwbb_500, mwwbb_600, mwwbb_700, mwwbb_800, mwwbb_900, mwwbb_1000, mwwbb_1100, mwwbb_1200, mwwbb_1300, mwwbb_1400, mwwbb_1500]
    mx_values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    mx_text = ['400', '500', '600', '700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']

    for i in range(12):
        n, bins, patches = plt.hist(mwwbb_raw[i][:,0], 50, histtype='stepfilled', alpha=0.75)
        plt.setp(patches)
        plt.title('mx = %s' %mx_values[i])
        plt.grid(True)
        plt.xlim([0, 4000])
        plt.ylim([0, 500])
        plt.savefig('plots/histograms/histo_%s.pdf'%mx_text[i])
        plt.savefig('plots/images/histograms/histo_%s.png'%mx_text[i])
        plt.clf()

def mwwbb_fixed(iterations):
    mwwbb_400 = np.loadtxt('data/mwwbb/mwwbb_400.dat')
    mwwbb_500 = np.loadtxt('data/mwwbb/mwwbb_500.dat')
    mwwbb_600 = np.loadtxt('data/mwwbb/mwwbb_600.dat')
    mwwbb_700 = np.loadtxt('data/mwwbb/mwwbb_700.dat')
    mwwbb_800 = np.loadtxt('data/mwwbb/mwwbb_800.dat')
    mwwbb_900 = np.loadtxt('data/mwwbb/mwwbb_900.dat')
    mwwbb_1000 = np.loadtxt('data/mwwbb/mwwbb_1000.dat')
    mwwbb_1100 = np.loadtxt('data/mwwbb/mwwbb_1100.dat')
    mwwbb_1200 = np.loadtxt('data/mwwbb/mwwbb_1200.dat')
    mwwbb_1300 = np.loadtxt('data/mwwbb/mwwbb_1300.dat')
    mwwbb_1400 = np.loadtxt('data/mwwbb/mwwbb_1400.dat')
    mwwbb_1500 = np.loadtxt('data/mwwbb/mwwbb_1500.dat')
    
    mwwbb_train = [mwwbb_400[:,0:2], 
                    mwwbb_500[:,0:2], 
                    mwwbb_600[:,0:2], 
                    mwwbb_700[:,0:2],
                    mwwbb_800[:,0:2], 
                    mwwbb_900[:,0:2], 
                    mwwbb_1000[:,0:2], 
                    mwwbb_1100[:,0:2],
                    mwwbb_1200[:,0:2], 
                    mwwbb_1300[:,0:2], 
                    mwwbb_1400[:,0:2], 
                    mwwbb_1500[:,0:2]]
    
    mwwbb_target = [mwwbb_400[:,2], 
                    mwwbb_500[:,2], 
                    mwwbb_600[:,2], 
                    mwwbb_700[:,2],
                    mwwbb_800[:,2], 
                    mwwbb_900[:,2], 
                    mwwbb_1000[:,2], 
                    mwwbb_1100[:,2],
                    mwwbb_1200[:,2], 
                    mwwbb_1300[:,2], 
                    mwwbb_1400[:,2], 
                    mwwbb_1500[:,2]]
    
    mwwbb_file = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    mwwbb_text = ['400', '500', '600', '700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']


    for i in range(len(mwwbb_train)):
        print 'Working on mu=%s' %mwwbb_text[i]
        traindata = mwwbb_train[i]
        targetdata = mwwbb_target[i]
        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', 
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    n_iter=iterations, 
                    #learning_momentum=0.1,
                    #batch_size=5,
                    learning_rule="nesterov",  
                    #valid_size=0.05,
                    #verbose=True,
                    #debug=True
                    ))])



        nn.fit(traindata, targetdata)
        
        fit_score = nn.score(traindata, targetdata)
        print 'score = %s' %fit_score
        outputs = nn.predict(traindata)
        plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5, label='$\mu=$%s' %mwwbb_text[i])
    
    plt.xlim([0, 3000])
    plt.ylim([-0.2, 1.2])    
    plt.legend(bbox_to_anchor=(0.85, 1.05), loc=2, borderaxespad=0)
    plt.grid(True)
    plt.suptitle('Theano NN fixed training for mwwbb input', fontsize=14, fontweight='bold')    
    #plt.show()
    plt.savefig('plots/fixedTraining.pdf')
    plt.savefig('plots/images/fixedTraining.png')
    plt.clf()

    pickle.dump(nn, open('data/fixed.pkl', 'wb'))


def mwwbb_parameterized(iterations):
    mwwbb_complete = np.loadtxt('data/mwwbb/mwwbb_complete.dat')

    traindata      = mwwbb_complete[:,0:2]
    targetdata      = mwwbb_complete[:,2]
    print traindata
    print targetdata

    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', 
            Regressor(
                layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                learning_rate=0.01,
                n_iter=iterations, 
                #learning_momentum=0.1,
                #batch_size=5,
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
    plt.ylabel('NN_output ( training_input )')
    plt.xlabel('training_input')
    plt.xlim([0, 4000])
    plt.ylim([-0.2, 1.2])
    #plt.axhline(y=0, color = 'black', linewidth = 2, alpha=0.75)
    #plt.axhline(y=1, color = 'black', linewidth = 2, alpha=0.75)
    plt.grid(True)
    plt.suptitle('Theano NN fixed training for mwwbb input', fontsize=14, fontweight='bold')
    plt.savefig('plots/paramTraining.pdf')
    plt.savefig('plots/images/paramTraining.png')
    #plt.show()
    plt.clf()

    pickle.dump(nn, open('data/param.pkl', 'wb'))

def scikitlearnFunc(x, alpha):
    nn = pickle.load(open('data/param.pkl','rb'))
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
        plt.plot(-4,0, plt_marker[i], alpha=0.5, label="$\mu=$%s" %alpha[i])
    plt.legend(bbox_to_anchor=(0.735, 0.98), loc=2, borderaxespad=0)
    plt.ylabel('NN_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([0, 4000])
    plt.ylim([-0.2, 1.2])
    plt.grid(True)
    plt.suptitle('Theano NN regression output for parameterized mwwbb input',
               fontsize=12, fontweight='bold')

    plt.savefig('plots/paramTraining_complete.pdf')
    plt.savefig('plots/images/paramTraining_complete.png')
    #plt.show()

def ROOT_importer():
    mwwbb_400 = 'data/1d_poi/xttbar_14tev_mx400_jes1.0.root'
    mwwbb_500 = 'data/1d_poi/xttbar_14tev_mx500_jes1.0.root'
    mwwbb_600 = 'data/1d_poi/xttbar_14tev_mx600_jes1.0.root'
    mwwbb_700 = 'data/1d_poi/xttbar_14tev_mx700_jes1.0.root'
    mwwbb_800 = 'data/1d_poi/xttbar_14tev_mx800_jes1.0.root'
    mwwbb_900 = 'data/1d_poi/xttbar_14tev_mx900_jes1.0.root'
    mwwbb_1000 = 'data/1d_poi/xttbar_14tev_mx1000_jes1.0.root'
    mwwbb_1100 = 'data/1d_poi/xttbar_14tev_mx1100_jes1.0.root'
    mwwbb_1200 = 'data/1d_poi/xttbar_14tev_mx1200_jes1.0.root'
    mwwbb_1300 = 'data/1d_poi/xttbar_14tev_mx1300_jes1.0.root'
    mwwbb_1400 = 'data/1d_poi/xttbar_14tev_mx1400_jes1.0.root'
    mwwbb_1500 = 'data/1d_poi/xttbar_14tev_mx1500_jes1.0.root'

    mwwbb_list = [mwwbb_400, mwwbb_500, mwwbb_600, mwwbb_700, mwwbb_800, mwwbb_900, mwwbb_1000, mwwbb_1100,
        mwwbb_1200, mwwbb_1300, mwwbb_1400, mwwbb_1500]
    for i in range(1):
        f = ROOT.TFile(mwwbb_list[i])
        t = f.Get('xtt')
        t.Print()
        t.Draw('mwwbb')
        array = []
        for n in range(100):
            a = t.Scan('mwwbb','mwwbb','',1,n)
            aa = a.GetValue(a)
            array.append(aa)
        print array
    #print f.ls()
    #t = f.Get('xtt')
    #t.Print()
    #mwwbb_array= np.zeros((2*1334, 3))
    #print t

if __name__ == '__main__':   
    ROOT_importer()
    #bkg_merge()
    #param_merge()
    #plt_histogram()
    #mwwbb_fixed(250)
    #mwwbb_parameterized(250)
    #mwwbbParameterizedRunner()
