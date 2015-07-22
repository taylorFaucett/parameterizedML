'''
author Taylor Faucett <tfaucett@uci.edu>

This script utilizes Theano/Pylearn2 and SKLearn-NeuralNetwork to create a fixed and parameterized
machine learning scheme. Datasets are generated for multiple gaussian shaped
signals and a uniform (i.e. flat) background. trainFixed uses a regression NN
to learn for n gaussians at fixed means (mu) which can map a 1D array to signal/background 
values of 1 or 0. trainParam trains for all n gaussians simultaneously and then trains for 
these gaussian signals with a parameter by a secondary input (alpha).
'''


import ROOT
import numpy as np
import pylab as P
from sklearn import svm
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer, Convolution
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt

import sys
import logging
import time

'''
logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)
'''

plt_marker=['bo', 'go', 'ro', 'co', 'mo', 'yo', 'bo', 'wo']

def makeData():
    print "Entering makeData"
    musteps  = 5
    numTrain = 500
    numTest  = numTrain

    # Initialize ROOTs RooWorkspace
    w = ROOT.RooWorkspace('w')

    # Generate Gaussian signals
    print "Generating Gaussians PDFs"
    w.factory('Gaussian::g(x[-5,5],mu[0,-3,3],sigma[0.25, 0, 2])')

    # Generate a flat background signal
    print "Generating a flat background PDF"
    w.factory('Uniform::e(x)')

    # Combine signal and background
    print "Generating a composite PDF of signal and background"
    w.factory('SUM::model(s[50,0,100]*g,b[100,0,1000]*e)')

    # Print and write data to file
    w.Print()
    w.writeToFile('data/workspace_GausSigOnFlatBkg.root')

    # Define variables
    x      = w.var('x')
    mu     = w.var('mu')
    pdf    = w.pdf('model')
    sigpdf = w.pdf('g')
    bkgpdf = w.pdf('e')

    # create training, testing data
    # np.zeros((rows, columns))
    traindata  = np.zeros((2 * numTrain * musteps, 2))
    targetdata = np.zeros(2 * numTrain * musteps)
    testdata   = np.zeros((numTest * musteps, 2))
    testdata1  = np.zeros((numTest * musteps, 2))

    # Fill traindata, testdata and testdata1
    for mustep, muval in enumerate(np.linspace(-1, 1, musteps)):
        mu.setVal(muval)
        sigdata = sigpdf.generate(ROOT.RooArgSet(x), numTrain)
        bkgdata = bkgpdf.generate(ROOT.RooArgSet(x), numTrain)
        alldata = pdf.generate(ROOT.RooArgSet(x), numTest)

        for i in range(numTrain):
            traindata[i + mustep * numTrain, 0] = sigdata.get(i).getRealValue('x')
            traindata[i + mustep * numTrain, 1] = muval
            targetdata[i + mustep * numTrain] = 1
        for i in range(numTrain):
            traindata[i + mustep * numTrain + musteps * numTrain,
                    0] = bkgdata.get(i).getRealValue('x')
            traindata[i + mustep * numTrain + musteps * numTrain, 1] = muval
            targetdata[i + mustep * numTrain + musteps * numTrain] = 0
        for i in range(numTest):
            testdata[i + mustep * numTest, 0] = alldata.get(i).getRealValue('x')
            testdata[i + mustep * numTest, 1] = 0.
        for i in range(numTest):
            testdata1[i + mustep * numTest, 0] = alldata.get(i).getRealValue('x')
            testdata1[i + mustep * numTest, 1] = 1  # optionally 2*(i%2)-1.

    np.savetxt("data/traindata.dat", np.column_stack((traindata, targetdata)), fmt='%f')
    np.savetxt("data/testdata.dat", testdata, fmt='%f')
    np.savetxt("data/testdata1.dat", testdata1, fmt='%f')


def plotPDF():
    print "Entering plotPDF"

    '''
    makePdfPlot pulls the generated data from traindata.dat
    and plots the components
    '''

    # Initialize a ROOT file
    f = ROOT.TFile("data/workspace_GausSigOnFlatBkg.root", 'r')

    # Initialize variables
    w  = f.Get('w')
    x  = w.var('x')
    mu = w.var('mu')

    # Initialize a pdf for the signal and background data
    sigpdf = w.pdf('g')
    bkgpdf = w.pdf('e')
    pdf    = w.pdf('model')

    # Initialize plotting frame and plots gaussian (g) and background (e)
    # Saves the output to modelPlot.pdf
    frame = x.frame()
    pdf.plotOn(frame)
    pdf.plotOn(frame, ROOT.RooFit.Components('g'), ROOT.RooFit.LineColor(ROOT.kRed))
    pdf.plotOn(frame, ROOT.RooFit.Components('e'), ROOT.RooFit.LineColor(ROOT.kGreen))
    c1 = ROOT.TCanvas()
    frame.Draw()
    c1.SaveAs('plots/modelPlot.pdf')
    c1.SaveAs('plots/images/modelPlot.png')



def trainFixed():
    '''
    train a machine learner based on data from some fixed parameter point.
    save to fixed.pkl
    '''

    print "Entering fixed number model training"

    '''
    Import traindata.dat and split the data into two components
    traindata cuts traindata.dat to include a 1D histogram and counting
    column. Targetdata cuts just the signal/background indicator.
    '''
    trainAndTarget = np.loadtxt('data/traindata.dat')
    testdata       = np.loadtxt('data/testdata.dat')
    traindata      = trainAndTarget[:, 0:2]
    targetdata     = trainAndTarget[:, 2]

    # muPoints is a list of the gaussian averages generated (e.g. [-1, 0, 1])
    # chunk is the size of each piece of data that corresponds to one of the muPoints
    muPoints = np.unique(traindata[:, 1])
    chunk      = len(traindata) / len(muPoints) / 2
    shift      = len(traindata) / 2

    # Initialize ML method (SVM or NN)
    print "Machine Learning method initialized"
    #nn = svm.NuSVR(nu=1)
    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', 
            Regressor(
                layers =[Layer("Sigmoid", units=10),Layer("Sigmoid")],
                learning_rate=0.01,
                n_iter=100, 
                #learning_momentum=0.1,
                #batch_size=5,
                learning_rule="nesterov",  
                #valid_size=0.05,
                #verbose=True,
                #debug=True
                ))])
    print nn
    #nn = Classifier(layers =[Layer("Maxout", units=100, pieces=2), Layer("Softmax")],learning_rate=0.02,n_iter=10)

    for i in range(len(muPoints)):
        # lowChunk and highChunk define the lower and upper bands of each
        # chunk as it moves through the data set.
        lowChunk      = i
        highChunk     = i + 1

        # reducedtrain and reducedtarget cut the full dataset down into pieces
        # consisting of just a single "chunk" of data corresponding to a specific
        # mu value
        reducedtrain  = np.concatenate((traindata[lowChunk * chunk: highChunk * chunk, 0],
                                       traindata[lowChunk * chunk + shift: highChunk * chunk + shift, 0]))
        reducedtarget = np.concatenate((targetdata[lowChunk * chunk: highChunk * chunk],
                                        targetdata[lowChunk * chunk + shift: highChunk * chunk + shift]))

        # ML fit function
        # fit(Training Vectors, Target Values)
        # reducedtrain.reshape((NUM OF VALUES, feature))
        nn.fit(reducedtrain.reshape((len(reducedtrain), 1)), reducedtarget)
        fit_score = nn.score(reducedtrain.reshape((len(reducedtrain), 1)), reducedtarget)
        print 'score = %s' %fit_score
        # predict(X) - Performs a regression on samples in X
        outputs = nn.predict(testdata[:, 0].reshape((len(testdata), 1)))
        plt.plot(testdata[:, 0], outputs, 'o', alpha=0.5, label='$\mu=$%s' % muPoints[i])

        #plt.axvline(x=muPoints[i], label="$\mu=%s$" %muPoints[i], linewidth = 2, color = colorArray[i])

    # Plot settings for the fixed training mode
    plt.legend(bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0)
    plt.ylabel('NN_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([-5, 5])
    plt.ylim([-0.2, 1.2])
    plt.grid(True)
    plt.suptitle('Theano NN regression output for fixed gaussians',
               fontsize=12, fontweight='bold')
    #plt.show()
    plt.savefig('plots/fixedTraining.pdf')
    plt.savefig('plots/images/fixedTraining.png')
    plt.clf()

    # export training results to fixed.pkl
    # joblib.dump is a numpy method which will "pickle" the training method
    # resulting in multiple npy files.
    pickle.dump(nn, open('data/fixed.pkl', 'wb'))

def trainParam():
    print "Entering trainParam"
    # import training and testing data from makeData
    trainAndTarget = np.loadtxt('data/traindata.dat')
    testdata       = np.loadtxt('data/testdata.dat')
    traindata      = trainAndTarget[:, 0:2]
    targetdata     = trainAndTarget[:, 2]

    # Initialize ML method (SVM or NN)
    print "Machine Learning method initialized"

    #nn = svm.NuSVR(nu=1)
    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', 
            Regressor(
                layers =[Layer("Sigmoid", units=10),Layer("Sigmoid")],
                learning_rate=0.01,
                n_iter=100, 
                #learning_momentum=0.1,
                #batch_size=5,
                learning_rule="nesterov",  
                #valid_size=0.05,
                #verbose=True,
                #debug=True
                ))])
    print nn

    #nn = Classifier(layers =[Layer("Maxout", units=100, pieces=2), Layer("Softmax")],learning_rate=0.02,n_iter=10)
    nn.fit(traindata, targetdata)
    
    fit_score = nn.score(traindata, targetdata)
    print 'score = %s' %fit_score
    # Training outputs
    outputs = nn.predict(traindata)

    # Plot settings
    plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5)
    plt.ylabel('sv_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([-5, 5])
    plt.ylim([-0.2, 1.2])
    #plt.axhline(y=0, color = 'black', linewidth = 2, alpha=0.75)
    #plt.axhline(y=1, color = 'black', linewidth = 2, alpha=0.75)
    plt.grid(True)
    plt.suptitle('Parametrized SV Mapping (SV Output vs Data Input)',
               fontsize=14, fontweight='bold')
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


def parameterizedRunner():
    alpha = [-1, -0.5, 0.0, +0.5, +1.0]
    step = 100
    print "Running on %s alpha values: %s" %(len(alpha), alpha)
    for a in range(len(alpha)):
        print 'working on alpha=%s' %alpha[a]
        for x in range(-500, 500, 1):
            outputs = scikitlearnFunc(x/100., alpha[a])
            plt.plot(x/100., outputs[0], plt_marker[a], alpha=0.5)
    for i in range(len(alpha)):
        plt.plot(-4,0, plt_marker[i], alpha=0.5, label="$\mu=$%s" %alpha[i])
    plt.legend(bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0)
    plt.ylabel('NN_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([alpha[0]-1, alpha[-1]+1])
    plt.ylim([-0.2, 1.2])
    plt.grid(True)
    plt.suptitle('Theano NN regression output for parameterized gaussians',
               fontsize=12, fontweight='bold')

    plt.savefig('plots/paramTraining_complete.pdf')
    plt.savefig('plots/images/paramTraining_complete.png')
    #plt.show()



def mwwbb_importer():
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


def mwwbb_histogram():
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

    mwwbb_list = [mwwbb_400, mwwbb_500, mwwbb_600, mwwbb_700, mwwbb_800, mwwbb_900, mwwbb_1000, mwwbb_1100,
        mwwbb_1200, mwwbb_1300, mwwbb_1400, mwwbb_1500]
    mwwbb_file = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    for i in range(12):
        n, bins, patches = P.hist(mwwbb_list[i][:,0], 50, histtype='stepfilled')
        P.setp(patches)
        P.savefig('plots/mwwbb/histo_%s.pdf'%mwwbb_file[i])
        P.clf()

def mwwbb_fixed():
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
    mwwbb_list = [mwwbb_400, mwwbb_500, mwwbb_600, mwwbb_700, mwwbb_800, mwwbb_900, mwwbb_1000, mwwbb_1100,
        mwwbb_1200, mwwbb_1300, mwwbb_1400, mwwbb_1500]
    mwwbb_file = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

    traindata = mwwbb_400[:, 0:2]
    targetdata = mwwbb_400[:, 2]
    print traindata
    print targetdata

    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', 
            Regressor(
                layers =[Layer("Sigmoid", units=10),Layer("Sigmoid")],
                learning_rate=0.01,
                n_iter=100, 
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
    # Training outputs
    outputs = nn.predict(traindata)

    plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5, label='$\mu=400$')
    #plt.xlim([-5, 5])
    plt.ylim([-0.2, 1.2])    
    plt.show()


if __name__ == '__main__':
    mwwbb_importer()
    #mwwbb_histogram()
    #mwwbb_fixed()
    #makeData()
    #plotPDF()
    #trainFixed()
    #trainParam()
    #parameterizedRunner()
