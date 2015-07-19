'''
author Taylor Faucett <tfaucett@uci.edu>
'''


import ROOT
import numpy as np
from sklearn import svm, linear_model, gaussian_process, cross_validation, datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer
import pickle

import sys
import logging

#logging.basicConfig(
#            format="%(message)s",
#            level=logging.DEBUG,
#            stream=sys.stdout)

import matplotlib.pyplot as plt


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
    for mustep, muval in enumerate(np.linspace(-2, 2, musteps)):
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
    c1.SaveAs('plots/modelPlot_5val.pdf')
    c1.SaveAs('plots/images/modelPlot_5val.png')


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

    nn = svm.NuSVR(nu=1)
    #nn = Regressor(layers =[Layer("Sigmoid", units=2),Layer("Sigmoid")],learning_rate=0.02,n_iter=100)
    #nn = Classifier(layers =[Layer("Maxout", units=100, pieces=2), Layer("Softmax")],learning_rate=0.02,n_iter=10)


    for i in range(len(muPoints)):
        # lowChunk and highChunk define the lower and upper bands of each
        # chunk as it moves through the data set.
        lowChunk      = i
        highChunk     = i + 1

        #colorArray    = ["blue", "green", "red"]
        # reducedtrain and reducedtarget cut the full dataset down into pieces
        # consisting of just a single "chunk" of data corresponding to a specific
        # mu value
        reducedtrain  = np.concatenate((traindata[lowChunk * chunk: highChunk * chunk, 0],
                                       traindata[lowChunk * chunk + shift: highChunk * chunk + shift, 0]))
        reducedtarget = np.concatenate((targetdata[lowChunk * chunk: highChunk * chunk],
                                        targetdata[lowChunk * chunk + shift: highChunk * chunk + shift]))

        # fit(Training Vectors, Target Values)
        # reducedtrain.reshape((NUM OF VALUES, feature))
        nn.fit(reducedtrain.reshape((len(reducedtrain), 1)), reducedtarget)

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
    plt.suptitle('Theano NN classification output for fixed gaussians',
               fontsize=12, fontweight='bold')

    #plt.show()
    plt.savefig('plots/fixedTraining_5val.pdf')
    plt.savefig('plots/images/fixedTraining_5val.png')
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

    # Training based on the complete data set provided from makeData
    nn = svm.NuSVR(nu=1)
    nn = Regressor(layers =[Layer("Sigmoid", units=100),Layer("Sigmoid", units=100)],learning_rate=0.02,n_iter=100)
    #nn = Classifier(layers =[Layer("Maxout", units=100, pieces=2), Layer("Softmax")],learning_rate=0.02,n_iter=10)
    nn.fit(traindata, targetdata)

    # Training outputs
    outputs = nn.predict(traindata)

    # Plot settings
    plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5)
    plt.ylabel('NN_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([-5, 5])
    plt.ylim([-0.2, 1.2])
    #plt.axhline(y=0, color = 'black', linewidth = 2, alpha=0.75)
    #plt.axhline(y=1, color = 'black', linewidth = 2, alpha=0.75)
    plt.grid(True)
    plt.suptitle('Parametrized NN Mapping (NN Output vs Data Input)',
               fontsize=12, fontweight='bold')
    plt.savefig('plots/paramTraining_5val.pdf')
    plt.savefig('plots/images/paramTraining_5val.png')
    #plt.show()
    plt.clf()

 
    #pickle.dump(nn, open('data/param.pkl', 'wb'))
    joblib.dump(nn, "data/param.pkl")


def scikitlearnFunc(x, alpha):
    #print "scikitlearnTest"
    #nn = pickle.load(open('data/param.pkl','rb'))
    nn = joblib.load('data/param.pkl')
    #print "inouttest input was", x
    traindata = np.array((x, alpha))
    outputs   = nn.predict(traindata)

    print 'x,alpha,output =', x, alpha, outputs[0]
    plt.plot(x, outputs, 'ro', alpha=0.5)
    #plt.show()


def testSciKitLearnWrapper():
    alpha = 0.5
    step = 100
    for x in range(-500, 500, 1):
        scikitlearnFunc(x/100., alpha)
    #plt.legend(bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0)
    plt.ylabel('NN_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([-5, 5])
    plt.ylim([-0.2, 1.2])
    plt.grid(True)
    plt.suptitle('Theano NN regression output for parameterized gaussians',
               fontsize=12, fontweight='bold')

    plt.savefig('plots/fixedTraining_5val.pdf')
    plt.savefig('plots/images/fixedTraining_5val.png')
    plt.show()
    #c1 = ROOT.TCanvas('c1')
    #frame = x.frame()
    #nn.plotOn(frame)
    #frame.Draw()
    #c1.SaveAs('plots/testWrapper.pdf')



if __name__ == '__main__':
    #makeData()
    #plotPDF()
    #trainFixed()
    #trainParam()
    testSciKitLearnWrapper()
