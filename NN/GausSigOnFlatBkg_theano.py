'''
author Taylor Faucett <tfaucett@uci.edu>

This script will use a NN method from Theano
'''


import ROOT
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

def makeData():
    print "Entering makeData"
    muList = [-2, -1, 0, +1, +2]
    N = 500

    np.savetxt('data/muData.dat', muList, fmt='%f')
    for i in range(len(muList)):
        print 'Generating Gaussians for mu=%s' %muList[i]
    print 'Generating %s samples per gaussian' %N
    print 'Generating %s total data points' %N*len(muList)

    # create training, testing data
    # np.zeros((rows, columns))
    trainData  = np.zeros((2*N,2))

    #print trainData
    #print targetData

    w = ROOT.RooWorkspace('w')

    print 'Processing mu = %s' %muList[i]
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

    x       = w.var('x')
    mu      = w.var('mu')
    pdf     = w.pdf('model')
    sigpdf  = w.pdf('g')
    bkgpdf  = w.pdf('e')

    for i in range(len(muList)):
        mu.setVal(muList[i])
        sigdata = sigpdf.generate(ROOT.RooArgSet(x), N)
        bkgdata = bkgpdf.generate(ROOT.RooArgSet(x), N)
        alldata = pdf.generate(ROOT.RooArgSet(x), N)
        for j in range(N):
            trainData[j, 0] = sigdata.get(j).getRealValue('x')
            trainData[j, 1] = 1
        for j in range(N):
            trainData[N+j, 0] = bkgdata.get(j).getRealValue('x')
            trainData[N+j, 1] = 0
        np.savetxt("data/traindata_mu_(%s).dat" %muList[i], trainData, fmt='%f')

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

    targetdata     = np.loadtxt('data/muData.dat')

    # Initialize SciKitLearns Nu-Support Vector Regression
    print "SciKit Learn initialized using Nu-Support Vector Regression (SVC)"
    clf = svm.SVR()

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

        # SciKitLearns Nu-Support Vector Regression fit function followed
        # fit(Training Vectors, Target Values)
        # reducedtrain.reshape((NUM OF VALUES, feature))
        clf.fit(reducedtrain.reshape((len(reducedtrain), 1)), reducedtarget)

        # predict(X) - Performs a regression on samples in X
        outputs = clf.predict(testdata[:, 0].reshape((len(testdata), 1)))
        plt.plot(testdata[:, 0], outputs, 'o', alpha=0.5, label='$\mu=$%s' % muPoints[i])

        #plt.axvline(x=muPoints[i], label="$\mu=%s$" %muPoints[i], linewidth = 2, color = colorArray[i])

    # Plot settings for the fixed training mode
    plt.legend(bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0)
    plt.ylabel('sv_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([0, 1])
    plt.ylim([-0.2, 1.2])
    plt.grid(True)
    plt.suptitle('Complete SV training output as a function of test data input',
               fontsize=14, fontweight='bold')

    #plt.show()
    plt.savefig('plots/fixedTraining.pdf')
    plt.savefig('plots/images/fixedTraining.png')
    plt.clf()

    # export training results to fixed.pkl
    # joblib.dump is a numpy method which will "pickle" the training method
    # resulting in multiple npy files.
    joblib.dump(clf, "data/fixed.pkl")


def trainParam():
    print "Entering trainParam"
    # import training and testing data from makeData
    trainAndTarget = np.loadtxt('data/traindata.dat')
    testdata       = np.loadtxt('data/testdata.dat')
    traindata      = trainAndTarget[:, 0:2]
    targetdata     = trainAndTarget[:, 2]

    # Training based on the complete data set provided from makeData
    clf = svm.SVR()
    clf.fit(traindata, targetdata)

    # Training outputs
    outputs = clf.predict(traindata)

    # Plot settings
    plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5)
    plt.ylabel('sv_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([0, 1])
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

    joblib.dump(clf, "data/param.pkl")


def scikitlearnFunc(x=0.0, alpha=0.5):
    # print "scikitlearnTest"
    clf = joblib.load('data/param.pkl')
    # print "inouttest input was", x
    traindata = np.array((x, alpha))
    outputs   = clf.predict(traindata)

    #print 'x,alpha,output =', x, alpha, outputs[0]
    plt.plot(x, outputs[0], 'ro', alpha=0.5)
    return outputs[0]


def scikitlearnFunc1(x=0.0, alpha=0.0):
    clf = joblib.load('data/param.pkl')
    traindata = np.array((x, alpha))
    outputs   = clf.predict(traindata)
    plt.plot(x, outputs[0], 'bo', alpha=0.5)
    return outputs[0]

def scikitlearnFunc2(x=0.0, alpha=0.25):
    clf = joblib.load('data/param.pkl')
    traindata = np.array((x, alpha))
    outputs   = clf.predict(traindata)
    plt.plot(x, outputs[0], 'go', alpha=0.5)
    return outputs[0]

def scikitlearnFunc3(x=0.0, alpha=0.5):
    clf = joblib.load('data/param.pkl')
    traindata = np.array((x, alpha))
    outputs   = clf.predict(traindata)
    plt.plot(x, outputs[0], 'ro', alpha=0.5)
    return outputs[0]

def scikitlearnFunc4(x=0.0, alpha=0.75):
    clf = joblib.load('data/param.pkl')
    traindata = np.array((x, alpha))
    outputs   = clf.predict(traindata)
    plt.plot(x, outputs[0], 'co', alpha=0.5)
    return outputs[0]

def scikitlearnFunc5(x=0.0, alpha=1.0):
    clf = joblib.load('data/param.pkl')
    traindata = np.array((x, alpha))
    outputs   = clf.predict(traindata)
    plt.plot(x, outputs[0], 'mo', alpha=0.5)
    return outputs[0]


def testSciKitLearnWrapper():
    # need a RooAbsReal to evaluate NN(x,mu)
    mu = 0.5
    ROOT.gSystem.Load('SciKitLearnWrapper/libSciKitLearnWrapper')
    x  = ROOT.RooRealVar('x', 'x', 0.2, -5, 5)

    nn1 = ROOT.SciKitLearnWrapper('nn1', 'nn1', x)
    nn1.RegisterCallBack(scikitlearnFunc1)

    nn2 = ROOT.SciKitLearnWrapper('nn2', 'nn2', x)
    nn2.RegisterCallBack(scikitlearnFunc2)

    nn3 = ROOT.SciKitLearnWrapper('nn3', 'nn3', x)
    nn3.RegisterCallBack(scikitlearnFunc3)

    nn4 = ROOT.SciKitLearnWrapper('nn4', 'nn4', x)
    nn4.RegisterCallBack(scikitlearnFunc4)

    nn5 = ROOT.SciKitLearnWrapper('nn5', 'nn5', x)
    nn5.RegisterCallBack(scikitlearnFunc5)

    c1    = ROOT.TCanvas('c1')
    frame = x.frame()
    nn1.plotOn(frame)
    nn2.plotOn(frame)
    nn3.plotOn(frame)
    nn4.plotOn(frame)
    nn5.plotOn(frame)

    frame.Draw()
    c1.SaveAs('plots/paramOutput.pdf')
    c1.SaveAs('plots/images/paramOutput.png')
    plt.ylabel('sv_output( training_input )')
    plt.xlabel('training_input')
    plt.xlim([-5, 5])
    plt.ylim([-0.2, 1.2])
    plt.plot(0,0,"bo", label="$\mu=0.0$" )
    plt.plot(0,0,"go", label="$\mu=0.25$" )
    plt.plot(0,0,"ro", label="$\mu=0.5$" )
    plt.plot(0,0,"co", label="$\mu=0.75$" )
    plt.plot(0,0,"mo", label="$\mu=1.0$" )
    plt.xlim([0, 1])
    plt.ylim([-0.2, 1.2])
    #plt.axvline(x=mu, label="$\mu=%s$" %mu, linewidth = 2)
    #plt.axhline(y=0, color = 'black', linewidth = 2, alpha=0.75)
    #plt.axhline(y=1, color = 'black', linewidth = 2, alpha=0.75)
    plt.grid(True)
    #plt.fill(True)
    plt.suptitle('Parametrized SV Mapping (SV Output vs Data Input)',
               fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0)
    plt.savefig('plots/paramTraining_complete.pdf')
    plt.savefig('plots/images/paramTraining_complete.png')
    #plt.show()


if __name__ == '__main__':
    makeData()
    plotPDF()
    #trainFixed()
    #trainParam()
    #testSciKitLearnWrapper()
