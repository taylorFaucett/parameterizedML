'''
author Taylor Faucett <tfaucett@uci.edu>

This script utilizes Theano/Pylearn2 and SKLearn-NeuralNetwork to create a fixed
and parameterized machine learning scheme. Datasets are generated for multiple
gaussian shaped signals and a uniform (i.e. flat) background. trainFixed uses a
regression NN to learn for n gaussians at fixed means (mu) which maps a 1D
array to values between signal/background values of 1 or 0. trainParam trains for all n
gaussians simultaneously and then trains for these gaussian signals with a
parameter by a secondary input (alpha).
'''


import ROOT
import numpy as np
import glob
from sklearn import svm
from array import *
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sknn.mlp import Regressor, Classifier, Layer, Convolution
import pickle
import smtplib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt

def gaussianData(MU, SIG, numTrain):
    # Initialize ROOTs RooWorkspace
    w = ROOT.RooWorkspace('w')

    # Generate Gaussian signals
    print "Generating Gaussians PDFs"
    w.factory('Gaussian::g(x[-10,10],mu[%0.3f,-3,3],sigma[%0.3f, 0, 2])' %(MU, SIG))

    # Generate a flat background signal
    print "Generating a flat background PDF"
    w.factory('Uniform::e(x)')

    # Combine signal and background
    print "Generating a composite PDF of signal and background"
    w.factory('SUM::model(0.5*g,0.5*e)')

    # Print and write data to file
    w.Print()
    w.writeToFile('data/workspace_GausSigOnFlatBkg.root')

    # Define variables
    x      = w.var('x')
    mu     = w.var('mu')
    pdf    = w.pdf('model')
    sigpdf = w.pdf('g')
    bkgpdf = w.pdf('e')    
    traindata  = np.zeros((2 * numTrain, 2))
    targetdata = np.zeros(2 * numTrain)

    sigdata = sigpdf.generate(ROOT.RooArgSet(x), numTrain)
    bkgdata = bkgpdf.generate(ROOT.RooArgSet(x), numTrain)
    alldata = pdf.generate(ROOT.RooArgSet(x), numTrain)

    for i in range(numTrain):
        traindata[i,0] = sigdata.get(i).getRealValue('x')
        traindata[i,1] = MU
        targetdata[i] = 1
    for i in range(numTrain):
        traindata[i + numTrain, 0] = bkgdata.get(i).getRealValue('x')
        traindata[i + numTrain, 1] = MU
        targetdata[i + numTrain] = 0

    np.savetxt("data/training_data/traindata_mu_%0.3f.dat" %MU, np.column_stack((traindata, targetdata)), fmt='%f')

def plt_histogram():
    mu_values = [-2.000, -1.000, 0.000, 1.000, 2.000]
    plt_color=['blue', 'green', 'red', 'cyan', 'magenta','black']
    for idx, mu in enumerate(mu_values):
        data = np.loadtxt('data/training_data/traindata_mu_%0.3f.dat' %mu)
        bin_size   = 200
        bin_width = 10.0/bin_size
        n, bins, patches = plt.hist(data[:,0],
                            bins=bin_size, histtype='stepfilled',
                            alpha=0.5, label='mu=%0.3f' %mu, color=plt_color[idx],
                            rasterized=True)
        plt.setp(patches)
    plt.ylabel('Number of events$/%0.2f x$' %bin_width)
    plt.xlabel('x')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1))
    plt.xlim([-5,5])
    #plt.ylim([0,10])
    plt.savefig('plots/histogram_gaussian.pdf', dpi=400)
    plt.savefig('plots/images/histogram_gaussian.png')
    #plt.show()
    plt.clf()

def fixed_training(iterations):
    print 'Entering fixed_training'
    mu_values = [-2.000, -1.000, 0.000, 1.000, 2.000]

    nn = Pipeline([
    ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
    ('neural network',
        Regressor(
            layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
            #learning_rate=0.01,
            n_iter=iterations,
            #learning_momentum=0.1,
            batch_size=100,
            learning_rule="nesterov",
            #valid_size=0.05,
            verbose=True,
            #debug=True
            ))])
    #print nn

    for idx, mu in enumerate(mu_values):
        print 'Training on mu=%0.3f' %mu
        data = np.loadtxt('data/training_data/traindata_mu_%0.3f.dat' %mu)
        training = data[:,0:1]
        target = data[:,2:]
        nn.fit(training, target)
        fit_score = nn.score(training, target)
        print 'score = %s' %fit_score

        outputs = nn.predict(training)
        output_data = np.vstack((training[:,0], target[:,0], outputs[:,0])).T
        np.savetxt('data/plot_data/fixed_%0.3f.dat' %mu, output_data, fmt='%f')

        actual = target
        predictions = outputs
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        ROC_plot = np.vstack((fpr, tpr)).T
        ROC_AUC = [auc(fpr, tpr)]
        np.savetxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %mu, ROC_plot, fmt='%f')
        np.savetxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %mu, ROC_AUC)
        pickle.dump(nn, open('data/fixed_%0.3f.pkl' %mu, 'wb'))

def fixed_output_plot():
    print 'Entering fixed_output_plot'
    mu_values = [-2.000, -1.000, 0.000, 1.000, 2.000]

    for idx, mu in enumerate(mu_values):
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %mu)
        inputs = data[:,0]
        outputs = data[:,2]
        plt.plot(inputs, outputs, '.', rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('input')
    plt.xlim([-5, 5])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.savefig('plots/fixed_output_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_output_plot.png')
    #plt.show()
    plt.clf()

def fixed_ROC_plot():
    print 'Entering fixed_ROC_plot'
    mu_values = [-2.000, -1.000, 0.000, 1.000, 2.000]

    for idx, mu in enumerate(mu_values):
        ROC = np.loadtxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %mu)
        AUC = np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %mu)
        xval = ROC[:,0]
        yval = ROC[:,1]
        plt.plot(xval, yval,
                    '.', 
                    label='$\mu_p=$%s (AUC=%0.2f)' %(mu, AUC),  
                    rasterized=True)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('1/Background efficiency')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig('plots/fixed_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_ROC_plot.png') 
    plt.clf()

def parameterized_training(iterations):
    print 'Entering parameterized_training'
    mu_complete = np.concatenate((np.loadtxt('data/training_data/traindata_mu_-1.000.dat'),
                                    np.loadtxt('data/training_data/traindata_mu_-1.000.dat'),
                                    np.loadtxt('data/training_data/traindata_mu_0.000.dat'),
                                    np.loadtxt('data/training_data/traindata_mu_1.000.dat'),
                                    np.loadtxt('data/training_data/traindata_mu_2.000.dat'),
                                    ))
    nn = Pipeline([
    ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
    ('neural network',
        Regressor(
            layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
            #learning_rate=0.01,
            n_iter=iterations,
            #learning_momentum=0.1,
            batch_size=10,
            learning_rule="nesterov",
            #valid_size=0.05,
            verbose=True,
            #debug=True
            ))])

    data = mu_complete
    training = data[:,0:2]
    target = data[:,2:]
    nn.fit(training, target)
    fit_score = nn.score(training, target)
    print 'score = %s' %fit_score

    outputs = nn.predict(training)
    output_data = np.vstack((training[:,0], target[:,0], outputs[:,0])).T
    np.savetxt('data/plot_data/param.dat', output_data, fmt='%f')
    pickle.dump(nn, open('data/param.pkl', 'wb'))


def parameterized_function(x, alpha, nn):
    traindata = np.array((x, alpha), ndmin=2)
    outputs   = nn.predict(traindata)
    return outputs[[0]]


def parameterized_runner():
    alpha_list = [-1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5]
    nn = pickle.load(open('data/param.pkl','rb'))
    for idx, alpha in enumerate(alpha_list):
        inputs = []
        predictions = []
        for idy in range(-500, 500, 1):
            outputs = parameterized_function(idy/100., alpha, nn)
            inputs.append(idy/100.)
            predictions.append(outputs[0])
        plt.plot(inputs, predictions, 'o')
    plt.xlabel('input')
    plt.ylabel('NN output')
    plt.xlim([-5,5])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.savefig('plots/interpolated_output.pdf', dpi=400)
    plt.savefig('plots/images/interpolated_output.png')


def ROC_plot_param(mu, fpr, tpr, roc_auc, plt_color):
    print "Plotting ROC curve for mu=%s" %mu
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'o', label='mu=%s (AUC=%0.2f)' %(mu, roc_auc), color=plt_color, linewidth=2, rasterized=True)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.grid(True)

if __name__ == '__main__':
    #for i in range(-20,30,10):
    #    gaussianData(i/10., 0.25, 50000)
    #plt_histogram()
    #fixed_training(20)
    #fixed_output_plot()
    #fixed_ROC_plot()
    #parameterized_training(20)
    #parameterized_runner()
