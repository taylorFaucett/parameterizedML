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
from array import *
from sklearn.metrics import roc_curve, auc
from sknn.mlp import Regressor, Layer
import pickle
import glob
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

param_colors = ['yellow', 'green', 'brown', 'red', 'orange', 'cyan', 'black']
fixed_colors = ['blue', 'green', 'red', 'cyan', 'magenta']

def generate_gaussian(MU, SIG, numTrain):
    '''
    generate_gaussian uses a RooWorkspace to generate data points with a gaussian and
    flat distribution. A single gaussian is generated and seperate data files with
    different nu parameter values is added separately
    '''

    print 'Entering generate_data'
    # Initialize ROOTs RooWorkspace
    w = ROOT.RooWorkspace('w')

    # Generate Gaussian signals
    print "Generating Gaussians PDFs"
    w.factory('Gaussian::g(x[-6,6],mu[%0.3f,-3,3],sigma[%0.3f, 0, 2])' %(MU, SIG))

    x      = w.var('x')
    mu     = w.var('mu')
    pdf    = w.pdf('model')
    sigpdf = w.pdf('g')
    bkgpdf = w.pdf('e')
    traindata  = np.zeros((numTrain, 2))
    targetdata = np.zeros(numTrain)

    sigdata = sigpdf.generate(ROOT.RooArgSet(x), numTrain)

    for i in range(numTrain):
        traindata[i,0] = sigdata.get(i).getRealValue('x')
        traindata[i,1] = MU
        targetdata[i] = 1

    np.savetxt("data/training_data/traindata_mu_%0.3f.dat" %MU, np.column_stack((traindata, targetdata)), fmt='%f')

def generate_nu_data():
    '''
    Adds a overall shift of nu to the gaussian generated in generate_gaussian, then outputs the data
    '''
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
    for idx, nu in enumerate(nu_values):
        data = np.loadtxt('data/training_data/traindata_mu_0.000.dat')
        training = np.array(data[:,0])
        training = training[:]+nu
        target = data[:,2]
        mu_array = data[:,1]
        nu_array = np.zeros(len(training))
        nu_array.fill(nu)
        np.savetxt("data/training_data/traindata_nu_%0.3f.dat" %nu, np.column_stack((training, target, mu_array, nu_array)), fmt='%f')
    print training

def generate_background(numTrain):
    '''
    generate_gaussian uses a RooWorkspace to generate data points with a gaussian and
    flat distribution. A single gaussian is generated and seperate data files with
    different nu parameter values is added separately
    '''

    print 'Entering generate_data'
    # Initialize ROOTs RooWorkspace
    w = ROOT.RooWorkspace('w')

    # Generate Gaussian signals
    print "Generating Gaussians PDFs"
    w.factory('Uniform::g(x[-6,6])')

    x      = w.var('x')
    mu     = w.var('mu')
    pdf    = w.pdf('model')
    sigpdf = w.pdf('g')
    bkgpdf = w.pdf('e')
    traindata  = np.zeros((numTrain, 2))
    targetdata = np.zeros(numTrain)

    sigdata = sigpdf.generate(ROOT.RooArgSet(x), numTrain)

    for i in range(numTrain):
        traindata[i,0] = sigdata.get(i).getRealValue('x')
        traindata[i,1] = 0.000
        targetdata[i] = 0

    np.savetxt("data/training_data/traindata_bkg.dat", np.column_stack((traindata, targetdata)), fmt='%f')

def nu_histogram():
    print 'Entering nu_histogram'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
    for idx, nu in enumerate(nu_values):
        sig = np.loadtxt('data/training_data/traindata_nu_%0.3f.dat' %nu)
        print sig
        bkg = np.loadtxt('data/training_data/traindata_bkg.dat')
        bkg_nu = np.zeros(len(bkg))
        bkg_nu.fill(nu)
        bkg = np.vstack((bkg[:,0], bkg[:,1], bkg[:,2], bkg_nu)).T
        print sig
        print bkg
        data = np.concatenate((sig, bkg))
        bin_size   = 200
        bin_width = 10.0/bin_size
        n, bins, patches = plt.hist(data[:,0],
                            bins=bin_size, histtype='stepfilled',
                            alpha=0.5, label='$\\nu=$%0.3f' %nu,
                            rasterized=True)
        plt.setp(patches)
        np.savetxt("data/training_data/traindata_complete_nu_%0.3f.dat" %nu, data, fmt='%f')
    plt.ylabel('Number of events$/%0.3f x$' %bin_width)
    plt.xlabel('x')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1))
    plt.xlim([-4,4])
    #plt.ylim([0,10])
    plt.savefig('plots/histogram_gaussian.pdf', dpi=200)
    plt.savefig('plots/images/histogram_gaussian.png')
    #plt.show()
    plt.clf()


def fixed_training(iterations):
    '''
    fixed_training uses SKLearn-NeuralNetwork and SciKit's Pipeline and min/max scaler
    to train a NN with training data generated from generate_data.
    '''

    print 'Entering fixed_training'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]

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
    #print nn

    for idx, nu in enumerate(nu_values):
        print 'Training on nu=%0.3f' %nu
        data = np.loadtxt('data/training_data/traindata_complete_nu_%0.3f.dat' %nu)
        training = data[:,0:1]
        target = data[:,1:2]
        print training
        print target
        nn.fit(training, target)
        fit_score = nn.score(training, target)
        print 'score = %s' %fit_score

        outputs = nn.predict(training)
        output_data = np.vstack((training[:,0], target[:,0], outputs[:,0])).T
        np.savetxt('data/plot_data/fixed_%0.3f.dat' %nu, output_data, fmt='%f')

        actual = target
        predictions = outputs
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        ROC_plot = np.vstack((fpr, tpr)).T
        ROC_AUC = [auc(fpr, tpr)]
        np.savetxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %nu, ROC_plot, fmt='%f')
        np.savetxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %nu, ROC_AUC)
        pickle.dump(nn, open('data/fixed_%0.3f.pkl' %nu, 'wb'))

def fixed_output_plot():
    '''
    fixed_output_plot plots the input and output data from the NN trained in fixed_training.
    Plotting is handled separately so that minor changes to the plot don't require re-running
    intensive NN training.
    '''

    print 'Entering fixed_output_plot'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]

    for idx, nu in enumerate(nu_values):
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %nu)
        inputs = data[:,0]
        outputs = data[:,2]
        plt.plot(inputs, outputs, '.', label='$\\nu_f=$%0.1f' %nu, rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('input')
    plt.xlim([-3, 3])
    plt.ylim([0.0, 1.1])
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig('plots/fixed_output_plot.pdf', dpi=200)
    plt.savefig('plots/images/fixed_output_plot.png')
    #plt.show()
    plt.clf()

def fixed_ROC_plot():
    '''
    fixed_ROC_plot plots the ROC data from the NN trained in fixed_training.
    Plotting is handled separately so that minor changes to the plot don't require re-running
    intensive NN training.
    '''

    print 'Entering fixed_ROC_plot'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]

    for idx, nu in enumerate(nu_values):
        ROC = np.loadtxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %nu)
        AUC = np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %nu)
        xval = ROC[:,0]
        yval = ROC[:,1]
        plt.plot(xval, yval,
                    '-',
                    label='$\\nu_f=$%0.1f (AUC=%0.3f)' %(nu, AUC),
                    rasterized=True)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('Signal efficiency')
    plt.xlabel('Background efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig('plots/fixed_ROC_plot.pdf', dpi=200)
    plt.savefig('plots/images/fixed_ROC_plot.png')
    plt.clf()

def parameterized_training(iterations):
    '''
    parameterized_training combines data produced from generate_data into one dataset. A NN
    is then trained with the data points and mean value (mu) as a secondary input which will
    later be used as a parameter during predictions.
    '''

    print 'Entering parameterized_training'
    nu_complete = np.concatenate((#np.loadtxt('data/training_data/traindata_complete_nu_-2.000.dat'),
                                    #np.loadtxt('data/training_data/traindata_complete_nu_-1.500.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_-1.000.dat'),
                                    #np.loadtxt('data/training_data/traindata_complete_nu_-0.500.dat'),
                                    #np.loadtxt('data/training_data/traindata_complete_nu_0.000.dat'),
                                    #np.loadtxt('data/training_data/traindata_complete_nu_0.500.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_1.000.dat'),
                                    #np.loadtxt('data/training_data/traindata_complete_nu_1.500.dat'),
                                    #np.loadtxt('data/training_data/traindata_complete_nu_2.000.dat')
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

    data = nu_complete
    training = np.vstack((data[:,0], data[:,3])).T
    target = data[:,1:2]
    print training
    print target
    nn.fit(training, target)
    fit_score = nn.score(training, target)
    print 'score = %s' %fit_score

    outputs = nn.predict(training)
    output_data = np.vstack((training[:,0], training[:,1], target[:,0], outputs[:,0])).T
    np.savetxt('data/plot_data/param_n1_p1.dat', output_data, fmt='%f')
    pickle.dump(nn, open('data/param_n1_p1.pkl', 'wb'))


def parameterized_function(x, alpha, nn):
    '''
    parameterized_function is a 2 variable function for making predictions with the
    mean value as a parameter. Given an input value and a mean value to interpolate
    with, parameterized_function will make a prediction of the score based on the
    parameterized NN generated in parameterized_training.

    Note: The pickled NN is initialized outside of the function for the purposes of
    processing speed.
    '''
    traindata = np.array((x, alpha), ndmin=2)
    outputs   = nn.predict(traindata)
    return outputs[[0]]


def parameterized_runner():
    '''
    parameterized_runner takes values from generate_data and plugs them into
    parameterized_function for both given mean values (mu=-1, 0, 1) and
    interpolations on values (mu=-1.5, -0.5, +0.5, +1.5). Input, output and
    ROC/AUC data is calculated using the sample input data and the predictions
    from parameterized_function.
    '''

    print 'Entering parameterized_runner'
    alpha_list = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
    nn = pickle.load(open('data/param_n1_p1.pkl','rb'))

    for idx, alpha in enumerate(alpha_list):
        actual_data = np.loadtxt('data/training_data/traindata_complete_nu_%0.3f.dat' %alpha)
        actual_input = actual_data[:,0]
        actual_target = actual_data[:,1]
        inputs = []
        predictions = []
        for idy in range(len(actual_input)):
            outputs = parameterized_function(actual_input[idy]/1., alpha, nn)
            inputs.append(actual_input[idy]/1.)
            predictions.append(outputs[0][0])
        output_data = np.vstack((inputs, predictions)).T
        np.savetxt('data/plot_data/param_%0.3f.dat' %alpha, output_data, fmt='%f')

        actual = actual_target
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        ROC_plot = np.vstack((fpr, tpr)).T
        ROC_AUC = [auc(fpr, tpr)]
        np.savetxt('data/plot_data/ROC/param_ROC_%0.3f.dat' %alpha, ROC_plot, fmt='%f')
        np.savetxt('data/plot_data/AUC/param_ROC_AUC_%0.3f.dat' %alpha, ROC_AUC)

def parameterized_output_plot():
    '''
    parameterized_output_plot plots the input and output data from the NN trained in parameterized_training.
    Plotting is handled separately so that minor changes to the plot don't require re-running
    intensive NN training.
    '''

    print 'Entering parameterized_output_plot'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]

    for idx, nu in enumerate(nu_values):
        data = np.loadtxt('data/plot_data/param_%0.3f.dat' %nu)
        inputs = data[:,0]
        outputs = data[:,1]
        plt.plot(inputs, outputs, 'o', label='$\\nu_p=$%0.1f' %nu, rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('input')
    plt.xlim([-3, 3])
    plt.ylim([0.0, 1.1])
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig('plots/parameterized_output_plot.pdf', dpi=200)
    plt.savefig('plots/images/parameterized_output_plot.png')
    #plt.show()
    plt.clf()

def parameterized_ROC_plot():
    '''
    parameterized_ROC_plot plots the ROC data from the NN trained in parameterized_training.
    Plotting is handled separately so that minor changes to the plot don't require re-running
    intensive NN training.
    '''

    print 'Entering parameterized_ROC_plot'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]

    for idx, nu in enumerate(nu_values):
        ROC = np.loadtxt('data/plot_data/ROC/param_ROC_%0.3f.dat' %nu)
        AUC = np.loadtxt('data/plot_data/AUC/param_ROC_AUC_%0.3f.dat' %nu)
        xval = ROC[:,0]
        yval = ROC[:,1]
        plt.plot(xval, yval,
                    '-',
                    label='$\\nu_p=$%0.1f (AUC=%0.3f)' %(nu, AUC),
                    rasterized=True)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('Signal efficiency')
    plt.xlabel('Background efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig('plots/param_ROC_plot.pdf', dpi=200)
    plt.savefig('plots/images/param_ROC_plot.png')
    plt.clf()

def fixed_vs_param_output_plot():
    '''
    fixed_vs_param_output_plot takes plot data for both fixed and parameterized training
    for comparison purposes.
    '''

    print 'Entering fixed_vs_param_output_plot'
    fixed_mu = [-2.000, -1.000, 0.000, 1.000, 2.000]
    param_mu = [-1.500, -1.000, -0.500, 0.000, 0.500, 1.000, 1.500]

    for idx, mu in enumerate(fixed_mu):
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %mu)
        inputs = data[:,0]
        outputs = data[:,2]
        plt.plot(inputs, outputs,
                    '.',
                    markevery= 1,
                    color=fixed_colors[idx],
                    label='$\mu_f=$%0.1f' %mu,
                    rasterized=True)
    for idx, mu in enumerate(param_mu):
        data = np.loadtxt('data/plot_data/param_%0.3f.dat' %mu)
        inputs = data[:,0]
        outputs = data[:,1]
        plt.plot(inputs, outputs,
                    'o',
                    alpha=0.5,
                    markevery=50,
                    color=param_colors[idx],
                    label='$\mu_p=$%0.1f' %mu,
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('input')
    plt.xlim([-4, 4])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=10)
    plt.savefig('plots/fixed_vs_parameterized_output_plot.pdf', dpi=200)
    plt.savefig('plots/images/fixed_vs_parameterized_output_plot.png')
    #plt.show()
    plt.clf()




def fixed_comparison_ROC(fixed):
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
    #fixed = 0.800
    nn = pickle.load(open('data/fixed_%0.3f.pkl' %fixed, 'rb'))
    print nn
    for idx, nu in enumerate(nu_values):
        print 'loading data for nu=%0.3f' %nu
        input_data = np.loadtxt('data/training_data/traindata_complete_nu_%0.3f.dat' %nu)
        size = len(input_data)
        input_data = np.concatenate((input_data[:size], input_data[-size:]), axis=0)
        training_data = input_data[:,0:1]
        actual = input_data[:,1]
        outputs = nn.predict(training_data)
        outputs = outputs.reshape((1,len(outputs)))
        predictions = outputs[0]
        fpr, tpr, thresholds = roc_curve(actual, predictions)
        AUC = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                    '-',
                    label='nu$_{f_{%0.3f}}$=%0.3f (AUC=%0.6f)' %(fixed, nu, AUC),
                    markevery=100,
                    rasterized=True)
        ROC_output = np.vstack((fpr, tpr)).T
        np.savetxt('data/ROC_raw/gaussian_fixed_%0.3f_nu_%0.3f.dat' %(fixed, nu), ROC_output, fmt='%f')
    plt.plot([0,1],[0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('Signal efficiency')
    plt.xlabel('Background efficiency')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig('plots/fixed_%0.3f_ROC.pdf' %fixed, dpi=200)
    plt.savefig('plots/images/fixed_%0.3f_ROC.png' %fixed)
    plt.clf()

def fixed_analysis_data():
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
    nn = pickle.load(open('data/fixed_0.000.pkl', 'rb'))
    for idx, nu in enumerate(nu_values):
        print 'processing nu=%0.3f' %nu

        data = np.loadtxt('data/training_data/traindata_complete_nu_%0.3f.dat' %nu)
        inputs = np.vstack((data[:,0], data[:,3])).T
        actuals = data[:,1:2]
        X = inputs[:,0]
        NU = inputs[:,1]
        outputs = nn.predict(inputs[:,0:1])
        predictions = outputs.reshape((1, len(NU)))
        nu_gen = np.zeros([1, len(NU)])
        nu_eval = np.zeros([1, len(NU)])
        nu_gen.fill(nu)
        nu_eval.fill(0.000000)
        data = np.vstack((actuals[:,0], X, NU, predictions, nu_gen, nu_eval)).T
        np.savetxt('data/analysis_data/fixed_%0.3f.dat' %nu, data, fmt='%f')

    output_data = np.concatenate((np.loadtxt('data/analysis_data/fixed_-2.000.dat'),
                                    np.loadtxt('data/analysis_data/fixed_-1.500.dat'),
                                    np.loadtxt('data/analysis_data/fixed_-1.000.dat'),
                                    np.loadtxt('data/analysis_data/fixed_-0.500.dat'),
                                    np.loadtxt('data/analysis_data/fixed_0.000.dat'),
                                    np.loadtxt('data/analysis_data/fixed_0.500.dat'),
                                    np.loadtxt('data/analysis_data/fixed_1.000.dat'),
                                    np.loadtxt('data/analysis_data/fixed_1.500.dat'),
                                    np.loadtxt('data/analysis_data/fixed_2.000.dat')),
                                    axis=0)

    np.savetxt('data/analysis_data/fixed.csv', output_data, fmt='%f', delimiter=',')

    files = glob.glob('data/analysis_data/fixed_*.dat')
    for idx, file in enumerate(files):
        os.remove(file)

def parameterized_analysis_data():
    print 'Entering parameterized_analysis_data'
    nu_values = [-2.00, -1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50, 2.00]
    nu_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    nu_complete = np.concatenate((np.loadtxt('data/training_data/traindata_complete_nu_-2.000.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_-1.500.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_-1.000.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_-0.500.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_0.000.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_0.500.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_1.000.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_1.500.dat'),
                                    np.loadtxt('data/training_data/traindata_complete_nu_2.000.dat')
                                    ))

    nu_val_list = nu_values

    label = []
    XX = []
    nu = []
    NN_output = []
    nu_gen = []
    nu_eval = []
    inu_list = []
    inu_true = []
    inu_param = []
    for i in range(len(nu_val_list)):
        inu_list.append(i)
    print nu_val_list
    for idx, nu in enumerate(nu_values):
        print 'processing nu=%0.3f' %nu
        sig = np.loadtxt('data/training_data/traindata_nu_%0.3f.dat' %nu)
        print sig
        bkg = np.loadtxt('data/training_data/traindata_bkg.dat')
        bkg_nu = np.zeros(len(bkg))
        bkg_nu.fill(nu)
        bkg = np.vstack((bkg[:,0], bkg[:,1], bkg[:,2], bkg_nu)).T
        print sig
        print bkg
        data = np.concatenate((sig, bkg))
        size = len(data[:,0])
        nn = pickle.load(open('data/param_complete.pkl', 'rb'))
        inputs = np.vstack((data[:,0], data[:,2])).T
        actuals = data[:,1]
        X=data[:,0]
        for x in range(0,size):
            for y in range(len(nu_val_list)):
                outputs = parameterized_function(X[x]/1., nu_val_list[y], nn)
                label.append(actuals[x]/1.)
                XX.append(X[x]/1.)
                NN_output.append(outputs[0][0])
                nu_gen.append(nu)
                nu_eval.append(nu_val_list[y])
                inu_true.append(nu_index[idx])
                inu_param.append(inu_list[y])             
        #print label
        #print XX
        #print NN_output
        #print nu_gen
        #print nu_eval
        #print inu_true
        #print inu_param
    data = np.vstack((label, XX, NN_output, nu_gen, nu_eval, inu_true, inu_param)).T
    np.savetxt('data/analysis_data/parameterized.csv', data, fmt='%f', delimiter=',')

if __name__ == '__main__':
    '''
    Generate data
    '''
    #generate_gaussian(0, 0.25, 50000)
    #generate_nu_data()
    #generate_background(100000)
    #nu_histogram()

    '''
    Fixed training and plots
    '''
    #fixed_training(50)
    #fixed_output_plot()
    #fixed_ROC_plot()

    #fixed_comparison_ROC(-2.000)
    #fixed_comparison_ROC(-1.000)
    #fixed_comparison_ROC(0.000)
    #fixed_comparison_ROC(1.000)
    #fixed_comparison_ROC(2.000)

    '''
    Parameterized training and plots
    '''
    parameterized_training(20)
    parameterized_runner()
    parameterized_output_plot()
    parameterized_ROC_plot()

    #fixed_analysis_data()
    #parameterized_analysis_data()
