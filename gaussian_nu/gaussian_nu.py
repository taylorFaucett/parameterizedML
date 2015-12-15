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

def generate_nu_data():
    '''
    Adds a overall shift of nu to the gaussian generated in generate_gaussian, then outputs the data
    '''
    nu_values = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
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

def nu_histogram():
    '''
    
    '''

    print 'Entering nu_histogram'
    nu_values = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
    for idx, nu in enumerate(nu_values):
        data = np.loadtxt('data/training_data/traindata_nu_%0.3f.dat' %nu)
        bin_size   = 1000
        bin_width = 10.0/bin_size
        n, bins, patches = plt.hist(data[:,0],
                            bins=bin_size, histtype='stepfilled',
                            alpha=0.5, label='$\\nu=$%0.3f' %nu,
                            rasterized=True)
        plt.setp(patches)
    plt.ylabel('Number of events$/%0.3f x$' %bin_width)
    plt.xlabel('x')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1))
    plt.xlim([-2,2])
    #plt.ylim([0,10])
    plt.savefig('plots/histogram_gaussian.pdf', dpi=400)
    plt.savefig('plots/images/histogram_gaussian.png')
    #plt.show()
    plt.clf() 


def fixed_training(iterations):
    '''
    fixed_training uses SKLearn-NeuralNetwork and SciKit's Pipeline and min/max scaler
    to train a NN with training data generated from generate_data.
    '''

    print 'Entering fixed_training'
    nu_values = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]

    nn = Pipeline([
    ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
    ('neural network',
        Regressor(
            layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
            #learning_rate=0.01,
            n_iter=iterations,
            #learning_momentum=0.1,
            #batch_size=10,
            learning_rule="nesterov",
            #valid_size=0.05,
            verbose=True,
            #debug=True
            ))])
    #print nn

    for idx, nu in enumerate(nu_values):
        print 'Training on nu=%0.3f' %nu
        data = np.loadtxt('data/training_data/traindata_nu_%0.3f.dat' %nu)
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
    nu_values = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]

    for idx, nu in enumerate(nu_values):
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %nu)
        inputs = data[:,0]
        outputs = data[:,2]
        plt.plot(inputs, outputs, '.', label='$\\nu_f=$%0.1f' %nu, rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('input')
    plt.xlim([-2, 2])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig('plots/fixed_output_plot.pdf', dpi=400)
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
    nu_values = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]

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
    '''
    parameterized_training combines data produced from generate_data into one dataset. A NN
    is then trained with the data points and mean value (mu) as a secondary input which will
    later be used as a parameter during predictions. 
    '''

    print 'Entering parameterized_training'
    nu_complete = np.concatenate((np.loadtxt('data/training_data/traindata_nu_-1.000.dat'),
                                    #np.loadtxt('data/training_data/traindata_nu_-0.750.dat'),
                                    np.loadtxt('data/training_data/traindata_nu_-0.500.dat'),
                                    #np.loadtxt('data/training_data/traindata_nu_-0.250.dat'),
                                    np.loadtxt('data/training_data/traindata_nu_0.000.dat'),
                                    #np.loadtxt('data/training_data/traindata_nu_0.250.dat'),
                                    np.loadtxt('data/training_data/traindata_nu_0.500.dat'),
                                    #np.loadtxt('data/training_data/traindata_nu_0.750.dat'),
                                    np.loadtxt('data/training_data/traindata_nu_1.000.dat'),
                                    ))
    nn = Pipeline([
    ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
    ('neural network',
        Regressor(
            layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
            #learning_rate=0.01,
            n_iter=iterations,
            #learning_momentum=0.1,
            #batch_size=10,
            learning_rule="nesterov",
            #valid_size=0.05,
            #verbose=True,
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
    np.savetxt('data/plot_data/param.dat', output_data, fmt='%f')
    pickle.dump(nn, open('data/param.pkl', 'wb'))


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
    alpha_list = [-1.000, -0.750, -0.500, -0.250, 0.000, 0.250, 0.500, 0.750, 1.000]
    nn = pickle.load(open('data/param.pkl','rb'))

    for idx, alpha in enumerate(alpha_list):
        actual_data = np.loadtxt('data/training_data/traindata_nu_%0.3f.dat' %alpha)
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
    nu_values = [-1.000, -0.750, -0.500, -0.250, 0.000, 0.250, 0.500, 0.750, 1.000]

    for idx, nu in enumerate(nu_values):
        data = np.loadtxt('data/plot_data/param_%0.3f.dat' %nu)
        inputs = data[:,0]
        outputs = data[:,1]
        plt.plot(inputs, outputs, 'o', label='$\\nu_p=$%0.1f' %nu, rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('input')
    plt.xlim([-2, 2])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig('plots/parameterized_output_plot.pdf', dpi=400)
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
    nu_values = [-1.000, -0.750, -0.500, -0.250, 0.000, 0.250, 0.500, 0.750, 1.000]

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
    plt.ylabel('1/Background efficiency')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig('plots/param_ROC_plot.pdf', dpi=400)
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
    plt.savefig('plots/fixed_vs_parameterized_output_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_vs_parameterized_output_plot.png')
    #plt.show()
    plt.clf()

def parameterized_vs_fixed_output_plot():
    '''
    parameterized_vs_fixed_output_plot generates an array of points between 0-3000
    which are used to make predictions using the fixed and parameterized NN training
    in fixed_*.pkl and param_*.pkl
    '''

    print 'Entering parameterized_vs_fixed_output_plot'
    nu_values = [-1.000, -0.750, -0.500, -0.250, 0.000, 0.250, 0.500, 0.750, 1.000]

    for idx, nu in enumerate(nu_values):
        print 'Plotting mass mwwbb, nu=%0.3f' %nu
        fixed = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %nu)
        param = np.loadtxt('data/plot_data/param_%0.3f.dat' %nu)
        plt.plot(fixed[:,0], fixed[:,4],
                    '.',
                    color=colors[idx],
                    markevery=1000,
                    label='jes$_f$=%0.3f' %jes,
                    rasterized=True
                    )
        plt.plot(param[:,0], param[:,2],
                    'o',
                    color=colors[idx],
                    markevery=2000,
                    alpha=0.3,
                    label='jes$_p$=%0.3f' %jes,
                    rasterized=True
                    )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.xlim([0,3000])
    plt.ylim([0,1])
    plt.xlabel('$m_{WWbb}$')
    plt.ylabel('NN output')
    plt.savefig('plots/parameterized_vs_fixed_output_plot_mwwbb.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_vs_fixed_output_plot_mwwbb.png')
    plt.clf()

    for idx, jes in enumerate(jes_list):
        print 'Plotting mass mjj, jes=%0.3f' %jes
        fixed = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %jes)
        param = np.loadtxt('data/plot_data/param_%0.3f.dat' %jes)
        plt.plot(fixed[:,1], fixed[:,4],
                        '.',
                        color=colors[idx],
                        markevery=1000,
                        label='jes$_f$=%0.3f' %jes,
                        rasterized=True
                        )
        plt.plot(param[:,1], param[:,2],
                        'o',
                        color=colors[idx],
                        markevery=2000,
                        alpha=0.3,
                        label='jes$_p$=%0.3f' %jes,
                        rasterized=True
                        )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.xlim([0,250])
    plt.ylim([0,1])
    plt.xlabel('$m_{jj}$')
    plt.ylabel('NN output')
    plt.savefig('plots/parameterized_vs_fixed_output_plot_mjj.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_vs_fixed_output_plot_mjj.png')
    plt.clf()

if __name__ == '__main__':
    '''
    Generate data
    '''
    #generate_gaussian(0, 0.05, 10000)
    #generate_nu_data()
    nu_histogram()

    '''
    Fixed training and plots
    '''
    #fixed_training(50)
    #fixed_output_plot()
    #fixed_ROC_plot()

    '''
    Parameterized training and plots
    '''
    #parameterized_training(50)
    #parameterized_runner()
    #parameterized_output_plot()
    #parameterized_ROC_plot()