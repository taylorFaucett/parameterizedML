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
import glob
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sknn.mlp import Regressor, Layer

''' 
The standard set of matplotlib colors are used in multiple functions so they
are defined globally here.
'''
colors  = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']


def file_runner():
    '''
    file_runner's only responsibility is to pull in files from the root_files directory
    and pass them to the file_generate command. 
    '''

    print 'Entering file_runner'
    sig_files = glob.iglob('data/root_files/xttbar_*.root')
    bkg_files = glob.iglob('data/root_files/smttbar.root')
    for data in sig_files:
    	file_generate(data, 1.000000)
    for data in bkg_files:
        file_generate(data, 0.000000)

def flat_bkg(bkgNum, low, high):
    '''
    flat_bkg creates a data set for a flat background. You can specify the number of data points
    with bkgNum and the domain of the data with low and high. Additionally, you can append mx
    values to the output file.
    '''

    print 'Entering flat_bkg'
    print 'Genterating a flat background with %s data points betwee %s-%s' %(bkgNum, low, high)
    mx_values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    w = ROOT.RooWorkspace('w')

    w.factory('Uniform::f(x[%s,%s])' %(low, high))

    # Define variables
    x      = w.var('x')
    bkgpdf = w.pdf('f')

    # Fill training_data, testdata and testdata1
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
    '''
    file_generate is a file runner for the root_export function. Values from a root tree 
    are scanned with the root_export function and then added to a numpy array before being
    saved as a .dat file. The target value is used to determine if the data set is a signal
    or background.
    '''

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
    '''
    root_export uses pyROOT to scan invidual values from a selected TTree and TLeaf. 
    The function returns an array of all values.

    Note: GetEntriesFast() is preferred over GetEntires() as it correctly returns a
    floating point number in case of a value of 0.0 rather than, in the case of GetEntries(),
    returning a NULL. 
    '''

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
    '''
    file_concatenater pulls in the seperate signal and background .dat files and 
    combines them prior to analysis. The concatenated are output into the directory
    of the same name.
    '''

    print 'Entering file_concatenater'
    sig_dat = glob.iglob('data/root_export/sig_mx_*.dat')
    bkg_dat = glob.iglob('data/root_export/bkg_mx_*.dat')
    for signal, background in zip(sig_dat, bkg_dat):
        sig = np.loadtxt(signal)
        bkg = np.loadtxt(background)
        data_complete = np.concatenate((sig, bkg), axis=0)
        np.savetxt('data/concatenated/ttbar_mx_%0.0f.dat' %sig[0,1], data_complete, fmt='%f')


'''
Fixed Training and Plots 
'''

def fixed_training():
    '''
    fixed_training takes concatenated data and splits the data set into training_data 
    (i.e. [mx_value, mx], e.g. [[532.1, 500], [728.4, 500]]) and target_data (i.e. [1, 1, ... 0, 0]).
    The input undergoes pre-processing via SKLearns pipeline and then a NN is trained using the 
    SKLearnNN wrapper which processes the data using Theano/PyLearn2. NN learning parameters 
    (e.g. learning_rate, n_iter, etc) are selected before hand. target_data from the inputs
    are then used along with predictions to calculate the Receiver Operating Characteristic (ROC) curve
    and Area Under the Curve (AUC). 
    '''

    print 'Entering fixed_training'

    # Training input files
    file_list = ['data/concatenated/ttbar_mx_500.dat',
                    'data/concatenated/ttbar_mx_750.dat',
                    'data/concatenated/ttbar_mx_1000.dat',
                    'data/concatenated/ttbar_mx_1250.dat',
                    'data/concatenated/ttbar_mx_1500.dat']
    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network',
            Regressor(
                layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                learning_rate=0.01,
                #n_stable=1,
                #f_stable=100,
                n_iter=100,
                #learning_momentum=0.1,
                batch_size=10,
                learning_rule="nesterov",
                #valid_size=0.05,
                #verbose=True,
                #debug=True
                ))])

    for file in file_list:
        input_data = np.loadtxt(file)

        training_data = input_data[:,0:2]
        target_data = input_data[:,2]
        
        mx = training_data[0,1]
        
        print 'Processing fixed training on mu=%0.0f' %mx

    	nn.fit(training_data, target_data)

    	fit_score = nn.score(training_data, target_data)
    	print 'score = %s' %fit_score

    	outputs = nn.predict(training_data)
    	outputs = outputs.reshape((1,len(outputs)))

    	output_data = np.vstack((training_data[:,0], outputs)).T
    	np.savetxt('data/plot_data/fixed_%0.0f.dat' %mx, output_data, fmt='%f')

    	actual = target_data
    	predictions = outputs[0]
    	fpr, tpr, thresholds = roc_curve(actual, predictions)
    	ROC_plot = np.vstack((fpr, tpr)).T
    	ROC_AUC = [auc(fpr, tpr)]
    	np.savetxt('data/plot_data/ROC/fixed_ROC_%0.0f.dat' %mx, ROC_plot, fmt='%f')
    	np.savetxt('data/plot_data/AUC/fixed_ROC_AUC_%0.0f.dat' %mx, ROC_AUC)

    	pickle.dump(nn, open('data/pickle/fixed_%0.0f.pkl' %mx, 'wb'))

def fixed_training_plot(): 
    '''
    fixed_training_plot is no longer necessary and has been replaced by
    parameterized_vs_fixed_output_plot. Initially this took prediction data 
    from fixed_training and plotted the values. However, it is more convenient to pickle
    the NN training from fixed training and then run predictions separately rather than
    generate prediction outputs during training. This is left here because I'm a hoarder.
    '''

    print 'Entering fixed_training_plot'
    file_list = [np.loadtxt('data/plot_data/fixed_500.dat'),
                np.loadtxt('data/plot_data/fixed_750.dat'),
                np.loadtxt('data/plot_data/fixed_1000.dat'),
                np.loadtxt('data/plot_data/fixed_1250.dat'),
                np.loadtxt('data/plot_data/fixed_1500.dat')
                ]
    mx = [500, 750, 1000, 1250, 1500]
    for idx, file in enumerate(file_list):
        plt.plot(file[:,0], file[:,1],
                    '.',
                    color=colors[idx], 
                    #linewidth = 2,
                    #alpha=1, 
                    #markevery = 1, 
                    #markersize = 1,
                    label='$\mu_f=$%s' %mx[idx], 
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('m$_{WWbb}$ [GeV]')
    plt.xlim([0,3000])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixed_training_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_training_plot.png')
    plt.clf()

def fixed_ROC_plot():
    '''
    fixed_ROC_plot takes in ROC and AUC values processed during training in fixed_training
    and plots the ROC curve. This will be deprecated in the future, for the same reason as
    fixed_training_plot, so that AUC and ROC values will be calculated and plotted separately
    from the training time.
    '''

    print "Entering fixed_ROC_plot"
    files = [np.loadtxt('data/plot_data/ROC/fixed_ROC_500.dat'),
                np.loadtxt('data/plot_data/ROC/fixed_ROC_750.dat'),
                np.loadtxt('data/plot_data/ROC/fixed_ROC_1000.dat'),
                np.loadtxt('data/plot_data/ROC/fixed_ROC_1250.dat'),
                np.loadtxt('data/plot_data/ROC/fixed_ROC_1500.dat')]
    AUC = [np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_500.dat'),
            np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_750.dat'),
            np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_1000.dat'),
            np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_1250.dat'),
            np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_1500.dat')]
    mx = [500, 750, 1000, 1250, 1500]
    for idx, files in enumerate(files):
        plt.plot(files[:,0], files[:,1],
                    '-', 
                    linewidth = 2,
                    color=colors[idx], 
                    alpha=1,  
                    label='$\mu_f=$%s (AUC=%0.2f)' %(mx[idx], AUC[idx]), 
                    rasterized=True)
    plt.plot([0,1],[0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('1/Background efficiency')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixed_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_ROC_plot.png')
    plt.clf()



''' 
Parameterized Training and Plots 
'''

def parameterized_training():
    '''
    parameterized_training Trains a NN with multiple signals. In each case, one signal is 
    excluded (e.g. mwwbb_complete500 trains for all signals excluding the one at mu=500). 
    A seperate NN is trained for each of these scenarios and then pickled in an appropriately 
    labeled file (e.g. Training excluding mu=500 is pickled as param_500.pkl)
    '''

    print 'Entering parameterized_training'

    mwwbb_complete500 = np.concatenate((
                        np.loadtxt('data/concatenated/ttbar_mx_750.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1250.dat'),
                        np.loadtxt('data/concatenated/ttbar_mx_1500.dat')),
                        axis=0)

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

    mwwbb_complete1500 = np.concatenate((
                np.loadtxt('data/concatenated/ttbar_mx_500.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_750.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1250.dat')),
                axis=0)

    training_list = [mwwbb_complete500[:,0:2],
                        mwwbb_complete750[:,0:2],
                        mwwbb_complete1000[:,0:2],
                        mwwbb_complete1250[:,0:2],
                        mwwbb_complete1500[:,0:2]
                    ]

    target_list = [mwwbb_complete500[:,2],
                    mwwbb_complete750[:,2],
                    mwwbb_complete1000[:,2],
                    mwwbb_complete1250[:,2],
                    mwwbb_complete1500[:,2]
                    ]

    mx_list = [500, 750, 1000, 1250, 1500]

    for idx, (training_data, target_data, mx) in enumerate(zip(training_list, target_list, mx_list)):
    	print 'Parameterized training on all signals except for mu=%s' %mx
        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network',
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    n_iter=100,
                    #n_stable=1,
                    #f_stable=0.001,
                    #learning_momentum=0.1,
                    batch_size=10,
                    learning_rule="nesterov",
                    #valid_size=0.05,
                    #verbose=True,
                    #debug=True
                    ))])

        nn.fit(training_data, target_data)

        fit_score = nn.score(training_data, target_data)
        print 'score = %s' %fit_score
        
        outputs = nn.predict(training_data)
        #outputs = outputs.reshape((1, len(outputs)))
        #param_plot = np.vstack((training_data[:,0], outputs)).T
        #np.savetxt('data/plot_data/param_%s.dat' %mx[idx], param_plot, fmt='%f')
        pickle.dump(nn, open('data/pickle/param_%s.pkl' %mx, 'wb'))

def parameterized_function(x, alpha, nn):
    '''
    parameterized_function acts as the function with an additional parameter alpha. 
    This is used in parameterized_function_runner as the function which interpolates 
    signals at alpha values. For example, a NN trained at mu=500, 750, 1250 and 1500
    can use the parameterized_function with a specificied value of alpha=1000 to 
    interpolate the curve, ROC and AUC values at mu=1000 despite not having been trained
    for that location. 
    '''

    training_data = np.array((x, alpha), ndmin=2)
    outputs   = nn.predict(training_data)
    return outputs[[0]]

def parameterized_function_runner():
    '''
    parameterized_function_runner takes the NN training from parameterized_training and
    calculates the outputs, and ROC/AUC from sample inputs. In each case of mu=500, 750,
    1000, 1250, 1500 the prediction of alpha is made using the NN which excluded that signal.
    For example, when a prediction is made with the parameter alpha=500, the prediction is made
    using the NN trained only at mu=750, 1000, 1250, 1500. Similarly, a prediction with the 
    parameter alpha=750 is made with the NN trained at mu=500, 1000, 1250, 1500. 
    '''

    print 'Entering parameterized_function_runner'
    alpha_list = [500, 
                    750, 
                    1000, 
                    1250, 
                    1500]

    file_list = [np.loadtxt('data/concatenated/ttbar_mx_500.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_750.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1000.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1250.dat'),
                np.loadtxt('data/concatenated/ttbar_mx_1500.dat')]

    for idx, (file, alpha) in enumerate(zip(file_list, alpha_list)):
        size = len(file[:,0])
        print 'processing using: data/pickle/param_%0.0f.pkl' %alpha
        nn = pickle.load(open('data/pickle/param_%0.0f.pkl' %alpha, 'rb'))
        inputs = file[:,0]
        actuals = file[:,2]
        predictions = []
        for x in range(0,size):
            outputs = parameterized_function(inputs[x]/1., alpha, nn)
            predictions.append(outputs[0][0])
            #print 'Percent: %0.3f' %((100.*x)/size)
        data = np.vstack((inputs, predictions)).T
        np.savetxt('data/plot_data/param_%0.0f.dat' %alpha, data, fmt='%f')
        fpr, tpr, thresholds = roc_curve(actuals, predictions)
        roc_auc = [auc(fpr, tpr)]

        roc_data = np.vstack((fpr, tpr)).T
        np.savetxt('data/plot_data/ROC/param_ROC_%0.0f.dat' %alpha, roc_data, fmt='%f')
        np.savetxt('data/plot_data/AUC/param_ROC_AUC_%0.0f.dat' %alpha, roc_auc)

def parameterized_training_plot(): 
    '''
    parameterized_training_plot plots the output values generated during 
    parameterized_function_runner.
    '''

    print 'Entering parameterized_training_plot'
    files = ['data/plot_data/param_500.dat',
                'data/plot_data/param_750.dat',
                'data/plot_data/param_1000.dat',
                'data/plot_data/param_1250.dat',
                'data/plot_data/param_1500.dat']

    mx = [500, 750, 1000, 1250, 1500]
    for idx, file in enumerate(files):
        data = np.loadtxt(file)
        plt.plot(data[:,0], data[:,1],
                    'o', 
                    color=colors[idx], 
                    alpha=0.5,
                    markevery = 100, 
                    label='$\mu_p=$%s' %mx[idx], 
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('m$_{WWbb}$ [GeV]')
    plt.xlim([250,3000])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/parameterized_training_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_training_plot.png')
    plt.clf()


def parameterized_ROC_plot():
    '''
    parameterized_ROC_plot plots the ROC and AUC values generated during
    parameterized_function_runner
    '''

    print 'Entering parameterized_ROC_plot'
    param_files = [np.loadtxt('data/plot_data/ROC/param_ROC_500.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_750.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_1000.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_1250.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_1500.dat')
                ]

    AUC_param = [np.loadtxt('data/plot_data/AUC/param_ROC_AUC_500.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_750.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_1000.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_1250.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_1500.dat')
                    ]

    mx = [500, 750, 1000, 1250,1500]

    for i in range(len(param_files)):
        plt.plot(param_files[i][:,0], param_files[i][:,1], 'o', markerfacecolor=colors[i],
                    alpha=0.5, markevery=2000, label='$\mu_p=$%s (AUC=%0.2f)' %(mx[i], AUC_param[i]),  
                    rasterized=True)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('1/Background efficiency')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right', bbox_to_anchor=(1.10, 0))
    plt.grid(True)
    plt.savefig('plots/parameterized_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_ROC_plot.png')  
    plt.clf()



'''
Comparison Training and Plots
'''

def parameterized_vs_fixed_output_plot():
    '''
    parameterized_vs_fixed_output_plot generates an array of points between 0-3000
    which are used to make predictions using the fixed and parameterized NN training
    in fixed_*.pkl and param_*.pkl
    '''

    print 'Entering parameterized_vs_fixed_output_plot'
    mx = [500, 750, 1000, 1250, 1500]
    size = 3000
    for i in range(len(mx)):
        inputs = np.zeros((size, 2))
        for x in range(size):
            inputs[x, 0] = x
            inputs[x, 1] = mx[i]
        nn = pickle.load(open('data/pickle/fixed_%0.0f.pkl' %mx[i], 'rb'))
        outputs = nn.predict(inputs)
        plt.plot(inputs[:,0], outputs,
                    '-', 
                    linewidth=2,
                    color=colors[i], 
                    label='$\mu_f=$%0.0f' %mx[i], 
                    markevery=20)
    for i in range(len(mx)):
        inputs = np.zeros((size, 2))
        for x in range(size):
            inputs[x, 0] = x
            inputs[x, 1] = mx[i]
        nn = pickle.load(open('data/pickle/param_%0.0f.pkl' %mx[i], 'rb'))
        outputs = nn.predict(inputs)
        plt.plot(inputs[:,0], outputs,
                    'o', 
                    color=colors[i], 
                    alpha=0.5,
                    label='$\mu_p=$%0.0f' %mx[i], 
                    markevery=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.xlim([0,size])
    plt.ylim([0,1])
    plt.xlabel('$m_{WWbb}$')
    plt.ylabel('NN output')
    plt.savefig('plots/parameterized_vs_fixed_output_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_vs_fixed_output_plot.png')
    plt.clf()


def parameterized_vs_fixed_ROC_plot():
    '''
    parameterized_vs_fixed_ROC_plot pulls the ROC/AUC data for both fixed
    and parameterized training to plot both on the same canvas.
    '''

    print 'Entering parameterized_vs_fixed_ROC_plot'
    param_files = [np.loadtxt('data/plot_data/ROC/param_ROC_500.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_750.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_1000.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_1250.dat'),
                np.loadtxt('data/plot_data/ROC/param_ROC_1500.dat')
                ]
    fixed_files = [np.loadtxt('data/plot_data/ROC/fixed_ROC_500.dat'),
                    np.loadtxt('data/plot_data/ROC/fixed_ROC_750.dat'),
                    np.loadtxt('data/plot_data/ROC/fixed_ROC_1000.dat'),
                    np.loadtxt('data/plot_data/ROC/fixed_ROC_1250.dat'),
                    np.loadtxt('data/plot_data/ROC/fixed_ROC_1500.dat')
                    ]
    AUC_param = [np.loadtxt('data/plot_data/AUC/param_ROC_AUC_500.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_750.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_1000.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_1250.dat'),
                    np.loadtxt('data/plot_data/AUC/param_ROC_AUC_1500.dat')
                    ]

    AUC_fixed = [np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_500.dat'),
                    np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_750.dat'),
                    np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_1000.dat'),
                    np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_1250.dat'),
                    np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_1500.dat')
                    ]
    fixed_markers = ['b-', 'g-', 'r-', 'c-', 'm-']
    mx = [500, 750, 1000, 1250,1500]

    for i in range(len(param_files)):
        plt.plot(param_files[i][:,0], param_files[i][:,1], 
                    'o', 
                    markerfacecolor=colors[i],
                    alpha=0.5, 
                    markevery=5000, 
                    label='$\mu_p=$%s (AUC=%0.2f)' %(mx[i], AUC_param[i]),  
                    rasterized=True)
    for i in range(len(fixed_files)):
        plt.plot(fixed_files[i][:,0], fixed_files[i][:,1], 
                    '-', 
                    color=colors[i], 
                    alpha=1, 
                    markevery=100, 
                    linewidth=2, 
                    label='$\mu_f=$%s (AUC=%0.2f)' %(mx[i], AUC_fixed[i]),  
                    rasterized=True)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('1/Background efficiency')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig('plots/parameterized_vs_fixed_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_vs_fixed_ROC_plot.png')  
    plt.clf()


'''
Histograms 
'''


def plot_histogram():
    '''
    plot_histogram plots a histogram of the signal and backgrounds pulled from 
    the root files in the root_export directory
    '''

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
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim([250, 3000])
    #plt.ylim([0, 35000])
    plt.savefig('plots/signal_background_histogram.pdf', dpi=400)
    plt.savefig('plots/images/signal_background_histogram.png')
    plt.clf()


def parameterized_vs_fixed_output_histogram():
    '''
    parameterized_vs_fixed_output_histogram plots the outputs of the fixed and 
    parameterized training outputs to see the distribution of signal/background
    after trianing.
    '''

    print 'Entering output_histogram'

    mx = [500, 750, 1000, 1250, 1500]
    fixed_files = ['data/plot_data/fixed_500.dat',
                    'data/plot_data/fixed_750.dat',
                    'data/plot_data/fixed_1000.dat',
                    'data/plot_data/fixed_1250.dat',
                    'data/plot_data/fixed_1500.dat']

    param_files = ['data/plot_data/param_500.dat',
                    'data/plot_data/param_750.dat',
                    'data/plot_data/param_1000.dat',
                    'data/plot_data/fixed_1250.dat',
                    'data/plot_data/fixed_1500.dat']
    for idx, file in enumerate(fixed_files):
        data = np.loadtxt(file)
        n, bins, patches = plt.hist([data[:,1]],
                    bins=50, 
                    normed=True,
                    histtype='step', 
                    color=colors[idx],
                    label='$\mu_f=$%s' %mx[idx],
                    alpha=0.5, 
                    rasterized=True)

    for idx, file in enumerate(param_files):
        data = np.loadtxt(file)
        n, bins, patches = plt.hist([data[:,1]],
                    bins=50, 
                    normed=True,
                    histtype='stepfilled', 
                    color=colors[idx],
                    label='$\mu_p=$%s' %mx[idx],
                    alpha=0.3, 
                    rasterized=True)
    plt.setp(patches)
    plt.xlim([0,1])
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 1), fontsize=10)
    plt.savefig('plots/parameterized_vs_fixed_output_histogram.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_vs_fixed_output_histogram.png')
    plt.clf()

if __name__ == '__main__':
    '''
    File Runners
    '''
    #file_runner()
    #flat_bkg(10000,0,5000)
    #file_concatenater()
    
    '''
    Fixed Training and Plots
    '''
    #fixed_training()
    #fixed_training_plot()
    #fixed_ROC_plot()

    '''
    Parameterized Training and Plots 
    '''
    #parameterized_training()
    #parameterized_function_runner()
    #parameterized_training_plot()    
    #parameterized_ROC_plot()

    '''
    Comparison Training and Plots
    '''
    parameterized_vs_fixed_output_plot()
    parameterized_vs_fixed_ROC_plot()
    
    '''
    Output Histograms
    '''
    #plot_histogram()
    #parameterized_vs_fixed_output_histogram()
