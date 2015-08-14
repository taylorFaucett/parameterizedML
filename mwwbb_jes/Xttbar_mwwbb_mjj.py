'''
author Taylor Faucett <tfaucett@uci.edu>
'''

import ROOT
import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer

''' 
The standard set of matplotlib colors plus some extras from the HTML/CSS standard
are used in multiple functions so they are defined globally here.
'''
colors  = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'orange']


def file_runner(directory):
    print 'Entering file_runner'
    files = glob.iglob(directory+'*.root')
    for data in files:
        file_generate(data)


def file_generate(root_file):
    print 'Entering file_generate'
    print 'Generating data using values from: %s' %root_file
    mwwbb = root_export(root_file,'xtt','mwwbb')
    mx = root_export(root_file,'xtt','mx')
    mjj = root_export(root_file, 'xtt', 'mjj')
    target = root_export(root_file,'xtt','target')
    jes = root_export(root_file, 'xtt', 'jes')
    size = len(mwwbb)
    data = np.zeros((size, 5))
    if target[0] == 0.000000:
        label = 'bkg'
    elif target[0] == 1.000000:
        label = 'sig'
    for i in range(size):
        data[i, 0] = mwwbb[i]
        data[i, 1] = mjj[i]
        data[i, 2] = jes[i]
        data[i, 3] = target[i]
        data[i, 4] = mx[i]
    np.savetxt('data/root_export/%s_mx_%0.0f_jes_%0.3f.dat' %(label, mx[0], jes[0]), data, fmt='%f')

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
        np.savetxt('data/concatenated/ttbar_mx_%0.3f.dat' %sig[0,2], data_complete, fmt='%f')


'''
Fixed Training and Plots 
'''

def fixed_training():
    '''
    fixed_training takes concatenated data and splits the data set into training_data 
    (i.e. [mwwbb_value, mjj_value, jes], e.g. [[532.1, 84.3, 0.750], [728.4, 121.3, 0.750]]) and target_data (i.e. [1, 1, ... 0, 0]).
    The input undergoes pre-processing via SKLearns pipeline and then a NN is trained using the 
    SKLearnNN wrapper which processes the data using Theano/PyLearn2. NN learning parameters 
    (e.g. learning_rate, n_iter, etc) are selected before hand. target_data from the inputs
    are then used along with predictions to calculate the Receiver Operating Characteristic (ROC) curve
    and Area Under the Curve (AUC). 
    '''

    print 'Entering fixed_training'

    # Training input files
    file_list = ['data/concatenated/ttbar_mx_0.750.dat',
                    'data/concatenated/ttbar_mx_0.900.dat',
                    'data/concatenated/ttbar_mx_0.950.dat',
                    'data/concatenated/ttbar_mx_0.975.dat',
                    'data/concatenated/ttbar_mx_1.000.dat',
                    'data/concatenated/ttbar_mx_1.025.dat',
                    'data/concatenated/ttbar_mx_1.050.dat',
                    'data/concatenated/ttbar_mx_1.100.dat',
                    'data/concatenated/ttbar_mx_1.250.dat']

    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network',
            Regressor(
                layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                learning_rate=0.01,
                #n_stable=1,
                #f_stable=100,
                n_iter=50,
                #learning_momentum=0.1,
                batch_size=10,
                learning_rule="nesterov",
                #valid_size=0.05,
                #verbose=True,
                #debug=True
                ))])

    for file in file_list:
        input_data = np.loadtxt(file)

        training_data = input_data[:,0:3]
        target_data = input_data[:,3]
        
        jes = training_data[0,2]
        
        print 'Processing fixed training on mu=%0.3f' %jes

    	nn.fit(training_data, target_data)

    	fit_score = nn.score(training_data, target_data)
    	print 'score = %s' %fit_score

    	outputs = nn.predict(training_data)
    	outputs = outputs.reshape((1,len(outputs)))

    	output_data = np.vstack((training_data[:,0], 
                                    training_data[:,1], 
                                    training_data[:,2],
                                    target_data, 
                                    outputs)).T
    	np.savetxt('data/plot_data/fixed_%0.3f.dat' %jes, output_data, fmt='%f')

    	actual = target_data
    	predictions = outputs[0]
    	fpr, tpr, thresholds = roc_curve(actual, predictions)
    	ROC_plot = np.vstack((fpr, tpr)).T
    	ROC_AUC = [auc(fpr, tpr)]
    	np.savetxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %jes, ROC_plot, fmt='%f')
    	np.savetxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %jes, ROC_AUC)

    	pickle.dump(nn, open('data/pickle/fixed_%0.3f.pkl' %jes, 'wb'))

def fixed_training_plot(): 
    '''
    fixed_training_plot is no longer necessary and has been replaced by
    parameterized_vs_fixed_output_plot. Initially this took prediction data 
    from fixed_training and plotted the values. However, it is more convenient to pickle
    the NN training from fixed training and then run predictions separately rather than
    generate prediction outputs during training. This is left here because I'm a hoarder.
    '''

    print 'Entering fixed_training_plot'

    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]
    for idx, jes in enumerate(jes_list):
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %jes)
        data.sort(axis=0)
        plt.plot(data[:,0], data[:,4], 
                    color=colors[idx], 
                    linewidth = 1,
                    #alpha=1, 
                    markevery = 100, 
                    #markersize = 1,
                    label='jes$_f$=%0.3f' %jes, 
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('$m_{WWbb}$')
    plt.xlim([0,3000])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixed_training_mwwbb_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_training_mwwbb_plot.png')
    plt.clf()

    for idx, jes in enumerate(jes_list):
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %jes)
        data.sort(axis=0)
        plt.plot(data[:,1], data[:,4], 
                    color=colors[idx], 
                    linewidth = 1,
                    #alpha=1, 
                    markevery = 100, 
                    #markersize = 1,
                    label='jes$_f$=%0.3f' %jes, 
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('$m_{jj}$')
    plt.xlim([0,250])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/fixed_training_mjj_plot.pdf', dpi=400)
    plt.savefig('plots/images/fixed_training_mjj_plot.png')
    plt.clf()

def fixed_ROC_plot():
    '''
    fixed_ROC_plot takes in ROC and AUC values processed during training in fixed_training
    and plots the ROC curve. This will be deprecated in the future, for the same reason as
    fixed_training_plot, so that AUC and ROC values will be calculated and plotted separately
    from the training time.
    '''

    print "Entering fixed_ROC_plot"
    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]
    for idx, jes in enumerate(jes_list):
        data = np.loadtxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %jes)
        AUC  = np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %jes)
        plt.plot(data[:,0], data[:,1],
                    '-', 
                    color=colors[idx], 
                    label='jes$_f$=%0.3f (AUC=%0.3f)' %(jes, AUC), 
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
    excluded (e.g. mwwbb_complete750 trains for all signals excluding the one at jes=0.750). 
    A seperate NN is trained for each of these scenarios and then pickled in an appropriately 
    labeled file (e.g. Training excluding jes=0.750 is pickled as param_750.pkl)
    '''

    print 'Entering parameterized_training'

    mx_750 = np.loadtxt('data/concatenated/ttbar_mx_0.750.dat')
    mx_900 = np.loadtxt('data/concatenated/ttbar_mx_0.900.dat')
    mx_950 = np.loadtxt('data/concatenated/ttbar_mx_0.950.dat')
    mx_975 = np.loadtxt('data/concatenated/ttbar_mx_0.975.dat')
    mx_1000 = np.loadtxt('data/concatenated/ttbar_mx_1.000.dat')
    mx_1025 = np.loadtxt('data/concatenated/ttbar_mx_1.025.dat')
    mx_1050 = np.loadtxt('data/concatenated/ttbar_mx_1.050.dat')
    mx_1100 = np.loadtxt('data/concatenated/ttbar_mx_1.100.dat')
    mx_1250 = np.loadtxt('data/concatenated/ttbar_mx_1.250.dat')
    print 'Files loaded'


    mwwbb_complete750 = np.concatenate((
                        mx_900,
                        mx_950,
                        mx_975,
                        mx_1000,
                        mx_1025,
                        mx_1050,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete900 = np.concatenate((
                        mx_750,
                        mx_950,
                        mx_975,
                        mx_1000,
                        mx_1025,
                        mx_1050,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete950 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_975,
                        mx_1000,
                        mx_1025,
                        mx_1050,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete975 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_950,
                        mx_1000,
                        mx_1025,
                        mx_1050,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete1000 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_950,
                        mx_975,
                        mx_1025,
                        mx_1050,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete1025 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_950,
                        mx_975,
                        mx_1000,
                        mx_1050,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete1050 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_950,
                        mx_975,
                        mx_1000,
                        mx_1025,
                        mx_1100,
                        mx_1250),
                        axis=0)
    mwwbb_complete1100 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_950,
                        mx_975,
                        mx_1000,
                        mx_1025,
                        mx_1050,
                        mx_1250),
                        axis=0)    
    mwwbb_complete1250 = np.concatenate((
                        mx_750,
                        mx_900,
                        mx_950,
                        mx_975,
                        mx_1000,
                        mx_1025,
                        mx_1050,
                        mx_1100),
                        axis=0)  

    training_list = [mwwbb_complete750[:,0:3],
                        mwwbb_complete900[:,0:3],
                        mwwbb_complete950[:,0:3],
                        mwwbb_complete975[:,0:3],
                        mwwbb_complete1000[:,0:3],
                        mwwbb_complete1250[:,0:3],
                        mwwbb_complete1050[:,0:3],
                        mwwbb_complete1100[:,0:3],
                        mwwbb_complete1250[:,0:3]
                    ]

    target_list = [mwwbb_complete750[:,3],
                        mwwbb_complete900[:,3],
                        mwwbb_complete950[:,3],
                        mwwbb_complete975[:,3],
                        mwwbb_complete1000[:,3],
                        mwwbb_complete1250[:,3],
                        mwwbb_complete1050[:,3],
                        mwwbb_complete1100[:,3],
                        mwwbb_complete1250[:,3]
                    ]

    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]

    for idx, (training_data, target_data, jes) in enumerate(zip(training_list, target_list, jes_list)):
    	print 'Parameterized training on all signals except for jes=%0.3f' %jes
        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network',
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    n_iter=50,
                    #n_stable=1,
                    #f_stable=0.001,
                    #learning_momentum=0.1,
                    batch_size=100,
                    learning_rule="nesterov",
                    #valid_size=0.05,
                    #verbose=True,
                    #debug=True
                    ))])

        nn.fit(training_data, target_data)

        fit_score = nn.score(training_data, target_data)
        print 'score = %s' %fit_score
        
        #outputs = nn.predict(training_data)
        #outputs = outputs.reshape((1, len(outputs)))
        #param_plot = np.vstack((training_data[:,0], outputs)).T
        #np.savetxt('data/plot_data/param_%s.dat' %mx[idx], param_plot, fmt='%f')
        pickle.dump(nn, open('data/pickle/param_%0.3f.pkl' %jes, 'wb'))

def parameterized_function(mwwbb, mjj, alpha, nn):
    '''
    parameterized_function acts as the function with an additional parameter alpha. 
    This is used in parameterized_function_runner as the function which interpolates 
    signals at alpha values. For example, a NN trained at mu=500, 750, 1250 and 1500
    can use the parameterized_function with a specificied value of alpha=1000 to 
    interpolate the curve, ROC and AUC values at mu=1000 despite not having been trained
    for that location. 
    '''

    training_data = np.array((mwwbb, mjj, alpha), ndmin=2)
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
    alpha_list = [0.750, 
                    0.900, 
                    0.950, 
                    0.975, 
                    1.000,
                    1.025,
                    1.050,
                    1.100,
                    1.250]

    for idx, alpha in enumerate(alpha_list):
        data = np.loadtxt('data/concatenated/ttbar_mx_%0.3f.dat' %alpha)
        size = len(data[:,0])
        print 'processing using: data/pickle/param_%0.3f.pkl' %alpha
        nn = pickle.load(open('data/pickle/param_%0.3f.pkl' %alpha, 'rb'))
        inputs = data[:,0:2]
        actuals = data[:,3]
        input1 = inputs[:,0]
        input2 = inputs[:,1]
        predictions = []
        for x in range(0,size):
            outputs = parameterized_function(input1[x]/1., input2[x]/1., alpha, nn)
            predictions.append(outputs[0][0])
            #print '(%s, %s, %s)' %(input1[x], input2[x], predictions[x])
        data = np.vstack((input1, input2, predictions, actuals)).T
        np.savetxt('data/plot_data/param_%0.3f.dat' %alpha, data, fmt='%f')
        fpr, tpr, thresholds = roc_curve(actuals, predictions)
        roc_auc = [auc(fpr, tpr)]

        roc_data = np.vstack((fpr, tpr)).T
        np.savetxt('data/plot_data/ROC/param_ROC_%0.3f.dat' %alpha, roc_data, fmt='%f')
        np.savetxt('data/plot_data/AUC/param_ROC_AUC_%0.3f.dat' %alpha, roc_auc)

def parameterized_training_plot(): 
    '''
    parameterized_training_plot plots the output values generated during 
    parameterized_function_runner.
    '''

    print 'Entering parameterized_training_plot'

    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]

    for idx, jes in enumerate(jes_list):
        data = np.loadtxt('data/plot_data/param_%0.3f.dat' %jes)
        data.sort(axis=0)
        plt.plot(data[:,0], data[:,2],
                    'o', 
                    color=colors[idx], 
                    linewidth = 1,
                    alpha=0.3, 
                    markevery = 10000, 
                    #markersize = 1,
                    label='jes$_p$=%0.3f' %jes, 
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('$m_{WWbb}$')
    plt.xlim([0,3000])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/parameterized_training_mwwbb_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_training_mwwbb_plot.png')
    plt.clf()

    for idx, jes in enumerate(jes_list):
        data = np.loadtxt('data/plot_data/param_%0.3f.dat' %jes)
        data.sort(axis=0)
        plt.plot(data[:,1], data[:,2], 
                    'o',
                    color=colors[idx], 
                    linewidth = 1,
                    alpha=0.3, 
                    markevery = 10000, 
                    #markersize = 1,
                    label='jes$_p$=%0.3f' %jes, 
                    rasterized=True)
    plt.ylabel('NN output')
    plt.xlabel('$m_{jj}$')
    plt.xlim([0,250])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/parameterized_training_mjj_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_training_mjj_plot.png')
    plt.clf()


def parameterized_ROC_plot():
    '''
    parameterized_ROC_plot plots the ROC and AUC values generated during
    parameterized_function_runner
    '''

    print 'Entering parameterized_ROC_plot'

    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]

    for idx, jes in enumerate(jes_list):
        ROC = np.loadtxt('data/plot_data/ROC/param_ROC_%0.3f.dat' %jes)
        AUC = np.loadtxt('data/plot_data/AUC/param_ROC_AUC_%0.3f.dat' %jes)
        plt.plot(ROC[:,0], ROC[:,1], 
                    'o', 
                    markerfacecolor=colors[idx],
                    alpha=0.5, 
                    markevery=2000, 
                    label='jes$_p$=%0.3f (AUC=%0.3f)' %(jes, AUC),  
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
    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]
    #mwwbb plot
    for idx, jes in enumerate(jes_list):
        print 'Plotting mass mwwbb, jes=%0.3f' %jes
        fixed = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %jes)
        fixed.sort(axis=0)
        param = np.loadtxt('data/plot_data/param_%0.3f.dat' %jes)
        param.sort(axis=0)
        plt.plot(fixed[:,0], fixed[:,4],
                    '-',
                    color=colors[idx],
                    markevery=1000,
                    label='jes$_f$=%0.3f' %jes,
                    rasterized=True
                    )
        plt.plot(param[:,0], param[:,2],
                    'o',
                    color=colors[idx],
                    markevery=10000,
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
        fixed.sort(axis=0)
        param = np.loadtxt('data/plot_data/param_%0.3f.dat' %jes)
        param.sort(axis=0)
        plt.plot(fixed[:,1], fixed[:,4],
                    '-',
                    color=colors[idx],
                    markevery=1000,
                    label='jes$_f$=%0.3f' %jes,
                    rasterized=True
                    )
        plt.plot(param[:,1], param[:,2],
                    'o',
                    color=colors[idx],
                    markevery=10000,
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

def parameterized_vs_fixed_ROC_plot():
    '''
    parameterized_vs_fixed_ROC_plot pulls the ROC/AUC data for both fixed
    and parameterized training to plot both on the same canvas.
    '''
    jes_list = [0.750, 0.900, 0.950, 0.975, 1.000, 1.025, 1.050, 1.100, 1.250]
    #mwwbb plot
    for idx, jes in enumerate(jes_list):
        print 'Plotting ROC for jes=%0.3f' %jes
        fixed_ROC = np.loadtxt('data/plot_data/ROC/fixed_ROC_%0.3f.dat' %jes)
        fixed_AUC = np.loadtxt('data/plot_data/AUC/fixed_ROC_AUC_%0.3f.dat' %jes)
        fixed_ROC.sort(axis=0)
        param_ROC = np.loadtxt('data/plot_data/ROC/param_ROC_%0.3f.dat' %jes)
        param_AUC = np.loadtxt('data/plot_data/AUC/param_ROC_AUC_%0.3f.dat' %jes)
        param_ROC.sort(axis=0)
        plt.plot(fixed_ROC[:,0], fixed_ROC[:,1],
                    '-',
                    color=colors[idx],
                    markevery=1000,
                    label='jes$_f$=%0.3f (AUC=%0.2f)' %(jes, fixed_AUC),
                    rasterized=True
                    )
        plt.plot(param_ROC[:,0], param_ROC[:,1],
                    'o',
                    color=colors[idx],
                    markevery=10000,
                    alpha=0.3,
                    label='jes$_p$=%0.3f (AUC=%0.2f)' %(jes, param_AUC),
                    rasterized=True
                    )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.plot([0,1],[0,1], 'r--')
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('1/Background efficiency')
    plt.xlabel('Signal efficiency')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.savefig('plots/parameterized_vs_fixed_ROC_plot.pdf', dpi=400)
    plt.savefig('plots/images/parameterized_vs_fixed_ROC_plot.png')
    plt.clf()


def fixed_output_plot_heat_map():
    print 'Entering fixed_output_plot_heat_map'
    jes_list = [0.750, 
                0.900, 
                0.950, 
                0.975, 
                1.000, 
                1.025, 
                1.050, 
                1.100, 
                1.250
                ]
    
    for idx, jes in enumerate(jes_list):
        print 'Plotting jes=%0.3f' %jes
        data = np.loadtxt('data/plot_data/fixed_%0.3f.dat' %jes)
        size = 5000
        x = data[:,0][:size]
        y = data[:,1][:size]
        z = data[:,4][:size]
        xmin = 0
        xmax = 3000
        ymin = 0
        ymax = 500
        zmin = 0
        zmax = 1
        color_map = 'CMRmap'
        #xmin = x.min()
        #xmax = x.max()
        #ymin = y.min()
        #ymax = y.max()
        #zmin = z.min()
        #zmax = z.max()

        # Set up a regular grid of interpolation points
        xi, yi = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
        zi = rbf(xi, yi)
        plt.imshow(zi, 
                    vmin=zmin, 
                    vmax=zmax, 
                    origin='lower',
                    extent=[xmin, xmax, ymin, ymax], 
                    aspect='auto',
                    #cmap=color_map
                    )
        plt.scatter(x, y, c=z, 
                    #cmap=color_map
                    )
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.title('jes=%0.3f' %jes)
        plt.clim(0,1)
        plt.colorbar(label='NN output')
        plt.xlabel('$m_{WWbb}$')
        plt.ylabel('$m_{jj}$')
        plt.savefig('plots/output_heat_map/fixed_output_plot_heat_map_%0.3f.pdf' %jes, dpi=400)
        plt.savefig('plots/output_heat_map/images/fixed/fixed_output_plot_heat_map_%0.3f.png' %jes)
        plt.clf()

def parameterized_output_plot_heat_map():
    print 'Entering parameterized_output_plot_heat_map'
    jes_list = [0.750, 
                0.900, 
                0.950, 
                0.975, 
                1.000, 
                1.025, 
                1.050, 
                1.100, 
                1.250
                ]
    
    for idx, jes in enumerate(jes_list):
        print 'Plotting jes=%0.3f' %jes
        data = np.loadtxt('data/plot_data/param_%0.3f.dat' %jes)
        size = 5000
        x = data[:,0][:size]
        y = data[:,1][:size]
        z = data[:,2][:size]
        xmin = 0
        xmax = 3000
        ymin = 0
        ymax = 500
        zmin = 0
        zmax = 1
        color_map = 'CMRmap'
        #xmin = x.min()
        #xmax = x.max()
        #ymin = y.min()
        #ymax = y.max()
        #zmin = z.min()
        #zmax = z.max()

        # Set up a regular grid of interpolation points
        xi, yi = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
        zi = rbf(xi, yi)
        plt.imshow(zi, 
                    vmin=zmin, 
                    vmax=zmax, 
                    origin='lower',
                    extent=[xmin, xmax, ymin, ymax], 
                    aspect='auto',
                    #cmap=color_map
                    )
        plt.scatter(x, y, c=z, 
                    #cmap=color_map
                    )
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.title('jes=%0.3f' %jes)
        plt.clim(0,1)
        plt.colorbar(label='NN output')
        plt.xlabel('$m_{WWbb}$')
        plt.ylabel('$m_{jj}$')
        plt.savefig('plots/output_heat_map/parameterized_output_plot_heat_map_%0.3f.pdf' %jes, dpi=400)
        plt.savefig('plots/output_heat_map/images/parameterized/parameterized_output_plot_heat_map_%0.3f.png' %jes)
        plt.clf()
'''
Histograms 
'''


def plot_histogram():
    '''
    plot_histogram plots a histogram of the signal and backgrounds pulled from 
    the root files in the root_export directory
    '''

    print 'Entering plot_histogram'
    bin_size   = 50
    sig_dat = glob.iglob('data/root_export/sig_*.dat')
    bkg_dat = glob.iglob('data/root_export/bkg_*.dat')

    for idx, (signal, background) in enumerate(zip(sig_dat, bkg_dat)):
        sig = np.loadtxt(signal)
        bkg = np.loadtxt(background)
        print 'Plotting mWWbb at jes=%0.3f' %sig[0,2]
        n, bins, patches = plt.hist([sig[:,0], bkg[:,0]],
                            bins=range(0,3000, bin_size), normed=True,
                            histtype='step', alpha=0.75, linewidth=2, 
                            label=['Signal', 'Background'],
                            rasterized=True)
        plt.setp(patches)
        plt.ylabel('Fraction of events$/%0.0f$ GeV' %bin_size)
        plt.xlabel('m$_{WWbb}$ [GeV]')
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=10)
        plt.xlim([0, 3000])
        #plt.ylim([0, 35000])
        plt.title('jes=%0.3f' %sig[0,2])
        plt.savefig('plots/histograms/mWWbb_histogram_%0.3f.pdf' %sig[0,2], dpi=400)
        plt.savefig('plots/histograms/images/mWWbb_histogram_%0.3f.png' %sig[0,2])
        plt.clf()
    sig_dat = glob.iglob('data/root_export/sig_*.dat')
    bkg_dat = glob.iglob('data/root_export/bkg_*.dat')
    bin_size   = 15
    for idx, (signal, background) in enumerate(zip(sig_dat, bkg_dat)):
        sig = np.loadtxt(signal)
        bkg = np.loadtxt(background)
        print 'Plotting mjj at jes=%0.3f' %sig[0,2]
        n, bins, patches = plt.hist([sig[:,1], bkg[:,1]],
                            bins=range(0,500, bin_size), normed=True,
                            histtype='step', alpha=0.75, linewidth=2, 
                            label=['Signal', 'Background'],
                            rasterized=True)
        plt.setp(patches)
        plt.ylabel('Fraction of events$/%0.0f$ GeV' %bin_size)
        plt.xlabel('m$_{jj}$ [GeV]')
        plt.grid(True)
        plt.title('jes=%0.3f' %sig[0,2])
        plt.legend(loc='upper right', fontsize=10)
        plt.xlim([0, 500])
        #plt.ylim([0, 35000])
        plt.savefig('plots/histograms/mjj_histogram_%0.3f.pdf' %sig[0,2], dpi=400)
        plt.savefig('plots/histograms/images/mjj_histogram_%0.3f.png' %sig[0,2])
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
    fixed_training_plot()
    fixed_ROC_plot()
    fixed_output_plot_heat_map()
    
    '''
    Parameterized Training and Plots 
    '''
    #parameterized_training()
    #parameterized_function_runner()
    parameterized_training_plot()    
    parameterized_ROC_plot()
    parameterized_output_plot_heat_map()
    
    '''
    Comparison Training and Plots
    '''
    parameterized_vs_fixed_output_plot()
    parameterized_vs_fixed_ROC_plot()
    
    '''
    Output Histograms
    '''
    #plot_histogram()
