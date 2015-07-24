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
from sklearn.externals import joblib
from sknn.mlp import Regressor, Classifier, Layer
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt


# Global items

# Plot marks (color circles - i.e. bo = blue circle, go = green circle)
plt_marker=['bo', 'go', 'ro', 'co', 'mo', 'yo', 'bo', 'wo']

def data_backg_merge():
    numTrain = 1500
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
    mwwbb_file = [400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500.]

    num_files = len(mwwbb_list)

    # Initialize ROOTs RooWorkspace
    w = ROOT.RooWorkspace('w')
    # Generate a flat background signal
    #print "Generating a flat background PDF"
    w.factory('Uniform::e(x[0,7000])')

    # Define variables
    x      = w.var('x')
    bkgpdf = w.pdf('e')

    # create training, testing data
    # np.zeros((rows, columns))

    # Fill traindata, testdata and testdata1
    print 'Generating background data and concatenting with signal data'
    bkg_values = bkgpdf.generate(ROOT.RooArgSet(x), numTrain*12)
    for j in range(len(mwwbb_list)):
        bkgdata = np.zeros((numTrain, 3))
        for i in range(numTrain):
            bkgdata[i, 0] = bkg_values.get(i).getRealValue('x')
            bkgdata[i, 1] = mwwbb_file[j]
            bkgdata[i, 2] = 0
        conc = np.concatenate((mwwbb_list[j],bkgdata))
        np.savetxt('data/dat_merge/conc_data_%s.dat' %j, conc, fmt='%f')


def data_backg_merge_param():
    numTrain = 20000
    mwwbb_complete = np.loadtxt('data/mwwbb/mwwbb_complete.dat')

    # Initialize ROOTs RooWorkspace
    w = ROOT.RooWorkspace('w')
    # Generate a flat background signal
    #print "Generating a flat background PDF"
    w.factory('Uniform::e(x[0,7000])')

    # Define variables
    x      = w.var('x')
    bkgpdf = w.pdf('e')

    print 'Generating background data and concatenting with signal data'
    bkg_values = bkgpdf.generate(ROOT.RooArgSet(x), numTrain)
    bkgdata = np.zeros((numTrain, 3))
    for i in range(numTrain):
        bkgdata[i, 0] = bkg_values.get(i).getRealValue('x')
        bkgdata[i, 1] = 0
        bkgdata[i, 2] = 0
    conc = np.concatenate((mwwbb_complete,bkgdata))
    np.savetxt('data/dat_merge/conc_data_complete.dat', conc, fmt='%f')



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
    mwwbb_400 = np.loadtxt('data/dat_merge/conc_data_0.dat')
    mwwbb_500 = np.loadtxt('data/dat_merge/conc_data_1.dat')
    mwwbb_600 = np.loadtxt('data/dat_merge/conc_data_2.dat')
    mwwbb_700 = np.loadtxt('data/dat_merge/conc_data_3.dat')
    mwwbb_800 = np.loadtxt('data/dat_merge/conc_data_4.dat')
    mwwbb_900 = np.loadtxt('data/dat_merge/conc_data_5.dat')
    mwwbb_1000 = np.loadtxt('data/dat_merge/conc_data_6.dat')
    mwwbb_1100 = np.loadtxt('data/dat_merge/conc_data_7.dat')
    mwwbb_1200 = np.loadtxt('data/dat_merge/conc_data_8.dat')
    mwwbb_1300 = np.loadtxt('data/dat_merge/conc_data_9.dat')
    mwwbb_1400 = np.loadtxt('data/dat_merge/conc_data_10.dat')
    mwwbb_1500 = np.loadtxt('data/dat_merge/conc_data_11.dat')
    mwwbb_train = [mwwbb_400[:,0:2], mwwbb_500[:,0:2], mwwbb_600[:,0:2], mwwbb_700[:,0:2],
        mwwbb_800[:,0:2], mwwbb_900[:,0:2], mwwbb_1000[:,0:2], mwwbb_1100[:,0:2],
        mwwbb_1200[:,0:2], mwwbb_1300[:,0:2], mwwbb_1400[:,0:2], mwwbb_1500[:,0:2]]
    mwwbb_target = [mwwbb_400[:,2], mwwbb_500[:,2], mwwbb_600[:,2], mwwbb_700[:,2],
        mwwbb_800[:,2], mwwbb_900[:,2], mwwbb_1000[:,2], mwwbb_1100[:,2],
        mwwbb_1200[:,2], mwwbb_1300[:,2], mwwbb_1400[:,2], mwwbb_1500[:,2]]
    mwwbb_file = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    mwwbb_text = ['400', '500', '600', '700', '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']


    for i in range(1):
        print 'Working on mu=%s' %mwwbb_text[i]
        traindata = mwwbb_train[i]
        targetdata = mwwbb_target[i]
        nn = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', 
                Regressor(
                    layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                    learning_rate=0.01,
                    n_iter=250, 
                    #learning_momentum=0.1,
                    #batch_size=5,
                    learning_rule="nesterov",  
                    #valid_size=0.05,
                    verbose=True,
                    #debug=True
                    ))])



        nn.fit(traindata, targetdata)
        
        fit_score = nn.score(traindata, targetdata)
        print 'score = %s' %fit_score
        # Training outputs
        outputs = nn.predict(traindata)

        plt.plot(traindata[:, 0], outputs, 'o', alpha=0.5, label='$\mu=$%s' %mwwbb_text[i])
    plt.xlim([0, 3000])
    plt.ylim([-0.2, 1.2])    
    plt.legend(bbox_to_anchor=(0.85, 1.05), loc=2, borderaxespad=0)    
    #plt.show()
    plt.savefig('plots/mwwbb_fixed.pdf')
    plt.savefig('plots/images/mwwbb_fixed.png')

def mwwbb_parameterized():
    mwwbb_complete = np.loadtxt('data/mwwbb/mwwbb_complete_3val.dat')

    traindata      = mwwbb_complete[:,0:2]
    targetdata      = mwwbb_complete[:,2]
    print traindata
    print targetdata
    # Initialize ML method (SVM or NN)
    print "Machine Learning method initialized"

    #nn = svm.NuSVR(nu=1)
    nn = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', 
            Regressor(
                layers =[Layer("Sigmoid", units=3),Layer("Sigmoid")],
                learning_rate=0.01,
                n_iter=250, 
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
    plt.ylabel('NN_output ( training_input )')
    plt.xlabel('training_input')
    plt.xlim([0, 4000])
    plt.ylim([-0.2, 1.2])
    #plt.axhline(y=0, color = 'black', linewidth = 2, alpha=0.75)
    #plt.axhline(y=1, color = 'black', linewidth = 2, alpha=0.75)
    plt.grid(True)
    plt.suptitle('Theano NN fixed training for mwwbb input',
               fontsize=14, fontweight='bold')
    plt.savefig('plots/mwwbbParamTraining.pdf')
    plt.savefig('plots/images/mwwbbParamTraining.png')
    #plt.show()
    plt.clf()

    pickle.dump(nn, open('data/mwwbb_param.pkl', 'wb'))

def scikitlearnFunc(x, alpha):
    nn = pickle.load(open('data/mwwbb_param.pkl','rb'))
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

    plt.savefig('plots/mwwbbParamTraining_complete.pdf')
    plt.savefig('plots/images/mwwbbParamTraining_complete.png')
    #plt.show()

if __name__ == '__main__':   
    #data_backg_merge()
    #mwwbb_importer()
    #mwwbb_histogram()
    #mwwbb_fixed()
    #parameterizedRunner()
    #data_backg_merge_param()
    mwwbb_parameterized()
    mwwbbParameterizedRunner()
