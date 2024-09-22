import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import PIL
import time
from IPython import display
import sys
import pickle
import gzip
from IPython.display import clear_output
import scipy
import random
import bisect
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,recall_score, precision_score
from tensorflow.keras import callbacks
from scipy.signal import find_peaks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Conv1D,Dropout,MaxPooling1D,MaxPooling2D,Flatten,Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
cwd = os.getcwd() #

def fullscrn():
    from IPython.core.display import display,HTML
    display(HTML('<style>'
        '#notebook { padding-top:0px !important; } ' 
        '.container { width:100% !important; } '
        '.end_space { min-height:0px !important; } '
        'html, body, .container{ margin:0!important;padding:0!important;}'
    '</style>'))

def Aug_data(cwd,clss,mthd,nmbr):                                    # load augmented data from folder
    dt=np.int16(pd.read_csv('{}/Aug/aug{}{}.csv'.format(cwd,clss,mthd)))
    if nmbr<1 :
        nmbr=1
    if len(dt)<nmbr :
        nmbr=len(dt)
    btch=int(int(len(dt))/nmbr)
    turn=np.random.randint(btch)
    data=[]
    for i in range (nmbr):
        data.append(dt[i*btch+turn])
    return (np.array(data)[:,1:])
        
    
def MITBIH(clss):
    if clss==1:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/1 NSR')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg1=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg1=np.append(ecg1,data2)
        ecg1=np.reshape(ecg1,(len(alldata),np.int(len(ecg1)/len(alldata))))
        cls=np.ones((len(alldata),1))
        ecg1=np.int32(np.concatenate((ecg1,cls),axis=1))
        #ecg1=np.random.permutation(ecg1)
        return(ecg1)

    elif clss==2:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/2 APB')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg2=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg2=np.append(ecg2,data2)
        ecg2=np.reshape(ecg2,(len(alldata),np.int(len(ecg2)/len(alldata))))
        cls=np.ones((len(alldata),1))*2
        ecg2=np.int32(np.concatenate((ecg2,cls),axis=1))
        #ecg2=np.random.permutation(ecg2)
        return(ecg2)

    elif clss==3:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/3 AFL')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg3=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg3=np.append(ecg3,data2)
        ecg3=np.reshape(ecg3,(len(alldata),np.int(len(ecg3)/len(alldata))))
        cls=np.ones((len(alldata),1))*3
        ecg3=np.int32(np.concatenate((ecg3,cls),axis=1))
        #ecg3=np.random.permutation(ecg3)
        return(ecg3)

    elif clss==4:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/4 AFIB')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg4=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg4=np.append(ecg4,data2)
        ecg4=np.reshape(ecg4,(len(alldata),np.int(len(ecg4)/len(alldata))))
        cls=np.ones((len(alldata),1))*4
        ecg4=np.int32(np.concatenate((ecg4,cls),axis=1))
        #ecg4=np.random.permutation(ecg4)
        return(ecg4)

    elif clss==5:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/5 SVTA')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg5=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg5=np.append(ecg5,data2)
        ecg5=np.reshape(ecg5,(len(alldata),np.int(len(ecg5)/len(alldata))))
        cls=np.ones((len(alldata),1))*5
        ecg5=np.int32(np.concatenate((ecg5,cls),axis=1))
        #ecg5=np.random.permutation(ecg5)
        return(ecg5)

    elif clss==6:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/6 WPW')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg6=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg6=np.append(ecg6,data2)
        ecg6=np.reshape(ecg6,(len(alldata),np.int(len(ecg6)/len(alldata))))
        cls=np.ones((len(alldata),1))*6
        ecg6=np.int32(np.concatenate((ecg6,cls),axis=1))
        #ecg6=np.random.permutation(ecg6)
        return(ecg6)

    elif clss==7:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/7 PVC')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg7=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg7=np.append(ecg7,data2)
        ecg7=np.reshape(ecg7,(len(alldata),np.int(len(ecg7)/len(alldata))))
        cls=np.ones((len(alldata),1))*7
        ecg7=np.int32(np.concatenate((ecg7,cls),axis=1))
        #ecg7=np.random.permutation(ecg7)
        return(ecg7)

    elif clss==8:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/8 Bigeminy')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg8=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg8=np.append(ecg8,data2)
        ecg8=np.reshape(ecg8,(len(alldata),np.int(len(ecg8)/len(alldata))))
        cls=np.ones((len(alldata),1))*8
        ecg8=np.int32(np.concatenate((ecg8,cls),axis=1))
        #ecg8=np.random.permutation(ecg8)
        return(ecg8)

    elif clss==9:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/9 Trigeminy')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg9=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg9=np.append(ecg9,data2)
        ecg9=np.reshape(ecg9,(len(alldata),np.int(len(ecg9)/len(alldata))))
        cls=np.ones((len(alldata),1))*9
        ecg9=np.int32(np.concatenate((ecg9,cls),axis=1))
        #ecg9=np.random.permutation(ecg9)
        return(ecg9)

    elif clss==10:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/10 VT')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg10=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg10=np.append(ecg10,data2)
        ecg10=np.reshape(ecg10,(len(alldata),np.int(len(ecg10)/len(alldata))))
        cls=np.ones((len(alldata),1))*10
        ecg10=np.int32(np.concatenate((ecg10,cls),axis=1))
        #ecg10=np.random.permutation(ecg10)
        return(ecg10)

    elif clss==11:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/11 IVR')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg11=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg11=np.append(ecg11,data2)
        ecg11=np.reshape(ecg11,(len(alldata),np.int(len(ecg11)/len(alldata))))
        cls=np.ones((len(alldata),1))*11
        ecg11=np.int32(np.concatenate((ecg11,cls),axis=1))
        #ecg11=np.random.permutation(ecg11)
        return(ecg11)

    elif clss==12:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/12 VFL')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg12=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg12=np.append(ecg12,data2)
        ecg12=np.reshape(ecg12,(len(alldata),np.int(len(ecg12)/len(alldata))))
        cls=np.ones((len(alldata),1))*12
        ecg12=np.int32(np.concatenate((ecg12,cls),axis=1))
        #ecg12=np.random.permutation(ecg12)
        return(ecg12)

    elif clss==13:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/13 fusion')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg13=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg13=np.append(ecg13,data2)
        ecg13=np.reshape(ecg13,(len(alldata),np.int(len(ecg13)/len(alldata))))
        cls=np.ones((len(alldata),1))*13
        ecg13=np.int32(np.concatenate((ecg13,cls),axis=1))
        #ecg13=np.random.permutation(ecg13)
        return(ecg13)

    elif clss==14:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/14 LBBBB')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg14=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg14=np.append(ecg14,data2)
        ecg14=np.reshape(ecg14,(len(alldata),np.int(len(ecg14)/len(alldata))))
        cls=np.ones((len(alldata),1))*14
        ecg14=np.int32(np.concatenate((ecg14,cls),axis=1))
        #ecg14=np.random.permutation(ecg14)
        return(ecg14)

    elif clss==15:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/15 RBBBB')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg15=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg15=np.append(ecg15,data2)
        ecg15=np.reshape(ecg15,(len(alldata),np.int(len(ecg15)/len(alldata))))
        cls=np.ones((len(alldata),1))*15
        ecg15=np.int32(np.concatenate((ecg15,cls),axis=1))
        #ecg15=np.random.permutation(ecg15)
        return(ecg15)

    elif clss==16:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/16 SDHB')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg16=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg16=np.append(ecg16,data2)
        ecg16=np.reshape(ecg16,(len(alldata),np.int(len(ecg16)/len(alldata))))
        cls=np.ones((len(alldata),1))*16
        ecg16=np.int32(np.concatenate((ecg16,cls),axis=1))
        #ecg16=np.random.permutation(ecg16)
        return(ecg16)

    elif clss==17:
        os.chdir('C:/Users/user/thesis/Data/ECG signals (1000 fragments)/MLII/17 PR')
        # Get a list for .mat files in current folder
        mat_files = glob.glob('*.mat')

        # List for stroring all the data
        alldata = []

        # Iterate mat files
        for fname in mat_files:
            # Load mat file data into data.
            data = scipy.io.loadmat(fname)
            #data = loadmat(fname)

            # Append data to the list
            alldata.append(data)

        ecg17=[]
        for i in range (len(alldata)):
            #data2=np.array(alldata[i]['val'][0])
            data2=np.ndarray.tolist(alldata[i]['val'][0])
            data2=np.array(data2).T
            #print(data2)
            #ecg.append(data2)
            ecg17=np.append(ecg17,data2)
        ecg17=np.reshape(ecg17,(len(alldata),np.int(len(ecg17)/len(alldata))))
        cls=np.ones((len(alldata),1))*17
        ecg17=np.int32(np.concatenate((ecg17,cls),axis=1))
        #ecg17=np.random.permutation(ecg17)
        return(ecg17)

def Ecg200(cls):
    cwd = os.getcwd()
    rate=480
    clsBase=int(cls)
    if cls==1:
        cls=-1
    else:        #cls==2
        cls=1
    train_set_path = cwd+"/ECG200/ECG200_TRAIN"
    test_set_path = cwd+"/ECG200/ECG200_TEST"
    train_data = np.loadtxt(train_set_path, delimiter=',') 
    test_data = np.loadtxt(test_set_path, delimiter=',')
    
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*clsBase
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*clsBase
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    return(trncls,tstcls)

def ECG5000(cls):
    rate=480
    train_set_path = cwd+"/ECG5000/ECG5000_TRAIN.txt"
    test_set_path = cwd+"/ECG5000/ECG5000_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def ECGFiveDays(cls):
    rate=480
    train_set_path = cwd+"/ECGFiveDays/ECGFiveDays_TRAIN.txt"
    test_set_path = cwd+"/ECGFiveDays/ECGFiveDays_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def TwoLeadECG(cls):
    rate=480
    train_set_path ="./TwoLeadECG/TwoLeadECG_TRAIN.txt"
    test_set_path = "./TwoLeadECG/TwoLeadECG_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def TwoLeadECG(cls):
    rate=480
    train_set_path ="./NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TRAIN.txt"
    test_set_path = "./NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def NonInvasiveFetalECGThorax1(cls):
    rate=480
    train_set_path ="./NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TRAIN.txt"
    test_set_path = "./NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def NonInvasiveFetalECGThorax2(cls):
    rate=480
    train_set_path ="./NonInvasiveFetalECGThorax2/NonInvasiveFetalECGThorax2_TRAIN.txt"
    test_set_path = "./NonInvasiveFetalECGThorax2/NonInvasiveFetalECGThorax2_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def EOGHorizontalSignal(cls):
    rate=480
    train_set_path ="./EOGHorizontalSignal/EOGHorizontalSignal_TRAIN.txt"
    test_set_path = "./EOGHorizontalSignal/EOGHorizontalSignal_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)


def EOGVerticalSignal(cls):
    rate=480
    train_set_path ="./EOGVerticalSignal/EOGVerticalSignal_TRAIN.txt"
    test_set_path = "./EOGVerticalSignal/EOGVerticalSignal_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def ACSF1(cls):
    rate=1460        #1460
    train_set_path ="./ACSF1/ACSF1_TRAIN.txt"
    test_set_path = "./ACSF1/ACSF1_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def DistalPhalanxOutlineCorrect(cls):
    rate=480        #81
    train_set_path ="./DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TRAIN"
    test_set_path = "./DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TEST"
    #train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    #test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    train_data = np.loadtxt(train_set_path, delimiter=',') 
    test_data = np.loadtxt(test_set_path, delimiter=',')
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def EthanolLevel(cls):
    rate=480        #1751
    train_set_path ="./EthanolLevel/EthanolLevel_TRAIN.txt"
    test_set_path = "./EthanolLevel/EthanolLevel_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def EthanolLevel0(cls):
    rate=1751        #1751
    train_set_path ="./EthanolLevel/EthanolLevel_TRAIN.txt"
    test_set_path = "./EthanolLevel/EthanolLevel_TEST.txt"
    train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def FISH(cls):
    rate=480        #500
    train_set_path ="./FISH/FISH_TRAIN"
    test_set_path = "./FISH/FISH_TEST"
    train_data = np.loadtxt(train_set_path, delimiter=',') 
    test_data = np.loadtxt(test_set_path, delimiter=',')
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def Ham(cls):
    rate=480        #431
    train_set_path ="./Ham/Ham_TRAIN"
    test_set_path = "./Ham/Ham_TEST"
    train_data = np.loadtxt(train_set_path, delimiter=',') 
    test_data = np.loadtxt(test_set_path, delimiter=',')
    #train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    #test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def Haptics(cls):
    rate=480        #1092
    train_set_path ="./Haptics/Haptics_TRAIN"
    test_set_path = "./Haptics/Haptics_TEST"
    train_data = np.loadtxt(train_set_path, delimiter=',') 
    test_data = np.loadtxt(test_set_path, delimiter=',')
    #train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    #test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)

def Haptics0(cls):
    rate=1092        #1092
    train_set_path ="./Haptics/Haptics_TRAIN"
    test_set_path = "./Haptics/Haptics_TEST"
    train_data = np.loadtxt(train_set_path, delimiter=',') 
    test_data = np.loadtxt(test_set_path, delimiter=',')
    #train_data = np.array(pd.read_csv(train_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    #test_data = np.array(pd.read_csv(test_set_path,sep='  ', header=None)) #,usecols=range(1,length+1)
    tr=np.array(train_data)
    ts=np.array(test_data)
    trncls=train_data[np.where(train_data[:,0]==cls),1:]
    if np.ndim(trncls)==3 :
        trncls=np.reshape(trncls,(np.shape(trncls)[1],np.shape(trncls)[2]))
    trncls=signal.resample(trncls,rate,axis=1)
    trnlbl=np.ones(len(trncls))
    trnlbl=trnlbl*cls
    trnlbl=np.int16(np.array([trnlbl]).T)
    trncls = np.concatenate((trncls,trnlbl), axis=1)
    tstcls=test_data[np.where(test_data[:,0]==cls),1:]
    if np.ndim(tstcls)==3 :
        tstcls=np.reshape(tstcls,(np.shape(tstcls)[1],np.shape(tstcls)[2]))
    tstcls=signal.resample(tstcls,rate,axis=1)
    tstlbl=np.ones(len(tstcls))
    tstlbl=tstlbl*cls
    tstlbl=np.int16(np.array([tstlbl]).T)
    tstcls = np.concatenate((tstcls,tstlbl), axis=1)
    if cls==1 :
        print('first_train_shape=',np.shape(tr))
        print('first_test_shape=',np.shape(ts))
        print('classes_quantity=',len(set(np.int16(tr[:,0]))))
        print('tr_lbls=\t',set(np.int16(tr[:,0])))
        print('Count_labels=',np.bincount(np.int16(tr[:,0]))[1:])
        print('max(train_feature_Altitude)=',np.max(tr))
        print('min(train_feature_Altitude)=',np.min(tr))
        print('first_train_sample=\n',tr[0])
    return(trncls,tstcls)