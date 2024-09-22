import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
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
tf.config.run_functions_eagerly(True)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,recall_score,precision_score
from tensorflow.keras import callbacks
from scipy.signal import find_peaks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Conv1D,Dropout,MaxPooling1D,MaxPooling2D,Flatten,Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation

#noise_len=90
#BUFFER_SIZE = 1000                        ### تعریفLoss و Optimizer ها:###
#BATCH_SIZE = 16
#generator = make_generator_model(noise_len)
#discriminator = make_discriminator_model()
# cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)   
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


def GAN_Gnrt(wndws,cls,AgTyp):
    cwd = os.getcwd()
    mx=np.max(wndws[3:-7])
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    z=int(cls)
    epch_sv_stp=10                        #Save Generator every 5-10-20 epoch for major-minor class
    num_examples_to_generate = 100        ### تعداد تولید نمونه در هر ایپاک در مولد
    aug_amnt_cfcnt=2             #1/2/3 are 50,000/100,000/200,000 samples for train GAN (300/600 epoch)
    snstvty=2               #1or2 ara low or high sensitivity to detect a beat
    print("\n\n\n Class ", str(cls)," Shape = ", np.shape(wndws))
    noise_len=90
    noise = tf.random.normal([1, noise_len])
    print('for class ', cls,'number of windows is = ',np.shape(wndws[0]))
     #dflt=50,000 or 100,000 تعداد نمونه هایی که برای آموزش در تمام ایپاک ها لازم است وارد شبکه شود:
     #EPOCHS = aug_amnt_cfcnt*int(50000/len(wndws))#(صد یا پنجاه هزار نمونه برای آموزش شبکه(به قسمت کلاستر رفته
    x_train=np.array(wndws[np.where(wndws[:,-1]==z),:-1]) #np.array(wndws):اگر آخر ردیف شماره کلاس نباشد
    if np.ndim(x_train)==3 :
        x_train=np.reshape(x_train,(np.shape(x_train)[0]*np.shape(x_train)[1],np.shape(x_train)[2]))    
    noise_dim = noise_len
    nd_nw_trn_smpl=2*mx_cls_smpl-len(x_train)
    bs_aug_pls=pd.read_csv(cwd+'\AugEvl\AugEvl'+str(cls)+str(AgTyp)+'0'+'.csv')#داده افزایی قبلی برای مجدد   
    os.chdir(cwd)    
    bs_aug_pls=np.array(bs_aug_pls)
    bs_aug_pls=np.array(bs_aug_pls[:nd_nw_trn_smpl,1:-6])
    # خوشه بندی پنجره ها
    #s_aug_pls=signal.resample(bs_aug_pls,480,axis=1)
    rateReal=len(bs_aug_pls[0])
    x_train=signal.resample(x_train,rateReal,axis=1)
    x_train=np.append(x_train,bs_aug_pls,axis=0)           ###### X=data is wndws
    data=np.array(x_train)               ###### X=data is wndws
    X = np.array(data)                   ###### X=data is wndws
    print('GAN Train Data shape=',np.shape(X))
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    linear = []
    ld = len(distortions)
    steep = (distortions[ld-1] - distortions[0]) / (ld - 1)
    c = distortions[ld-1] - steep * ld
    for x in range(0,ld):
        linear.append(steep * (x+1) + c)
    # And last we look for max distance between the points
    distances = np.array(linear)-np.array(distortions)
    max_index = distances.argmax(axis=0)+1
    # Print the optimal cluster number
    print('Optimal cluster number: ',max_index)
    k=int(max_index)
    kmeanModel = KMeans(n_clusters=k).fit(X)  ###### X is wndws 
    kmeanModel.fit(X)
    clstrs=np.int16(kmeanModel.labels_)
    smpl_rte=np.shape(X)[1]
    t=np.linspace(0,smpl_rte,smpl_rte)
    lngnrtds=int(np.shape(X)[0])
    pltrw=int(np.round(lngnrtds/2))+1

    for i in range(max_index):                                     #تعریف آرایه کلاسترها
        vars()['x_train'+str(i)]=np.array([])
    i=0
    for i in range (len(x_train)):
        j=clstrs[i]
        vars()['x_train'+str(j)]=np.append(vars()['x_train'+str(j)],x_train[i])#درج کلاستر در آرایه مجزا
    lngth=len(x_train[0])     
    for i in range(max_index):             ### تبدیل آرایه ی کلاسترها از یک ردیف به تعداد ردیف مساوی نمونه ها
        #print(len(vars()['x_train'+str(i)])/lngth)
        vars()['x_train'+str(i)]=np.reshape(vars()['x_train'+str(i)],(int(len(vars()['x_train'+str(i)])/lngth),lngth))
        #print('class ',cls,'clstr', i, 'clstr shape=',np.shape(vars()['x_train'+str(i)]))
    BUFFER_SIZE = 1000                                        ### تعریفLoss و Optimizer ها:###
    BATCH_SIZE = 8
    generator = make_generator_model(noise_len)
    discriminator = make_discriminator_model(rateReal)
    # cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)   
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range (max_index):
        gnrtds=np.empty((0,smpl_rte+3), float)# تعریف اولیه آرایه تولید نمونه برای هر کلاستر بطول سیگنال+6شاخصه
        clstr=i                                                       ## تولید سیگنال برای کلاستر آی
        EPOCHS = aug_amnt_cfcnt*int(50000/len(vars()['x_train'+str(i)]))
        if EPOCHS>4000:
            EPOCHS=4000
            epch_sv_stp=50     #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 160   #dflt=100,save 100 samples per epoch #40
        elif 4000>EPOCHS>3000:
            epch_sv_stp=30     #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 200   # dflt=100 #50
        elif 3000>EPOCHS>1000:
            epch_sv_stp=20      #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 300   # dflt=100 #75
        mdl=int(str(aug_amnt_cfcnt)+str(snstvty))             ## مدل با تعداد ایپاک و حساسیت به ضربان متفاوت
        x_train=np.array(vars()['x_train'+str(i)])       #احصای نمونه های موجود در خوشه
        mx=np.max(x_train)                               #####  نرمال سازی داده ها  #####
        mn=np.min(x_train)
        dis=np.max(x_train)-np.min(x_train)
        x_train = ((x_train - mn)-(dis/2)) / (dis/2)    # Normalize the signals to [-1, 1]
        #print('x_train shape=', np.shape(x_train))
        ##### Batch and shuffle the data And Train, Generate With GAN ######
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        gnrtds=train(train_dataset,x_train, EPOCHS, gnrtds, cls, mdl, clstr, num_examples_to_generate, epch_sv_stp, BATCH_SIZE, noise_dim, generator, discriminator) #تولید و ذخیره داده
        print('class=',z,'cluster=',i, 'epochs=',EPOCHS,' >> shape generate = ', np.shape(gnrtds))
        #pd.DataFrame(gnrtds).to_csv('{}/GAN_Cls{}_clstr_{}.csv'.format(cwd,cls,clstr)) #ذخیره  
        if i ==0 : 
            wdth_data=len(gnrtds[0])
            data=np.empty((0,wdth_data),float)
        data=np.append(data,gnrtds,axis=0)
    new=np.array(data[data[:,-2].argsort()]) #data Sorting Base of min Distance to same class 
    mx_ln_aug=np.int(10*mx_cls_smpl)
    if len(new)>mx_ln_aug:       
        new=np.array(new[:int(len(new)/4),:])#Remove generated data in first epochs
        ln_nw_Unzro=max(len(new),mx_ln_aug)
        last_indx=min(mx_ln_aug,len(new))
        ##Every Other sample Selecting:
        indxs=(np.arange(0,last_indx))*int(ln_nw_Unzro/mx_ln_aug) 
        new=np.array(new[indxs])
    new[:,:-3]=new[:,:-3]+1                        #UnNormalizing Data
    new[:,:-3]=np.float16((((new[:,:-3])*mx)/2)+1) #UnNormalizing Data
    print('cls ',z,' final shape, Generated by GAN =',np.shape(new))
    return(new[:,:-2])                          #-2 Remove Disance & variance
                  
def GAN_GnrtV(wndws,cls,AgTyp):
    cwd = os.getcwd()
    mx=np.max(wndws[3:-7])
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    z=int(cls)
    epch_sv_stp=10                        #Save Generator every 5-10-20 epoch for major-minor class
    num_examples_to_generate = 100        ### تعداد تولید نمونه در هر ایپاک در مولد
    aug_amnt_cfcnt=2             #1/2/3 are 50,000/100,000/200,000 samples for train GAN (300/600 epoch)
    snstvty=2               #1or2 ara low or high sensitivity to detect a beat
    print("\n\n\n Class ", str(cls)," Shape = ", np.shape(wndws))
    noise_len=90
    noise = tf.random.normal([1, noise_len])
    print('for class ', cls,'number of windows is = ',np.shape(wndws[0]))
     #dflt=50,000 or 100,000 تعداد نمونه هایی که برای آموزش در تمام ایپاک ها لازم است وارد شبکه شود:
     #EPOCHS = aug_amnt_cfcnt*int(50000/len(wndws))#(صد یا پنجاه هزار نمونه برای آموزش شبکه(به قسمت کلاستر رفته
    x_train=np.array(wndws[np.where(wndws[:,-1]==z),:-1]) #np.array(wndws):اگر آخر ردیف شماره کلاس نباشد
    if np.ndim(x_train)==3 :
        x_train=np.reshape(x_train,(np.shape(x_train)[0]*np.shape(x_train)[1],np.shape(x_train)[2]))    
    noise_dim = noise_len
    nd_nw_trn_smpl=2*mx_cls_smpl-len(x_train)
    bs_aug_pls=pd.read_csv(cwd+'\AugEvlV\AugEvl'+str(cls)+str(AgTyp)+'0'+'.csv')#داده افزایی قبلی برای مجدد   
    os.chdir(cwd)    
    bs_aug_pls=np.array(bs_aug_pls)
    bs_aug_pls=np.array(bs_aug_pls[:nd_nw_trn_smpl,1:-6])
    # خوشه بندی پنجره ها
    x_train=np.append(x_train,bs_aug_pls,axis=0)           ###### X=data is wndws
    data=np.array(x_train)               ###### X=data is wndws
    X = np.array(data)                   ###### X=data is wndws
    print('GAN Train Data shape=',np.shape(X))
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    linear = []
    ld = len(distortions)
    steep = (distortions[ld-1] - distortions[0]) / (ld - 1)
    c = distortions[ld-1] - steep * ld
    for x in range(0,ld):
        linear.append(steep * (x+1) + c)
    # And last we look for max distance between the points
    distances = np.array(linear)-np.array(distortions)
    max_index = distances.argmax(axis=0)+1
    # Print the optimal cluster number
    print('Optimal cluster number: ',max_index)
    k=int(max_index)
    kmeanModel = KMeans(n_clusters=k).fit(X)  ###### X is wndws 
    kmeanModel.fit(X)
    clstrs=np.int16(kmeanModel.labels_)
    smpl_rte=np.shape(X)[1]
    t=np.linspace(0,smpl_rte,smpl_rte)
    lngnrtds=int(np.shape(X)[0])
    pltrw=int(np.round(lngnrtds/2))+1

    for i in range(max_index):                                     #تعریف آرایه کلاسترها
        vars()['x_train'+str(i)]=np.array([])
    i=0
    for i in range (len(x_train)):
        j=clstrs[i]
        vars()['x_train'+str(j)]=np.append(vars()['x_train'+str(j)],x_train[i])#درج کلاستر در آرایه مجزا
    lngth=len(x_train[0])     
    for i in range(max_index):             ### تبدیل آرایه ی کلاسترها از یک ردیف به تعداد ردیف مساوی نمونه ها
        #print(len(vars()['x_train'+str(i)])/lngth)
        vars()['x_train'+str(i)]=np.reshape(vars()['x_train'+str(i)],(int(len(vars()['x_train'+str(i)])/lngth),lngth))
        #print('class ',cls,'clstr', i, 'clstr shape=',np.shape(vars()['x_train'+str(i)]))
    BUFFER_SIZE = 1000                                        ### تعریفLoss و Optimizer ها:###
    BATCH_SIZE = 8
    generator = make_generator_model(noise_len)
    discriminator = make_discriminator_model()
    # cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)   
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range (max_index):
        gnrtds=np.empty((0,smpl_rte+3), float)# تعریف اولیه آرایه تولید نمونه برای هر کلاستر بطول سیگنال+6شاخصه
        clstr=i                                                       ## تولید سیگنال برای کلاستر آی
        EPOCHS = aug_amnt_cfcnt*int(50000/len(vars()['x_train'+str(i)]))
        if EPOCHS>4000:
            EPOCHS=4000
            epch_sv_stp=50     #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 160   #dflt=100,save 100 samples per epoch #40
        elif 4000>EPOCHS>3000:
            epch_sv_stp=30     #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 200   # dflt=100 #50
        elif 3000>EPOCHS>1000:
            epch_sv_stp=20      #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 300   # dflt=100 #75
        mdl=int(str(aug_amnt_cfcnt)+str(snstvty))             ## مدل با تعداد ایپاک و حساسیت به ضربان متفاوت
        x_train=np.array(vars()['x_train'+str(i)])       #احصای نمونه های موجود در خوشه
        mx=np.max(x_train)                               #####  نرمال سازی داده ها  #####
        mn=np.min(x_train)
        dis=np.max(x_train)-np.min(x_train)
        x_train = ((x_train - mn)-(dis/2)) / (dis/2)    # Normalize the signals to [-1, 1]
        #print('x_train shape=', np.shape(x_train))
        ##### Batch and shuffle the data And Train, Generate With GAN ######
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        gnrtds=train(train_dataset,x_train, EPOCHS, gnrtds, cls, mdl, clstr, num_examples_to_generate, epch_sv_stp, BATCH_SIZE, noise_dim, generator, discriminator) #تولید و ذخیره داده
        print('class=',z,'cluster=',i, 'epochs=',EPOCHS,' >> shape generate = ', np.shape(gnrtds))
        #pd.DataFrame(gnrtds).to_csv('{}/GAN_Cls{}_clstr_{}.csv'.format(cwd,cls,clstr)) #ذخیره  
        if i ==0 : 
            wdth_data=len(gnrtds[0])
            data=np.empty((0,wdth_data),float)
        data=np.append(data,gnrtds,axis=0)
    new=np.array(data[data[:,-2].argsort()]) #data Sorting Base of min Distance to same class 
    mx_ln_aug=np.int(10*mx_cls_smpl)
    if len(new)>mx_ln_aug:       
        new=np.array(new[:int(len(new)/4),:])#Remove generated data in first epochs
        ln_nw_Unzro=max(len(new),mx_ln_aug)
        last_indx=min(mx_ln_aug,len(new))
        ##Every Other sample Selecting:
        indxs=(np.arange(0,last_indx))*int(ln_nw_Unzro/mx_ln_aug) 
        new=np.array(new[indxs])
    new[:,:-3]=new[:,:-3]+1                        #UnNormalizing Data
    new[:,:-3]=np.float16((((new[:,:-3])*mx)/2)+1) #UnNormalizing Data
    print('cls ',z,' final shape, Generated by GAN =',np.shape(new))
    return(new[:,:-2])                          #-2 Remove Disance & variance

def GAN_GnrtH(wndws,cls,AgTyp):
    cwd = os.getcwd()
    mx=np.max(wndws[3:-7])
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    z=int(cls)
    epch_sv_stp=10                        #Save Generator every 5-10-20 epoch for major-minor class
    num_examples_to_generate = 100        ### تعداد تولید نمونه در هر ایپاک در مولد
    aug_amnt_cfcnt=2             #1/2/3 are 50,000/100,000/200,000 samples for train GAN (300/600 epoch)
    snstvty=2               #1or2 ara low or high sensitivity to detect a beat
    print("\n\n\n Class ", str(cls)," Shape = ", np.shape(wndws))
    noise_len=90
    noise = tf.random.normal([1, noise_len])
    print('for class ', cls,'number of windows is = ',np.shape(wndws[0]))
     #dflt=50,000 or 100,000 تعداد نمونه هایی که برای آموزش در تمام ایپاک ها لازم است وارد شبکه شود:
     #EPOCHS = aug_amnt_cfcnt*int(50000/len(wndws))#(صد یا پنجاه هزار نمونه برای آموزش شبکه(به قسمت کلاستر رفته
    x_train=np.array(wndws[np.where(wndws[:,-1]==z),:-1]) #np.array(wndws):اگر آخر ردیف شماره کلاس نباشد
    if np.ndim(x_train)==3 :
        x_train=np.reshape(x_train,(np.shape(x_train)[0]*np.shape(x_train)[1],np.shape(x_train)[2]))    
    noise_dim = noise_len
    nd_nw_trn_smpl=2*mx_cls_smpl-len(x_train)
    bs_aug_pls=pd.read_csv(cwd+'\AugEvlH\AugEvl'+str(cls)+str(AgTyp)+'0'+'.csv')#داده افزایی قبلی برای مجدد   
    os.chdir(cwd)    
    bs_aug_pls=np.array(bs_aug_pls)
    bs_aug_pls=np.array(bs_aug_pls[:nd_nw_trn_smpl,1:-6])
    # خوشه بندی پنجره ها
    x_train=np.append(x_train,bs_aug_pls,axis=0)           ###### X=data is wndws
    data=np.array(x_train)               ###### X=data is wndws
    X = np.array(data)                   ###### X=data is wndws
    print('GAN Train Data shape=',np.shape(X))
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    linear = []
    ld = len(distortions)
    steep = (distortions[ld-1] - distortions[0]) / (ld - 1)
    c = distortions[ld-1] - steep * ld
    for x in range(0,ld):
        linear.append(steep * (x+1) + c)
    # And last we look for max distance between the points
    distances = np.array(linear)-np.array(distortions)
    max_index = distances.argmax(axis=0)+1
    # Print the optimal cluster number
    print('Optimal cluster number: ',max_index)
    k=int(max_index)
    kmeanModel = KMeans(n_clusters=k).fit(X)  ###### X is wndws 
    kmeanModel.fit(X)
    clstrs=np.int16(kmeanModel.labels_)
    smpl_rte=np.shape(X)[1]
    t=np.linspace(0,smpl_rte,smpl_rte)
    lngnrtds=int(np.shape(X)[0])
    pltrw=int(np.round(lngnrtds/2))+1

    for i in range(max_index):                                     #تعریف آرایه کلاسترها
        vars()['x_train'+str(i)]=np.array([])
    i=0
    for i in range (len(x_train)):
        j=clstrs[i]
        vars()['x_train'+str(j)]=np.append(vars()['x_train'+str(j)],x_train[i])#درج کلاستر در آرایه مجزا
    lngth=len(x_train[0])     
    for i in range(max_index):             ### تبدیل آرایه ی کلاسترها از یک ردیف به تعداد ردیف مساوی نمونه ها
        #print(len(vars()['x_train'+str(i)])/lngth)
        vars()['x_train'+str(i)]=np.reshape(vars()['x_train'+str(i)],(int(len(vars()['x_train'+str(i)])/lngth),lngth))
        #print('class ',cls,'clstr', i, 'clstr shape=',np.shape(vars()['x_train'+str(i)]))
    BUFFER_SIZE = 1000                                        ### تعریفLoss و Optimizer ها:###
    BATCH_SIZE = 8
    generator = make_generator_model(noise_len)
    discriminator = make_discriminator_model()
    # cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)   
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range (max_index):
        gnrtds=np.empty((0,smpl_rte+3), float)# تعریف اولیه آرایه تولید نمونه برای هر کلاستر بطول سیگنال+6شاخصه
        clstr=i                                                       ## تولید سیگنال برای کلاستر آی
        EPOCHS = aug_amnt_cfcnt*int(50000/len(vars()['x_train'+str(i)]))
        if EPOCHS>4000:
            EPOCHS=4000
            epch_sv_stp=50     #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 160   #dflt=100,save 100 samples per epoch #40
        elif 4000>EPOCHS>3000:
            epch_sv_stp=30     #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 200   # dflt=100 #50
        elif 3000>EPOCHS>1000:
            epch_sv_stp=20      #dflt=10,Save Generator every 5-10-20 epoch for major-minor class
            num_examples_to_generate = 300   # dflt=100 #75
        mdl=int(str(aug_amnt_cfcnt)+str(snstvty))             ## مدل با تعداد ایپاک و حساسیت به ضربان متفاوت
        x_train=np.array(vars()['x_train'+str(i)])       #احصای نمونه های موجود در خوشه
        mx=np.max(x_train)                               #####  نرمال سازی داده ها  #####
        mn=np.min(x_train)
        dis=np.max(x_train)-np.min(x_train)
        x_train = ((x_train - mn)-(dis/2)) / (dis/2)    # Normalize the signals to [-1, 1]
        #print('x_train shape=', np.shape(x_train))
        ##### Batch and shuffle the data And Train, Generate With GAN ######
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        gnrtds=train(train_dataset,x_train, EPOCHS, gnrtds, cls, mdl, clstr, num_examples_to_generate, epch_sv_stp, BATCH_SIZE, noise_dim, generator, discriminator) #تولید و ذخیره داده
        print('class=',z,'cluster=',i, 'epochs=',EPOCHS,' >> shape generate = ', np.shape(gnrtds))
        #pd.DataFrame(gnrtds).to_csv('{}/GAN_Cls{}_clstr_{}.csv'.format(cwd,cls,clstr)) #ذخیره  
        if i ==0 : 
            wdth_data=len(gnrtds[0])
            data=np.empty((0,wdth_data),float)
        data=np.append(data,gnrtds,axis=0)
    new=np.array(data[data[:,-2].argsort()]) #data Sorting Base of min Distance to same class 
    mx_ln_aug=np.int(10*mx_cls_smpl)
    if len(new)>mx_ln_aug:       
        new=np.array(new[:int(len(new)/4),:])#Remove generated data in first epochs
        ln_nw_Unzro=max(len(new),mx_ln_aug)
        last_indx=min(mx_ln_aug,len(new))
        ##Every Other sample Selecting:
        indxs=(np.arange(0,last_indx))*int(ln_nw_Unzro/mx_ln_aug) 
        new=np.array(new[indxs])
    new[:,:-3]=new[:,:-3]+1                        #UnNormalizing Data
    new[:,:-3]=np.float16((((new[:,:-3])*mx)/2)+1) #UnNormalizing Data
    print('cls ',z,' final shape, Generated by GAN =',np.shape(new))
    return(new[:,:-2])                          #-2 Remove Disance & variance


def make_generator_model(noise_len):
    model = tf.keras.Sequential()
    model.add(layers.Dense(20*2048, use_bias=False, input_shape=(noise_len,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((20, 2048)))
    #assert model.output_shape == (None, 7, 4096) # Note: None is the batch size
    #print('None, 5, 2048 ==>', model.output_shape)

    model.add(layers.Conv1DTranspose(1024, (5), strides=(3), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 4096)
    #print('None, 25, 1024 ==>', model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(512, (5), strides=(2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 4096)
    #print('None, 225, 512 ==>', model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(256, (5), strides=(2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 14, 2048)
    #print('None, 450, 256==>', model.output_shape)
    model.add(MaxPooling1D(pool_size=3,strides=3))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(128, (5), strides=(2), padding='same', use_bias=False))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    #assert model.output_shape == (None, 14, 1024)
    #print('None, noise_len0, 128==>', model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, (5), strides=(1), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 512)
    #print('None, 7, 512==>', model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(1, (5), strides=(1), padding='same', use_bias=False, activation='tanh'))
    #print('None, 3600, 1==>', model.output_shape)
    #assert model.output_shape == (None, 3600, 1)
    return model

def make_discriminator_model(rateReal=480):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, (5), strides=(2), padding='same',input_shape=[rateReal, 1])) ###input_shape=[480, 1]))###
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #print('None, 1, 128==>', model.output_shape)

    model.add(layers.Conv1D(128, (5), strides=(2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #print('None, 2, 128==>', model.output_shape)

    model.add(layers.Conv1D(256, (5), strides=(2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))    
    #print('None, 3, 128==>', model.output_shape)

    model.add(layers.Conv1D(512, (5), strides=(2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #print('None, 4, 512==>', model.output_shape)
    
    model.add(layers.Conv1D(1024, (5), strides=(2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #print('None, 5, 128==>', model.output_shape)

    model.add(layers.Conv1D(2048, (5), strides=(2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #print('None, 6, 128==>', model.output_shape)

    model.add(layers.Flatten())
    #print('None, fltn, 128==>', model.output_shape)
    model.add(layers.Dense(1))
    #print('None, 7, 128==>', model.output_shape)
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

@tf.function
def train_step(generator, discriminator, signals, BATCH_SIZE, noise_dim):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      print('nise shape=',np.shape(noise))
      generated_signals = generator(noise, training=True)
      print('generated_signals shape=',np.shape(generated_signals))
      real_output = discriminator(signals, training=True)
      fake_output = discriminator(generated_signals, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def dis_var(dataset2,P):             #برگرداندن فاصله و واریانس بردار فاصله ی هر کدام از ردیف های پی از دیتاست2
    var1,i,h,w=[],0,np.shape(P)[0],np.shape(P)[1]
    end=len(dataset2[0])
    for i in range (h):
        ds1 = dataset2-P[i,:end]      #.reshape(-1,1) #distance of every training instance
        dist_array=(ds1*10)**3       #معیار فاصله با حساسیت به توان 3 (حفظ علامت و تاثیر بیشتر تفاوت زیاد در نقاط)
        ds1=min(np.sum(np.absolute(dist_array),axis=1))
        #print('dist2 = ', dist2)
        vr1=min(np.var((dist_array),axis=1))
        var1.append(ds1)
        var1.append(vr1)
    var1=np.reshape(var1,(int(len(var1)/2),2))
    return(var1)

    
def dis_var0(dataset2,P):
    var1,i,h,w=[],0,np.shape(P)[0],np.shape(P)[1]
    for i in range (h):
        ds1 = dataset2-P[i,:-4]      #.reshape(-1,1) #distance of every training instance
        dist_array=(ds1*10)**3       #معیار فاصله با حساسیت به توان 3 (حفظ علامت و تاثیر بیشتر تفاوت زیاد در نقاط)
        ds1=min(np.sum(np.absolute(dist_array),axis=1))
        #print('dist2 = ', dist2)
        vr1=min(np.var((dist_array),axis=1))
        var1.append(ds1)
        var1.append(vr1)
    var1=np.reshape(var1,(int(len(var1)/2),2))
    
    return(var1)
    
def generate_and_save_signals(dataset1, x_trn1, model, epoch1, test_input, gnrtds1, cls1, mdl1, clstr1): 
    #کلاس-مدل-خوشه-ایپاک-تفاوت با نزدیکترین-واریانسش
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions=np.reshape(predictions,(np.shape(predictions)[0],np.shape(predictions)[1]))
    clm0vlu=np.zeros(len(predictions))
    cls_clm,mdl_clm,clstr_clm,epch_clm=clm0vlu,clm0vlu,clm0vlu,clm0vlu
    cls_clm,mdl_clm,clstr_clm,epch_clm=np.int16(cls_clm+cls1),np.int16(mdl_clm+mdl1),np.int16(clstr_clm+clstr1),np.int16(epch_clm+epoch1)
    cls_clm,mdl_clm,clstr_clm,epch_clm=np.transpose([cls_clm]),np.transpose([mdl_clm]),np.transpose([clstr_clm]),np.transpose([epch_clm])
    predictions=np.concatenate((predictions,cls_clm),axis=1)
    #predictions=np.concatenate((predictions,clstr_clm),axis=1)        #افزودن شاخص ها به نمونه تولیدی
    #predictions=np.concatenate((predictions,mdl_clm),axis=1)
    #predictions=np.concatenate((predictions,epch_clm),axis=1)
    predictions=np.concatenate((predictions,dis_var(x_trn1, predictions)),axis=1)
    #print('pred.shape=', np.shape(predictions))
    gnrtds1=np.append(gnrtds1,predictions,axis=0)
    #print('shape generated=', np.shape(gnrtds1))
    return(gnrtds1)

def train(dataset, x_trn, epochs, gnrtds, cls0, mdl0, clstr0, num_examples_to_generate, epch_sv_stp0,BATCH_SIZE, noise_dim, generator, discriminator):
    for epoch in range(epochs):
        #print('epoch = ', epoch)
        #start = time.time()
        for signal_batch in dataset:
            train_step(generator, discriminator, signal_batch, BATCH_SIZE, noise_dim)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])      #Input Noise to Generator
        #clear_output(wait=True)
        gnrtds=generate_and_save_signals(dataset, x_trn, generator, epoch, seed, gnrtds, cls0, mdl0, clstr0)
        #print ('Time for epoch {}/{} is {} sec'.format(epoch + 1, epochs, time.time()-start))
    #clear_output(wait=True)
    return(gnrtds)