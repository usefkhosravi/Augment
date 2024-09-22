import os
import glob
import scipy
import random
import bisect
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from preprocess import *                         #ماژول استخراج پنجره ها
from data import *                          #ماژول محلی ورود داده ها
from augment import *                            #ماژول های داده افزایی
from lstm_cnn import *                           #ماژول های داده افزایی با lstm_cnn
import augment
import importlib
importlib.reload(augment)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,recall_score,precision_score
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Conv1D,Dropout,MaxPooling1D,MaxPooling2D,Flatten,Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
#import cupy as cp
cwd = os.getcwd() #


def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def maxindx(ar):
    ar_cpy=np.array(ar)
    pred=np.argmax(ar,axis=1)
    for i in range (len(pred)):
        if pred[i]==0:
            ar_cpy[i,0]=-1000
    pred=np.argmax(ar_cpy,axis=1)
    return(pred)
    
def euclidean_dist(pointA, pointB):
    distance = np.square(pointA - pointB) # (ai-bi)**2 for every point in the vectors
    distance = np.sum(distance) # adds all values
    distance = np.sqrt(distance) 
    return distance

def distance_from_all_training(P_aray,rw):
    dist_array = np.array([])
    for train_point in P_aray:
        dist = euclidean_dist(rw, train_point)
        dist_array = np.append(dist_array, dist)
    return dist_array


def euclidean_dist_f(pointA, pointB):
    distance = np.square(pointA - pointB) # (ai-bi)**2 for every point in the vectors
    ##distance = np.sum(distance) # adds all values
    distance = np.sqrt(distance) 
    return (distance)

def distance_from_all_training_f(P_aray,rw):
    #print('shape of array in distance_from_all_training_f =',np.shape(P_aray)[0],np.shape(P_aray)[1])
    dist_array = np.array([])
    for train_point in P_aray:
        dist = euclidean_dist_f(rw, train_point)
        dist_array = np.append(dist_array, dist)
    dist_array=dist_array.reshape(np.shape(P_aray)[0],np.shape(P_aray)[1])
    return (dist_array)

def cdf(weights):                           # تابع تولید احتمال تجمعی برای ارزش های نمونه های کلاس حداقل
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return (result)

def choice(population, weights):                        # تابع انتخاب چرخ رولت
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)                  #محاسبه ی احتمال تجمعی برای تولید بازه هایی برای وزن ها
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)        #اندیس بازه ی انتخاب شده از بین بازه های موجود بین صفر و یک
    if (idx>=len(population)):
        idx=len(population)-1
    return (idx)

def train_permut(traindata):
    traindata0=np.random.permutation(traindata)
    trainlabel0=np.int32(traindata0[:,-1])
    traindata0=np.delete(traindata0, -1, 1)
    return(traindata0,trainlabel0)

def valuation(P,tr,trlt,grade=1):                 #محاسبه ی ارزش هر نمونه ی اقلیت (مثبت) جهت تکثیر
    k=8
    importnc=[]
    for i in range (len(P)):
        dist_array = np.square(tr-P[i])
        dist_array = np.array([np.sum(np.sqrt(dist_array),axis=1)]).T
        neighbors = np.concatenate((dist_array, trlt), axis = 1) 
        neighbors_sorted = neighbors[neighbors[:, 0].argsort()]#sorts instances points base on distance
        k_neighbors = neighbors_sorted[:k]                       # selects k-nearest neighbors
        Num_kN=0                                                    # Number Of Major Class Instances
        dis_k=0                                                   #Total Distances of K neighbors
        dis_kN=0                                                  #Total Distance of Major Class 
        for j in range (k):
            dis_k=k_neighbors[j,0]+dis_k
            if k_neighbors[j,1]==0 :                             #if lable class==0 Negative_count++
                Num_kN=Num_kN+1
                dis_kN=k_neighbors[j,0]+dis_kN                     #lable 1 Positive and 0 Negative
        Num_kN=Num_kN/k
        if dis_k==0:
            dis_k=1
            dis_kN=0
        dis_kN=dis_kN/dis_k
        importnc.append(Num_kN+dis_kN)
    importnc=np.array(importnc)
    importnc=importnc**grade
    return(importnc)

def valuation_old(P,tr,trlt,grade=1):                 #محاسبه ی ارزش هر نمونه ی اقلیت (مثبت) جهت تکثیر
    k=8
    importnc=[]
    for i in range (len(P)):
        dist_array = distance_from_all_training(tr,P[i]).reshape(-1,1) #distance from every training instance
        #print('shape dist_array, trlt =',np.shape(dist_array),np.shape(trlt))
        neighbors = np.concatenate((dist_array, trlt), axis = 1) 
        neighbors_sorted = neighbors[neighbors[:, 0].argsort()]#sorts instances points base on distance
        k_neighbors = neighbors_sorted[:k]                       # selects k-nearest neighbors
        imk=0                                                    # Number Of Major Class Instances
        disk=0                                                   #Total Distances of K neighbors
        diskN=0                                                  #Total Distance of Major Class neighbors
        for j in range (k):
            disk=k_neighbors[j,0]+disk
            if k_neighbors[j,1]==0 :                             #if lable class==0 Negative_count++
                imk=imk+1
                diskN=k_neighbors[j,0]+diskN                     #lable 1 Positive and 0 Negative
        imk=imk/k
        if disk==0:
            disk=1
        diskN=diskN/disk
        importnc.append(imk+diskN)
    importnc=np.array(importnc)
    importnc=importnc**grade
    return(importnc)

def euclidean_dist_cupy(pointA, pointB):
    distance = cp.square(pointA - pointB) # (ai-bi)**2 for every point in the vectors
    distance = cp.sum(distance)           # adds all values
    distance = cp.sqrt(distance) 
    return distance

def distance_from_all_training_cupy(P_aray,rw):
    dist_array = cp.array([])
    for train_point in P_aray:
        dist = euclidean_dist_cupy(rw, train_point)
        dist_array = cp.append(dist_array, dist)
    return dist_array

def valuation_cupy(P,tr,trlt,grade=1):                 #محاسبه ی ارزش هر نمونه ی اقلیت (مثبت) جهت تکثیر
    k=8
    importnc=[]
    for i in range (len(P)):
        dist_array = distance_from_all_training_cupy(tr,P[i]).reshape(-1,1) #distance from every training instance
        #print('shape dist_array, trlt =',cp.shape(dist_array),cp.shape(trlt))
        neighbors = cp.concatenate((dist_array, trlt), axis = 1) 
        neighbors_sorted = neighbors[neighbors[:, 0].argsort()]#sorts instances points base on distance
        k_neighbors = neighbors_sorted[:k]                       # selects k-nearest neighbors
        imk=0                                                    # Number Of Major Class Instances
        disk=0                                                   #Total Distances of K neighbors
        diskN=0                                                  #Total Distance of Major Class neighbors
        for j in range (k):
            disk=k_neighbors[j,0]+disk
            if k_neighbors[j,1]==0 :                             #if lable class==0 Negative_count++
                imk=imk+1
                diskN=k_neighbors[j,0]+diskN                     #lable 1 Positive and 0 Negative
        imk=imk/k
        if disk==0:
            disk=1
        diskN=diskN/disk
        importnc.append(imk+diskN)
    importnc=cp.array(importnc)
    importnc=importnc**grade
    return(importnc)

def sigma_f(P):                                #محاسبه ی واریانس هر نمونه ی اقلیت (مثبت) جهت تکثیر
    var=[]
    i=0
    h=np.shape(P)[0]
    try:                                       #رفع خطای اندیس برای آرایه 1 بعدی
        w =  np.shape(P)[1]
    except IndexError:
        w = 1
    for i in range (h):
        j=0
        dist_arrayP = distance_from_all_training_f(P,P[i])#.reshape(-1,1) #distance of every training instance
        #print('dist_arrayP=\n',dist_arrayP)
        #print(dist_arrayP.shape())
        onenn=np.zeros(np.shape(P)[1])
        for j in range (w):
            g=0
            for g in range (h):
                #print('g&j=',g,j)
                if dist_arrayP[g,j]==0:
                    dist_arrayP[g,j]=100000
            g=0
            #minj=np.argmin(dist_arrayP(:,j))
            onenn[j]=min(dist_arrayP[:,j])
        var.append(onenn)
    return(var)

def sigma(P):                                #محاسبه ی واریانس هر نمونه ی اقلیت (مثبت) جهت تکثیر
    var=[]
    for i in range (len(P)):
        dist_arrayP = distance_from_all_training(P,P[i]).reshape(-1,1) #distance of every training instance
        g=0
        for g in range (len(P)):
            if dist_arrayP[g]==0:
                dist_arrayP[g]=100
                break
        g=0
        onenn=np.argmin(dist_arrayP)
        onenn=int(onenn)
        vari = np.sqrt(np.square(P[i] - P[onenn]))
        var.append(vari)
    return(var)

#def varpeak(pks):

    
def rescale_width(x,xbar):
    strch=PosNormal(x, xbar)
    f=np.array(x[:int(strch*len(x))])
    f = signal.resample(f, int(len(f)/strch))
    return(f)

def PosNormal(x, xbar):
    delta=np.abs((x-xbar)/2.5)
    xnew = np.random.normal(x,delta)
    return(xnew/x if (x<xnew<xbar or xbar<xnew<x) else PosNormal(x,xbar))
#def rescale_height(x,xbar):


def wndwng_sig_cupy(x,snstvty=2):                   #روش تشخیص ایپاک ماتریس 2 بعدی
    x=np.array(x.get())
    smpl_rte=480
    i=0
    wndws=np.array([])
    #for i in range (1,2):                 #print(np.shape(x))
    pnts=len(x[0])-1                       #طول سیگنال منهای یک 
    if snstvty==1 :                        #تشخیص ایپاک پایه
        for j in range (np.shape(x)[0]):
            xj=np.array(x[j])
            xj=np.array(xj[1:-2])
            peaks,_ = find_peaks(xj, height=0.03*max(xj), distance=200,prominence=300, width=3)
            if len(peaks)<6 :
                peaks,_ = find_peaks(xj, height=0.03*max(xj), distance=160,prominence=100, width=3)
                #print('class ', i, 'Row = ', j, 'have ',len(peaks),'peaks')
            if len(peaks)>1 :
                for z in range (len(peaks)):
                    if (peaks[z]<pnts and peaks[z]!=peaks[-1]) :
                        win=np.array(xj[peaks[z]:peaks[z+1]])
                        win = signal.resample(win, smpl_rte)
                        #if len(win)==smpl_rte
                        wndws=np.append(wndws, win)
                        #else:  #print('1- len(win) not ',smpl_rte,len(win))
            #else: #print('2- len(peaks) not > 2',len(peaks))
    if snstvty==2 :                        #تشخیص ایپاک حساس تر
        #print('np.shape(x)[0]=',np.shape(x)[0])
        for j in range (np.shape(x)[0]):
            xj=np.array(x[j])
            xj=np.array(xj[1:-2])
            peaks,_ = find_peaks(xj, height=0.02*max(xj), distance=160,prominence=200, width=3)
            if len(peaks)<6 :
                peaks,_ = find_peaks(xj, height=0.01*max(xj), distance=120,prominence=60, width=3)
            if len(peaks)>1 :
                for z in range (len(peaks)):
                    if (peaks[z]<pnts and peaks[z]!=peaks[-1]) :   #peaks[z]<3100
                        win=np.array(xj[peaks[z]:peaks[z+1]])
                        win = signal.resample(win, smpl_rte)
                        #if len(win)==smpl_rte :
                        wndws=np.append(wndws, win)

    rows=np.int(len(wndws)/(smpl_rte))            #/(smpl_rte+1))
    wndws0=np.reshape(wndws,(rows,(smpl_rte)))    #(wndws,(rows,(smpl_rte+1)))
    return(cp.array(wndws0))


def wndwng_sig(x,snstvty=2):                   #روش تشخیص ایپاک ماتریس 2 بعدی
    smpl_rte=480
    i=0
    wndws=np.array([])
    #for i in range (1,2):                 #print(np.shape(x))
    pnts=len(x[0])-1                       #طول سیگنال منهای یک 
    if snstvty==1 :                        #تشخیص ایپاک پایه
        for j in range (np.shape(x)[0]):
            xj=np.array(x[j])
            xj=np.array(xj[1:-2])
            peaks,_ = find_peaks(xj, height=0.03*max(xj), distance=200,prominence=300, width=3)
            if len(peaks)<6 :
                peaks,_ = find_peaks(xj, height=0.03*max(xj), distance=160,prominence=100, width=3)
                #print('class ', i, 'Row = ', j, 'have ',len(peaks),'peaks')
            if len(peaks)>1 :
                for z in range (len(peaks)):
                    if (peaks[z]<pnts and peaks[z]!=peaks[-1]) :
                        win=np.array(xj[peaks[z]:peaks[z+1]])
                        win = signal.resample(win, smpl_rte)
                        #if len(win)==smpl_rte
                        wndws=np.append(wndws, win)
                        #else:  #print('1- len(win) not ',smpl_rte,len(win))
            #else: #print('2- len(peaks) not > 2',len(peaks))

    if snstvty==2 :                        #تشخیص ایپاک حساس تر
        #print('np.shape(x)[0]=',np.shape(x)[0])
        for j in range (np.shape(x)[0]):
            xj=np.array(x[j])
            xj=np.array(xj[1:-2])
            peaks,_ = find_peaks(xj, height=0.02*max(xj), distance=160,prominence=200, width=3)
            if len(peaks)<6 :
                peaks,_ = find_peaks(xj, height=0.01*max(xj), distance=120,prominence=60, width=3)
            if len(peaks)>1 :
                for z in range (len(peaks)):
                    if (peaks[z]<pnts and peaks[z]!=peaks[-1]) :   #peaks[z]<3100
                        win=np.array(xj[peaks[z]:peaks[z+1]])
                        win = signal.resample(win, smpl_rte)
                        #if len(win)==smpl_rte :
                        wndws=np.append(wndws, win)

    rows=np.int(len(wndws)/(smpl_rte))            #/(smpl_rte+1))
    wndws0=np.reshape(wndws,(rows,(smpl_rte)))    #(wndws,(rows,(smpl_rte+1)))
    return(wndws0)

def peak_hght(sig):
    peaks,_ = find_peaks(sig, height=0.03*max(sig), distance=200,prominence=300, width=3)
    if len(peaks)<6 :
        peaks,_ = find_peaks(sig, height=0.03*max(sig), distance=160,prominence=100, width=3)
        #print('class ', i, 'Row = ', j, 'have ',len(peaks),'peaks') 
    return(sig[peaks])

def dist_hght(pointA, pointB):
    A=np.mean(peak_hght(pointA))
    B=np.mean(peak_hght(pointB))
    distance = abs(A-B)
    return distance

def distance_from_all_training_hght(P_aray,rw):
    dist_array = np.array([])
    for train_point in P_aray:
        dist = dist_hght(rw, train_point)
        dist_array = np.append(dist_array, dist)
    return dist_array

def sigma_hght(P):                                #محاسبه ی واریانس هر نمونه ی اقلیت (مثبت) جهت تکثیر
    var=[]
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    for i in range (len(P)):
        dist_arrayP = distance_from_all_training_hght(P,P[i]).reshape(-1,1) #distance of every training instance
        g=0
        for g in range (len(P)):
            if dist_arrayP[g]==0:
                dist_arrayP[g]=100
                break
        g=0
        onenn=np.argmin(dist_arrayP)
        onenn=int(onenn)
        vari = dist_arrayP[onenn]
        var.append(vari)
    var=np.array(var)
    return(var)

def scl_rndm(ecg,cls,volum=700,mag=.1):    #Scale Real
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        Pn= (np.random.normal(0,mag)+1)*P[indx]
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                              #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)

def scl_rndm_beat(ecg,cls,volum=700,mag=.1):    #Scale Real
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        Pn= (np.random.normal(0,mag)+1)*P[indx]
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                              #تکمیل داده های آموزشی جدید
    p_wn=np.array(P)
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)


def MgWrp_rndm(ecg,cls,volum=700,mag=.1):
    volum=volum+2
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    for z in range (frqncy):                       #########تولید نمونه های جدید#########       
        indx=np.random.randint(lnp)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X, sigma = mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn) 

def MgWrp_rndm_beat(ecg,cls,volum=700,mag=.1):
    volum=volum+2
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    for z in range (frqncy):                       #########تولید نمونه های جدید#########       
        indx=np.random.randint(lnp)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X, sigma = mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(P)
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn) 


def TimWrp_rndm(ecg,cls,volum=700,mag=.1):
    volum=volum+2
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(time_warp(X,sigma=mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)

def TimWrp_rndm_beat(ecg,cls,volum=700,mag=.1):
    volum=volum+2
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(time_warp(X,sigma=mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(P)
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)

def GDO_rndm(ecg,cls,volum=700,mag=.1):     #Gaussian Distribution Oversampling
    volum=volum+2
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    wdthp=len(P[0])
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        var=np.random.normal(0,mag,wdthp)+1
        Pn=P[indx]*var
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)  

def GDO_rndm_beat(ecg,cls,volum=700,mag=.1):     #Gaussian Distribution Oversampling
    volum=volum+2
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    lnp=len(P)
    wdthp=len(P[0])
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        var=np.random.normal(0,mag,wdthp)+1
        Pn=P[indx]*var
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(P)
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)  

def frqnc_rndm(furir_windws,cls,volum=700,mag=.1):     #Gaussian Distribution Oversampling
    volum=volum+2
    ecg=np.array(furir_windws)
    i=int(cls)
    c,alpha,PL=0,mag+1,1
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    ###################################### Real+Imaginary Part #######################################
    realP=P.real
    ImgP=P.imag
    lnp=len(P)
    wdthp=len(P[0])
    for z in range (frqncy):                     #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        old=np.array(P[indx])        
        var1=np.random.normal(0,mag,wdthp)+1
        old.real=old.real*var1                            #old.real #تولید نمونه
        var2=np.random.normal(0,mag,wdthp)+1
        old.imag=ImgP[indx]*var2                          #تولید نمونه  
        new=np.append(new,old)
    new=new.reshape(frqncy,int(len(new)/frqncy))
    new=np.fft.ifft(new)
    P = np.array(new)                                     #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    p_wn = np.int16(np.concatenate((P,pl), axis=1))
    return(np.float16(p_wn))                            #Generated signals and labels
    
def frqnc_rndm_beat(furir_windws,cls,volum=700,mag=.1):     #Gaussian Distribution Oversampling
    volum=volum+2
    ecg=np.array(furir_windws)
    i=int(cls)
    c,alpha,PL=0,mag+1,1
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    ###################################### Real+Imaginary Part #######################################
    realP=P.real
    ImgP=P.imag
    lnp=len(P)
    wdthp=len(P[0])
    for z in range (frqncy):                     #########تولید نمونه های جدید#########
        indx=np.random.randint(lnp)                #انتخاب نمونه
        old=np.array(P[indx])        
        var1=np.random.normal(0,mag,wdthp)+1
        old.real=old.real*var1                            #old.real #تولید نمونه
        var2=np.random.normal(0,mag,wdthp)+1
        old.imag=ImgP[indx]*var2                          #تولید نمونه  
        new=np.append(new,old)
    new=new.reshape(frqncy,int(len(new)/frqncy))
    new=np.fft.ifft(new)
    P = np.array(new)                                     #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    p_wn = np.float16(np.concatenate((P,pl), axis=1))
    return(np.float16(p_wn))                            #Generated signals and labels


def scl0(ecg,cls,volum=273,mag=.1): #Scale avrgpeak
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                   # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        #print('indx', indx)
        avrgpeak=np.mean(peak_hght(P[indx]))
        avrge= (np.random.normal(0,mag)+1)*np.mean(var[indx])+np.zeros_like(var[indx])
        Pn=P[indx]*abs((np.random.normal(0,avrge)+avrgpeak)/avrgpeak)
        #Pn=np.random.normal(P[indx], alpha*var[indx])#تولید نمونه گوسی با واریانس نزدیکترین نمونه هم کلاس
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)


def scl(ecg,cls,volum=273,mag=.1):    #Scale Real
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]==i :                    # != !!to ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    #var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        #print('indx', indx)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        Pn= (np.random.normal(0,mag)+1)*P[indx]
        #Pn=np.random.normal(P[indx], alpha*var[indx])#تولید نمونه گوسی با واریانس نزدیکترین نمونه هم کلاس
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)

def sclRaw(ecg,cls,volum=273,mag=.1):              #Scale Real
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]==i :                    # != !!to ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    #var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        Pn= (np.random.normal(0,mag)+1)*P[indx]
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_aug_raw = np.concatenate((P,pl), axis=1)
    return(np.float16(p_aug_raw))

def generate_curve(data, sigma = 0.1, knot = 4):
    # independent variables in increasing order. 
    xx = (np.ones((data.shape[1],1))*(np.arange(0, data.shape[0], (data.shape[0]-1)/(knot+1)))).transpose() #(6,3)
    #print('xx.shape=',np.shape(xx))
    # dependent variables
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, data.shape[1])) #(6,3)
    #print('yy.shape=',np.shape(yy))
    x_range = np.arange(data.shape[0])
    cs_i = []
    for dim in range(data.shape[1]):
        cs = CubicSpline(xx[:,dim], yy[:,dim])
        #print('dim=',dim,'cs.shape=',np.shape(cs))
        cs_i.append(cs(x_range))
        #print('dim=',dim,'csi.shape=',np.shape(cs_i))
    #print('(np.array(cs_i).transpose()).shape=',np.shape(np.array(cs_i).transpose()))
    return np.array(cs_i).transpose()

def magnitute_warp(data, sigma = 0.05):
    return data * generate_curve(data, sigma)


def MgWrp(ecg,cls,volum=273,mag=.1):
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                     # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X, sigma = mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)    

def MgWrpRaw(ecg,cls,volum=273,mag=.1):
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                     # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X, sigma = mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_aug_raw = np.concatenate((P,pl), axis=1)
    return(p_aug_raw)    

def generate_curve2(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def magnitute_warp2(X, sigma=0.2):    #DA_MagWarp
    X=np.array([X]).T
    return X * generate_curve2(X, sigma)


def MgWrp2(ecg,cls,volum=273,mag=.1):
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp2(X, sigma = mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)    

def time_warp(x, sigma=0.1, knot=4):
    #x=np.array([x]).T
    data=np.array(x)
    #print('X.shape=',np.shape(data))
    orig_steps = np.arange(x.shape[0])
    
    # independent variables
    warp_steps=(np.ones((data.shape[1],1))*(np.arange(0, data.shape[0], (data.shape[0]-1)/(knot+1)))).transpose()
    # dependent variables
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, data.shape[1]))
    
    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[:,dim])(orig_steps)
        scale = (x.shape[0]-1)/time_warp[-1]
        ret[:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[0]-1), x[:,dim]).T
    return ret

def TimWrp(ecg,cls,volum=273,mag=.1):
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(time_warp(X,sigma=mag)[:,0]).T
        new=np.append(new,Pn)
        
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)  

def TimWrpRaw(ecg,cls,volum=273,mag=.1):
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(time_warp(X,sigma=mag)[:,0]).T
        new=np.append(new,Pn) 
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_aug_raw = np.concatenate((P,pl), axis=1)
    return(p_aug_raw) 


def time_warp2(X, sigma=0.1):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

def TimWrp2(ecg,cls,volum=273,mag=.1):
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_hght(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(time_warp2(X,sigma=mag)[:,0]).T
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)

def GDO(ecg,cls,volum=273,mag=.1):     #Gaussian Distribution Oversampling
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_f(P))                    # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.random.normal(P[indx], mag*var[indx])
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    p_wn=np.array(wndwng_sig(P))
    pl=np.ones(len(p_wn))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_wn = np.concatenate((p_wn,pl), axis=1)
    return(p_wn)  

def GDORaw(ecg,cls,volum=273,mag=.1):     #Gaussian Distribution Oversampling
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    #P=np.array(vars()['ecg'+str(i)][:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    #print('for Class ',i,'shape positive is = ',np.shape(P))
    for j in range (N_smpls):
        if train_labels[j]!=i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    #print('for Class ',i,'shape Negative is = ',np.shape(N))
    #trlt=np.array(trl).reshape(-1,)
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    new=np.array([])
    var=np.array(sigma_f(P))                  # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    #print(var)
    imprtnc=valuation(P,tr,trlt)
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    #print('class required augmenting =', frqncy)
    #print('first class samples      =', len(P))
    for z in range (frqncy):                       #########تولید نمونه های جدید#########
        '''indx=z%LenP                                         #انتخاب نمونه
        pplt=pplt+1
        #print('indx', indx)
        old_new=np.append(old_new,P[indx])
        old_new_lbl=np.append(old_new_lbl,PL)
        #avrgpeak=np.mean(peak_hght(P[indx]))
        #X=P[indx:indx+3]    #Time Warping
        #X=np.reshape(X,(np.shape(X)[1],np.shape(X)[0]))
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.array(magnitute_warp(X)[:,0]).T
        #print('Pn.Shape=',np.shape(Pn))
        #Pn=np.random.normal(P[indx], alpha*var[indx])      
        new=np.append(new,Pn)'''        
        indx=choice(P,imprtnc)                     #انتخاب نمونه
        X=np.array(P[indx:indx+3,:]).T
        Pn=np.random.normal(P[indx], mag*var[indx])
        new=np.append(new,Pn)
    new=new.reshape(frqncy,np.shape(P)[1])
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_aug_raw = np.concatenate((P,pl), axis=1)
    return(p_aug_raw) 


def sigma(P):                                #محاسبه ی واریانس هر نمونه ی اقلیت (مثبت) جهت تکثیر
    var=[]
    for i in range (len(P)):
        dist_arrayP = np.square(P-P[i])#.reshape(-1,1) #distance of every training instance
        dist_arrayP=np.sqrt(dist_arrayP)
        g=0
        for g in range (len(P)):
            if np.sum(dist_arrayP[g])==0:
                dist_arrayP[g]=dist_arrayP[g]+1000000
                break
        g=0
        onenn=int(np.argmin(np.sum(dist_arrayP,axis=1)))
        vari = np.sqrt(np.square(P[i] - P[onenn]))
        var.append(vari)
    return(var)

def sigma_f(P): #محاسبه ی واریانس هر نمونه اقلیت(مثبت)جهت تکثیر.(مقایسه مجزای هر ويژگی با ویژگی های باقی نمونه ها) 
    var=[]
    i=0
    h=np.shape(P)[0]
    try:                                       #رفع خطای اندیس برای آرایه 1 بعدی
        w =  np.shape(P)[1]
    except IndexError:
        w = 1
    for i in range (h):
        j=0
        dist_arrayP = np.square(P-P[i])           #distance of every training instance
        dist_arrayP=np.sqrt(dist_arrayP)
        #print('dist_arrayP=\n',dist_arrayP)
        #print(dist_arrayP.shape())
        onenn=np.zeros(np.shape(P)[1])
        for j in range (w):
            g=0
            for g in range (h):
                #print('g&j=',g,j)
                if dist_arrayP[g,j]==0:
                    dist_arrayP[g,j]=100000
            g=0
            #minj=np.argmin(dist_arrayP(:,j))
            onenn[j]=min(dist_arrayP[:,j])
        var.append(onenn)
    return(var)

def sigma_hght(P):                                #محاسبه ی واریانس هر نمونه ی اقلیت (مثبت) جهت تکثیر
    var=[]
    for i in range (len(P)):
        dist_arrayP = distance_from_all_training_hght(P,P[i]).reshape(-1,1) #distance of every training instance
        g=0
        for g in range (len(P)):
            if dist_arrayP[g]==0:
                dist_arrayP[g]=100
                break
        g=0
        onenn=np.argmin(dist_arrayP)
        onenn=int(onenn)
        vari = dist_arrayP[onenn]
        var.append(vari)
    var=np.array(var)
    return(var)

def furir(PP):
    ff=np.fft.fft(PP[0])
    fflen=int(len(ff))
    for i in range (1,len(PP)):
        fff=np.fft.fft(PP[i])
        ff=np.append(ff,fff,axis=0)
    ff=np.reshape(ff,(int(len(ff)/fflen),fflen))
    return(ff)

def frqnc(furir_windws,cls,volum=273,mag=1):     #!!!!! input is ===> Furier Signals    !!!!!!
    ecg=np.array(furir_windws)
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,mag+1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    for j in range (N_smpls):
        if train_labels[j]==i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    if np.ndim(N)==3 :
        N=np.reshape(N,(np.shape(N)[1],np.shape(N)[2]))
    imprtnc=valuation(P,tr,trlt)
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    ###################################### Real+Imaginary Part #######################################
    realP=P.real
    varReal=np.array(sigma_f(realP))                     # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    ImgP=P.imag
    varImg=np.array(sigma_f(ImgP))                        # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    t1=time.time()
    for z in range (frqncy):                     #########تولید نمونه های جدید#########
        indx=choice(P,imprtnc)                      #انتخاب نمونه
        old=np.array(P[indx])        
        old.real=np.random.normal(old.real, mag*varReal[indx]) #old.real #تولید نمونه
        old.imag=np.random.normal(ImgP[indx], mag*varImg[indx])    #تولید نمونه  
        new=np.append(new,old)
    new=new.reshape(frqncy,int(len(new)/frqncy))
    new=np.fft.ifft(new)
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    p_wn = np.int16(np.concatenate((P,pl), axis=1))
    print('frq=',frqncy,'augshape=',np.shape(P))
    print("first Augmented =")
    fullprint(p_wn[0])
    return(np.float16(p_wn))                            #Generated signals and labels
    
def frqncRaw(furir_sig,cls,volum=273,mag=1):     #!!!!! input is ===> Furier Signals    !!!!!!
    ecg=np.array(furir_sig)
    volum=int(volum+3)
    i=int(cls)
    c,alpha,PL=0,mag+1,1
    P=np.array([])                                 #آرایه ی کلاس اقلیت
    N=np.array([])                                 #آرایه ی کلاس اکثریت    
    N_smpls=len(ecg)                               #تعداد کل نمونه های داده ها
    train_labels=np.array(ecg[:,-1])
    trl=np.array(train_labels)
    tr=np.array(ecg[:,:-1])
    P=np.array(ecg[np.where(ecg[:,-1]==cls),:-1])
    N=np.array(ecg[np.where(ecg[:,-1]!=cls),:-1])
    for j in range (N_smpls):
        if train_labels[j]==i :                    # !!!! ==
            trl[j]=1
        else :
            trl[j]=0
    trlt=np.array([trl]).T
    if np.ndim(P)==3 :
        P=np.reshape(P,(np.shape(P)[1],np.shape(P)[2]))
    if np.ndim(N)==3 :
        N=np.reshape(N,(np.shape(N)[1],np.shape(N)[2]))
    imprtnc=valuation(P,tr,trlt)
    new=np.array([])
    frqncy= int(volum)                             #تعداد تولید نمونه ی جدید
    ###################################### Real+Imaginary Part #######################################
    realP=P.real
    varReal=np.array(sigma_f(realP))                     # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    ImgP=P.imag
    varImg=np.array(sigma_f(ImgP))                        # واریانس ویژگی ها برای تولید هر نمونه ی اقلیت
    lnp=len(P)
    wdthp=len(P[0])
    for z in range (frqncy):                     #########تولید نمونه های جدید#########
        indx=choice(P,imprtnc)                      #انتخاب نمونه
        old=np.array(P[indx])        
        old.real=np.random.normal(old.real, mag*varReal[indx]) #old.real #تولید نمونه
        old.imag=np.random.normal(old.imag, mag*varImg[indx])          #تولید نمونه ImgP[indx] 
        new=np.append(new,old)
    new=new.reshape(frqncy,int(len(new)/frqncy))
    new=np.fft.ifft(new)
    P = np.array(new)                                      #تکمیل داده های آموزشی جدید
    pl=np.ones(len(P))
    pl=pl*cls
    pl=np.int16(np.array([pl]).T)
    #print('shape pl =', np.shape(pl))
    p_aug_raw = np.concatenate((P,pl), axis=1)
    return(np.float16(p_aug_raw))


def dstrbtn_indx2(ttl_ln,nd_ln,dstr):#(data,total_data_length,needed_data_length,distribution)
    ttl_ln=int(ttl_ln)
    nd_ln=int(nd_ln)
    ttl_ln=ttl_ln-2
    indxs=[]
    indxs_ttl=np.arange(0,ttl_ln)
    if dstr==1 or nd_ln>.7*ttl_ln:      #sequencial=select Respectively from first of list
        indxs=np.arange(0,nd_ln)                    
    elif dstr==2 :                        #select linear 
        while len(indxs)<(nd_ln):       #triangular(left, mode, right, size=None)
            nw_indx=min(int(np.random.triangular(0,0,ttl_ln,1)),ttl_ln)
            indxs.append(indxs_ttl[nw_indx])
            indxs_ttl=np.delete(indxs_ttl[nw_indx])
            ttl_ln=ttl_ln-1
            #nw_indx=int(max(0,nw_indx))
    elif dstr==3 :
        while len(indxs)<(nd_ln):         #beta(a, b, size=None)
            nw_indx=min((ttl_ln-1)-int(np.random.beta(1, .7)*ttl_ln),ttl_ln)
            indxs.append(indxs_ttl[nw_indx])
            indxs_ttl=np.delete(indxs_ttl[nw_indx])
            ttl_ln=ttl_ln-1
    elif dstr==4 :
        while len(indxs)<(nd_ln):         #exponential(scale=1.0, size=None)
            nw_indx=min(int(np.random.exponential(.06)*ttl_ln),ttl_ln)
            indxs.append(indxs_ttl[nw_indx])
            indxs_ttl=np.delete(indxs_ttl[nw_indx])
            ttl_ln=ttl_ln-1
    elif dstr==5 :
        indxs=np.random.randint(0,ttl_ln,nd_ln)
    '''if dstr==6 :                       #error: data in range second half was not selected
        while len(indxs)<(nd_ln):         #gamma(shape, scale=1.0, size=None)
            nw_indx=min(int(np.random.gamma(1,.05)*ttl_ln),ttl_ln)
            #nw_indx=int(max(0,nw_indx))
            indxs.append(nw_indx) if nw_indx not in indxs else indxs.append(nw_indx+1) if (nw_indx+1) not in indxs else indxs.append(nw_indx-1) if ((nw_indx-1) not in indxs and (nw_indx-1)>-1) else indxs'''               
    return(np.int16(indxs))



def dstrbtn_indx(ttl_ln,nd_ln,dstr):#(data,total_data_length,needed_data_length,distribution)
    ttl_ln=int(ttl_ln)
    nd_ln=int(nd_ln)
    ttl_ln=ttl_ln-2
    indxs=[]
    if dstr==1 or nd_ln>.7*ttl_ln :     #sequencial=select Respectively from first of list
        nd_ln=min(nd_ln,ttl_ln)
        indxs=np.arange(0,nd_ln)
    elif dstr==2 :                        #select linear 
        while len(indxs)<(nd_ln):       #triangular(left, mode, right, size=None)
            nw_indx=min(int(np.random.triangular(0,0,ttl_ln,1)),ttl_ln)
            #nw_indx=int(max(0,nw_indx))
            indxs.append(nw_indx) if nw_indx not in indxs else indxs.append(nw_indx+1) if (nw_indx+1) not in indxs else indxs.append(nw_indx-1) if ((nw_indx-1) not in indxs and (nw_indx-1)>-1) else indxs 
    
    elif dstr==3 :
        while len(indxs)<(nd_ln):         #beta(a, b, size=None)
            nw_indx=min((ttl_ln-1)-int(np.random.beta(1, .7)*ttl_ln),ttl_ln)
            #nw_indx=int(max(0,nw_indx))
            indxs.append(nw_indx) if nw_indx not in indxs else indxs.append(nw_indx+1) if (nw_indx+1) not in indxs else indxs.append(nw_indx-1) if ((nw_indx-1) not in indxs and (nw_indx-1)>-1) else indxs  
               
    elif dstr==4 :
        while len(indxs)<(nd_ln):         #exponential(scale=1.0, size=None)
            nw_indx=min(int(np.random.exponential(.06)*ttl_ln),ttl_ln)
            #nw_indx=int(max(0,nw_indx))
            indxs.append(nw_indx) if nw_indx not in indxs else indxs.append(nw_indx+1) if (nw_indx+1) not in indxs else indxs.append(nw_indx-1) if ((nw_indx-1) not in indxs and (nw_indx-1)>-1) else indxs  
    
    elif dstr==5 :
        indxs=np.random.randint(0,ttl_ln,nd_ln)
    '''if dstr==6 :                       #error: data in range second half was not selected
        while len(indxs)<(nd_ln):         #gamma(shape, scale=1.0, size=None)
            nw_indx=min(int(np.random.gamma(1,.05)*ttl_ln),ttl_ln)
            #nw_indx=int(max(0,nw_indx))
            indxs.append(nw_indx) if nw_indx not in indxs else indxs.append(nw_indx+1) if (nw_indx+1) not in indxs else indxs.append(nw_indx-1) if ((nw_indx-1) not in indxs and (nw_indx-1)>-1) else indxs'''               
    return(np.int16(indxs))
    
def srtd_data(mthd,i,nmbr,dstr,srt,invrs=0):           #import data from gan generated data
    import glob
    #from glob import glob
    nmbr=int(nmbr)
    cwd = os.getcwd() #
    adrs=('{}/AugEvl/AugEvl{}{}.csv'.format(cwd,i,mthd))   #'_3\*.csv' select base on quality 3,4,5
    alldata = []                                                      # List for storing all the data
    #print('alldata shape = ', np.shape(alldata))
    data = pd.read_csv(adrs)                                     # Load mat file data into data.
    data=np.array(data)
    if invrs==0 :
        dt=data[data[:,srt].argsort()]                                # sort bae of quality $Append data to the list
    else:
        dt=data[(-data[:,srt]).argsort()]                                # sort bae of quality $Append data to the list
    '''print('data first row 2 last column = ', dt[0,-2:])
    print('data end row 2 last column = ', dt[-1,-2:])
    print('min dis = ', np.min(dt[:,-2]))
    print('min var = ', np.min(dt[:,-1]))
    print('csv shape = ', np.shape(dt))'''
    indxes=dstrbtn_indx(len(dt),nmbr,dstr)               #Distribution base selection
    dt=list(dt[indxes])
    alldata=list(alldata)
    alldata.append(dt)
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))
    return alldata[:nmbr,1:-5]    #not have 3 item in the end mthd=22,clstr,epoch

def srtd_dataH(mthd,i,nmbr,dstr,srt,invrs=0):           #import data from gan generated data
    import glob
    #from glob import glob
    nmbr=int(nmbr)
    cwd = os.getcwd() #
    adrs=('{}/AugEvlH/AugEvl{}{}.csv'.format(cwd,i,mthd))   #'_3\*.csv' select base on quality 3,4,5
    alldata = []                                                      # List for storing all the data
    #print('alldata shape = ', np.shape(alldata))
    data = pd.read_csv(adrs)                                     # Load mat file data into data.
    data=np.array(data)
    if invrs==0 :
        dt=data[data[:,srt].argsort()]                                # sort bae of quality $Append data to the list
    else:
        dt=data[(-data[:,srt]).argsort()]                                # sort bae of quality $Append data to the list
    '''print('data first row 2 last column = ', dt[0,-2:])
    print('data end row 2 last column = ', dt[-1,-2:])
    print('min dis = ', np.min(dt[:,-2]))
    print('min var = ', np.min(dt[:,-1]))
    print('csv shape = ', np.shape(dt))'''
    indxes=dstrbtn_indx(len(dt),nmbr,dstr)               #Distribution base selection
    dt=list(dt[indxes])
    alldata=list(alldata)
    alldata.append(dt)
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))
    return alldata[:nmbr,1:-5]    #not have 3 item in the end mthd=22,clstr,epoch

def srtd_dataV(mthd,i,nmbr,dstr,srt,invrs=0):           #import data from gan generated data
    import glob
    #from glob import glob
    nmbr=int(nmbr)
    cwd = os.getcwd() #
    adrs=('{}/AugEvlV/AugEvl{}{}.csv'.format(cwd,i,mthd))   #'_3\*.csv' select base on quality 3,4,5
    alldata = []                                                      # List for storing all the data
    #print('alldata shape = ', np.shape(alldata))
    data = pd.read_csv(adrs)                                     # Load mat file data into data.
    data=np.array(data)
    if invrs==0 :
        dt=data[data[:,srt].argsort()]                                # sort bae of quality $Append data to the list
    else:
        dt=data[(-data[:,srt]).argsort()]                                # sort bae of quality $Append data to the list
    '''print('data first row 2 last column = ', dt[0,-2:])
    print('data end row 2 last column = ', dt[-1,-2:])
    print('min dis = ', np.min(dt[:,-2]))
    print('min var = ', np.min(dt[:,-1]))
    print('csv shape = ', np.shape(dt))'''
    indxes=dstrbtn_indx(len(dt),nmbr,dstr)               #Distribution base selection
    dt=list(dt[indxes])
    alldata=list(alldata)
    alldata.append(dt)
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))
    return alldata[:nmbr,1:-5]    #not have 3 item in the end mthd=22,clstr,epoch

def srtd_data_raw(mthd,i,nmbr,dstr,srt,invrs=0):           #import data from gan generated data
    import glob
    #from glob import glob
    cwd = os.getcwd() #
    adrs=('{}/AugEvl/AugEvl{}{}.csv'.format(cwd,i,mthd))   #'_3\*.csv' select base on quality 3,4,5
    alldata = []                                                      # List for storing all the data
    #print('alldata shape = ', np.shape(alldata))
    data = pd.read_csv(adrs)                                     # Load mat file data into data.
    data=np.array(data)
    if invrs==0 :
        dt=data[data[:,srt].argsort()]                                # sort bae of quality $Append data to the list
    else:
        dt=data[(-data[:,srt]).argsort()]                                # sort bae of quality $Append data to the list
    '''print('data first row 2 last column = ', dt[0,-2:])
    print('data end row 2 last column = ', dt[-1,-2:])
    print('min dis = ', np.min(dt[:,-2]))
    print('min var = ', np.min(dt[:,-1]))
    print('csv shape = ', np.shape(dt))'''
    indxes=dstrbtn_indx(len(dt),nmbr,dstr)               #Distribution base selection
    dt=list(dt[indxes])
    alldata=list(alldata)
    alldata.append(dt)
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))
    return alldata[:nmbr,1:-2]    #not have 6 item in the end (mthd=22,clstr,epoch,dis,varDist,qlty3)


def gan_data(i,nmbr,dstr,srt):                          #import data from gan generated data
    import glob
    #from glob import glob
    cwd = os.getcwd() #
    adrs=cwd+'\GAN_Cls'+str(i)+'\*.csv'                 # '_3\*.csv' select base on quality 3,4,5
    files = glob.glob(adrs)
    alldata = []                                                      # List for storing all the data
    clstrs=len(files)
    nmbr_clstr=int(nmbr/clstrs)
    for fname in files:                                               # Iterate mat files
        #print('alldata shape = ', np.shape(alldata))
        data = pd.read_csv(fname)                                     # Load mat file data into data.
        data=np.array(data)
        dt=data[data[:,srt].argsort()]                                # Append data to the list
        '''print('data first row 2 last column = ', dt[0,-2:])
        print('data end row 2 last column = ', dt[-1,-2:])
        print('min dis = ', np.min(dt[:,-2]))
        print('min var = ', np.min(dt[:,-1]))
        print('csv shape = ', np.shape(dt))'''
        indxes=dstrbtn_indx(len(dt),nmbr_clstr,dstr)               #Distribution base selection
        dt=list(dt[indxes])
        alldata=list(alldata)
        alldata.append(dt)
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))
    return alldata[:nmbr,1:-8] 
#9 end colmn: 1-Class,2-Cluster,3-snsvty,4-epoch,5-distance,6-variance,7-dis_major,8-GDO_Qualty,9-Invrs_GDO_Qualty


def lstm_data(i,nmbr,dstr,srt=-5):                             #import data from gan generated data
    cwd = os.getcwd() #
    adrs=cwd+'\LSTM_Cls'+str(i)+'\*.csv'
    files = glob.glob(adrs)
    pnts=len(np.array(pd.read_csv(files[0]))[0])
    alldata = np.empty([0,pnts])                                # List for storing all the data
    clstrs=len(files)
    ii=0
    nmbr_clstr=int(nmbr/clstrs)
    for fname in files:                                         # Iterate mat files
        data = pd.read_csv(fname)                               # Load mat file data into data.
        data=np.array(data)
        dt=data[data[:,srt].argsort()]                          # Append data to the list
        if nmbr_clstr>(len(dt)*.8):
            alldata=np.append(alldata,dt,axis=0)
        else:
            indxes=dstrbtn_indx(len(dt),nmbr_clstr,dstr)        #Distribution base selection
            dt=list(dt[indxes])
            alldata=np.append(alldata,dt,axis=0)
    return alldata[:,1:-8] 
#9 end colmn: 1-Class,2-Cluster,3-snsvty,4-epoch,5-distance,6-variance,7-dis_major,8-GDO_Qualty,9-Invrs_GDO_Qualty


def lstm_data0(i,nmbr,dstr,srt=-5):                             #import data from gan generated data
    import glob
    #from glob import glob
    cwd = os.getcwd() #
    adrs=cwd+'\LSTM_Cls'+str(i)+'\*.csv'
    files = glob.glob(adrs)
    alldata = []                                                      # List for storing all the data
    clstrs=len(files)
    #ii=0
    nmbr_clstr=int(nmbr/clstrs)
    for fname in files:                                               # Iterate mat files
        data = pd.read_csv(fname)                                     # Load mat file data into data.
        data=np.array(data)
        dt=data[data[:,srt].argsort()]                                 # Append data to the list
        '''print('data first row 2 last column = ', dt[0,-2:])
        print('data end row 2 last column = ', dt[-1,-2:])
        print('min dis = ', np.min(dt[:,-2]))
        print('min var = ', np.min(dt[:,-1]))
        print('csv shape = ', np.shape(dt),'need0 =', nmbr_clstr)'''
        if nmbr_clstr>(len(dt)*.8):
            #nmbr_clstr=len(dt)-2
            alldata.append(dt)
            #print('csv shape = ', np.shape(dt),'need second =', nmbr_clstr)
        #dt=dt[:int(nmbr_clstr+1)]
        #print('step ', ii, ' dt shape = ', np.shape(dt))
        else:
            indxes=dstrbtn_indx(len(dt),nmbr_clstr,dstr)           #Distribution base selection
            dt=list(dt[indxes])
            alldata.append(dt)
            #print('step ', ii, ' alldata shape = ', np.shape(alldata))
        #ii=ii+1
    print('last alldata shape =',np.shape(alldata))
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    else:
        alldata=np.array(alldata)
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))
    return alldata[:,1:-5] 
#9 end colmn: 1-Class,2-Cluster,3-snsvty,4-epoch,5-distance,6-variance,7-dis_major,8-GDO_Qualty,9-Invrs_GDO_Qualty

    
def lstm_data1(i,nmbr,dstr,srt=-5):                             #import data from gan generated data
    import glob
    #from glob import glob
    cwd = os.getcwd() #
    adrs=cwd+'\LSTM_Cls'+str(i)+'_3\*.csv'
    files = glob.glob(adrs)
    pnts=len(np.array(pd.read_csv(files[0]))[0])
    alldata = np.empty([0,pnts])                                # List for storing all the data
    clstrs=len(files)
    ii=0
    nmbr_clstr=int(nmbr/clstrs)
    for fname in files:                                         # Iterate mat files
        data = pd.read_csv(fname)                               # Load mat file data into data.
        data=np.array(data)
        #pnts=int(np.shape(data)[1])
        dt=data[data[:,srt].argsort()]                          # Append data to the list
        '''print('data first row 2 last column = ', dt[0,-2:])
        print('data end row 2 last column = ', dt[-1,-2:])
        print('min dis = ', np.min(dt[:,-2]))
        print('min var = ', np.min(dt[:,-1]))
        print('csv shape = ', np.shape(dt),'need0 =', nmbr_clstr)
        print('nmbr_clstr>(len(dt)*.8)=',nmbr_clstr,len(dt)*.8)'''
        if nmbr_clstr>(len(dt)*.8):
            alldata=np.append(alldata,dt,axis=0)
            #dt=list(dt)
            #print('1_np.shape(dt)=',np.shape(dt))
            #alldata.append(dt)
            #print('1_np.shape(alldata)=',np.shape(alldata))
            #nmbr_clstr=len(dt)-2
            #print('csv shape = ', np.shape(dt),'need second =', nmbr_clstr)
        #dt=dt[:int(nmbr_clstr+1)]
        #print('step ', ii, ' dt shape = ', np.shape(dt))
        else:
            indxes=dstrbtn_indx(len(dt),nmbr_clstr,dstr)        #Distribution base selection
            dt=list(dt[indxes])
            alldata=np.append(alldata,dt,axis=0)
            #print('2_np.shape(dt)=',np.shape(dt))
            #alldata.append(dt)
            #print('2_np.shape(alldata)=',np.shape(alldata))
        #print('step ', ii, ' alldata shape = ', np.shape(alldata))
        #ii=ii+1
    #print('last alldata shape =',np.shape(alldata))
    '''if np.ndim(alldata)==1 :
        alldata=np.reshape(alldata,(int(len(alldata)/pnts),pnts))
    if np.ndim(alldata)==3 :
        alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    else:
        alldata=np.array(alldata)
    alldata=np.array(alldata)
    #print('final shape= ' , np.shape(alldata))'''
    return alldata[:,3:-8] 
#9 end colmn: 1-Class,2-Cluster,3-snsvty,4-epoch,5-distance,6-variance,7-dis_major,8-GDO_Qualty,9-Invrs_GDO_Qualty

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
    
    
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def ECG_model(config):
    """ 
    implementation of the model in https://www.nature.com/articles/s41591-018-0268-3 
    also have reference to codes at 
    https://github.com/awni/ecg/blob/master/ecg/network.py 
    and 
    https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/train_model.py
    """
    def first_conv_block(inputs, config):
        layer = Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        shortcut = MaxPooling1D(pool_size=1,
                      strides=1)(layer)

        layer =  Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(config.drop_rate)(layer)
        layer =  Conv1D(filters=config.filter_length,
                        kernel_size=config.kernel_size,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        return add([shortcut, layer])

    def main_loop_blocks(layer, config):
        filter_length = config.filter_length
        n_blocks = 15
        for block_index in range(n_blocks):
            def zeropad(x):
                """ 
                zeropad and zeropad_output_shapes are from 
                https://github.com/awni/ecg/blob/master/ecg/network.py
                """
                y = K.zeros_like(x)
                return K.concatenate([x, y], axis=2)

            def zeropad_output_shape(input_shape):
                shape = list(input_shape)
                assert len(shape) == 3
                shape[2] *= 2
                return tuple(shape)

            subsample_length = 2 if block_index % 2 == 0 else 1
            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            # 5 is chosen instead of 4 from the original model
            if block_index % 4 == 0 and block_index > 0 :
                # double size of the network and match the shapes of both branches
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides=subsample_length,
                            kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(config.drop_rate)(layer)
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides= 1,
                            kernel_initializer='he_normal')(layer)
            layer = add([shortcut, layer])
        return layer

    def output_block(layer, config):
        from keras.layers.wrappers import TimeDistributed
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        #layer = Flatten()(layer)
        outputs = TimeDistributed(Dense(len_classes, activation='softmax'))(layer)
        model = Model(inputs=inputs, outputs=outputs)
        
        adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer= adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.summary()
        return model

    classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S'] are too few or not in the trainset, so excluded out
    len_classes = len(classes)

    inputs = Input(shape=(config.input_size, 1), name='input')
    layer = first_conv_block(inputs, config)
    layer = main_loop_blocks(layer, config)
    return output_block(layer, config)