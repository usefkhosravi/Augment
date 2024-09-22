#import pandas as pd
import os
from scipy import signal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Conv1D,Flatten,MaxPooling1D,Bidirectional,LSTM,Dropout,TimeDistributed,MaxPool2D
from keras.layers import Dense,GlobalAveragePooling2D
import matplotlib.pyplot as plt
import tensorflow as tf

def LSTM_Gnrt_Pridc(wndws,c,AgTyp):
    # در اینجا سیگنال تولید شده مشابه سیگنالی از اتصال سیگنال ها (حداکثر 60تا) در کلاس است
    #این سیگنال تولید شده در محل اتصال سیگنال ها (نمونه ها) دارای شکست خواهد بود
    # از سیگنال تولید شده تپش های قلب استخراج خواهد شد
    #و فاصله تا نزدیک ترین همسایه(موجود در تپش های داده ی اصلی)و واریانس بردار فاصله به انتهای تپش افزوده می شود##
    cwd = os.getcwd()
    print('bincount=',np.bincount(np.int16(wndws[:,-1])))
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    smpl_rte=len(wndws[0])-1  # decrease class lable -1
    points= int(smpl_rte)
    wdth=len(wndws[0])
    rte_strch=2                  # increase sample rate to 1x or 2x or ...
    window_size = int(rte_strch*smpl_rte/8)   #windows point in         50 or 100 or ...
    print('LSTM Window size=',window_size)
    aug_quantity=1200
    cls=int(c)
    print('CNN_LSTM Generation for Class = ', c)
    ecg_cls=np.array(wndws[np.where(wndws[:,-1]==c),:-1])             # حذف برچسب کلاس + جداسازی کلاس
    ecg_cls = signal.resample(ecg_cls,smpl_rte*rte_strch,axis=1)
    len_cls=len(ecg_cls)
    print('len_cls ', cls,'=',len_cls)
    epch=int(10*((mx_cls_smpl/len_cls)))                      
    print('epochs=',epch)
    epch=1                                       ####### !!!!!!!!!!    Delete   !!!!!!!!!!
    ecg_cls_1d=[]
    for s in range (len(ecg_cls)):
        ecg_cls_1d=np.append(ecg_cls_1d,ecg_cls[s])               #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'ecg_cls_1d.shape= ',np.shape(ecg_cls_1d))
    #آموزش شبکه با کل نمونه های کلاس 
    X = []
    Y = []
    sgnl= np.array(ecg_cls_1d)
    #print('sgnl shape = ', np.shape(sgnl))
    for i in range(0,len(sgnl) - window_size -rte_strch): #استخراج پنجره ونرمال سازی حسب نقطه اولش
        first = sgnl[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((sgnl[i + j] - first) / first)
        temp2.append((sgnl[i +window_size] - first) / first)
        X.append(np.array(temp).reshape(window_size, 1))
        Y.append(np.array(temp2).reshape(1,1))
    #print('shape X = ', np.shape(X))
    #print('shape Y = ', np.shape(Y))
    train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
    len_tr = len(train_X)
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_X = train_X.reshape(train_X.shape[0],1,window_size,1)
    test_X = test_X.reshape(test_X.shape[0],1,window_size,1)
    model = lstm_cnn(window_size)
    model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=epch,batch_size=64, shuffle = False)

    bs_aug_pls=pd.read_csv(cwd+'\AugEvl\AugEvl'+str(cls)+str(AgTyp)+'0'+'.csv')#داده افزایی قبلی برای داده افزایی مجدد   
    os.chdir(cwd)    
    bs_aug_pls=np.array(bs_aug_pls)
    bs_aug_pls=np.array(bs_aug_pls[:,1:-6])
    print('bs_aug_pls shape=',np.shape(bs_aug_pls))
    print('bs_aug_pls[0]=',bs_aug_pls[0])
    bs_aug_pls = signal.resample(ecg_cls,smpl_rte*rte_strch,axis=1)
    bs_aug_1d=[]
    for s in range (len(bs_aug_pls)):
        bs_aug_1d=np.append(bs_aug_1d,bs_aug_pls[s])               #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'bs_aug_1d.shape= ',np.shape(bs_aug_1d))
    bs_aug_1d= np.array(bs_aug_1d)           #سیگنال حاوی همه ی نمونه ها که اشکالش در محل اتصال دو نمونه است 
    print('bs_aug_1d shape = ', np.shape(bs_aug_1d))           #و در نتیجه نمونه ی دارای شکست تولید می کند
    i,j,z=0,0,0
    bs_aug_1d_Nrm=[]
    #Y=[]
    for i in range(0,len(bs_aug_1d)-window_size): #سیگنال نرمال شده حاوی همه نمونه ها(حسب نقطه اول پنجره)
        first = bs_aug_1d[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((bs_aug_1d[i + j] - first) / first)
        bs_aug_1d_Nrm.append(np.array(temp).reshape(window_size, 1))           #ردیف های نرمال شده
    print('bs_aug_1d_Nrm shape = ', np.shape(bs_aug_1d_Nrm))
    X_prd=np.reshape(bs_aug_1d_Nrm,(int(np.shape(bs_aug_1d_Nrm)[0]),1,int(np.shape(bs_aug_1d_Nrm)[1]),1))
    print('X_prd shape = ', np.shape(X_prd))     
    X_prdction=np.array(X_prd)            
    predicted  = model.predict(X_prdction)                # پیش بینی برچسب
    predicted = np.array(predicted[:,0]).reshape(-1)
    #بازگشت از فرم نورمال به واقعی برای انتهای سیگنال خام (محدوده ی تست و پیش بینی آن)
    pred_unrm=np.array(predicted)
    for z in range(len(predicted)):
        temp = bs_aug_1d[z]
        pred_unrm[z] = predicted[z] * temp + temp
    augment=[]
    augment.append(pred_unrm)
    augment=np.reshape(augment,(-1))
    #clear_output(wait=True)
    augment_fnl=[]
    augment_fnl=[]
    for k in range(int(len(augment)/smpl_rte)-1):
        augment_fnl.append(augment[3*window_size+k*smpl_rte:3*window_size+(k+1)*smpl_rte])
    cls_clm=np.zeros(len(augment_fnl))
    cls_clm=np.int16(cls_clm+cls)
    cls_clm=np.array([cls_clm]).T
    augment_fnl=np.concatenate((augment_fnl,cls_clm),axis=1)
    print('augmented shape for class ',c , np.shape(augment_fnl))
    return(augment_fnl)

def LSTM_Gnrt_PridcV(wndws,c,AgTyp):
    # در اینجا سیگنال تولید شده مشابه سیگنالی از اتصال سیگنال ها (حداکثر 60تا) در کلاس است
    #این سیگنال تولید شده در محل اتصال سیگنال ها (نمونه ها) دارای شکست خواهد بود
    # از سیگنال تولید شده تپش های قلب استخراج خواهد شد
    #و فاصله تا نزدیک ترین همسایه(موجود در تپش های داده ی اصلی)و واریانس بردار فاصله به انتهای تپش افزوده می شود##
    cwd = os.getcwd()
    print('bincount=',np.bincount(np.int16(wndws[:,-1])))
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    smpl_rte=len(wndws[0])-1  # decrease class lable -1
    points= int(smpl_rte)
    wdth=len(wndws[0])
    rte_strch=2                  # increase sample rate to 1x or 2x or ...
    window_size = int(rte_strch*smpl_rte/8)   #windows point in         50 or 100 or ...
    print('LSTM Window size=',window_size)
    aug_quantity=1200
    cls=int(c)
    print('CNN_LSTM Generation for Class = ', c)
    ecg_cls=np.array(wndws[np.where(wndws[:,-1]==c),:-1])             # حذف برچسب کلاس + جداسازی کلاس
    ecg_cls = signal.resample(ecg_cls,smpl_rte*rte_strch,axis=1)
    len_cls=len(ecg_cls)
    print('len_cls ', cls,'=',len_cls)
    epch=int(10*((mx_cls_smpl/len_cls)))                      
    print('epochs=',epch)
    epch=1                                       ####### !!!!!!!!!!    Delete   !!!!!!!!!!
    ecg_cls_1d=[]
    for s in range (len(ecg_cls)):
        ecg_cls_1d=np.append(ecg_cls_1d,ecg_cls[s])               #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'ecg_cls_1d.shape= ',np.shape(ecg_cls_1d))
    #آموزش شبکه با کل نمونه های کلاس 
    X = []
    Y = []
    sgnl= np.array(ecg_cls_1d)
    #print('sgnl shape = ', np.shape(sgnl))
    for i in range(0,len(sgnl) - window_size -rte_strch): #استخراج پنجره ونرمال سازی حسب نقطه اولش
        first = sgnl[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((sgnl[i + j] - first) / first)
        temp2.append((sgnl[i +window_size] - first) / first)
        X.append(np.array(temp).reshape(window_size, 1))
        Y.append(np.array(temp2).reshape(1,1))
    #print('shape X = ', np.shape(X))
    #print('shape Y = ', np.shape(Y))
    train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
    len_tr = len(train_X)
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_X = train_X.reshape(train_X.shape[0],1,window_size,1)
    test_X = test_X.reshape(test_X.shape[0],1,window_size,1)
    model = lstm_cnn(window_size)
    model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=epch,batch_size=64, shuffle = False)

    bs_aug_pls=pd.read_csv(cwd+'\AugEvlV\AugEvl'+str(cls)+str(AgTyp)+'0'+'.csv')#داده افزایی قبلی برای داده افزایی مجدد   
    os.chdir(cwd)    
    bs_aug_pls=np.array(bs_aug_pls)
    bs_aug_pls=np.array(bs_aug_pls[:,1:-6])
    print('bs_aug_pls shape=',np.shape(bs_aug_pls))
    print('bs_aug_pls[0]=',bs_aug_pls[0])
    bs_aug_pls = signal.resample(ecg_cls,smpl_rte*rte_strch,axis=1)
    bs_aug_1d=[]
    for s in range (len(bs_aug_pls)):
        bs_aug_1d=np.append(bs_aug_1d,bs_aug_pls[s])               #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'bs_aug_1d.shape= ',np.shape(bs_aug_1d))
    bs_aug_1d= np.array(bs_aug_1d)           #سیگنال حاوی همه ی نمونه ها که اشکالش در محل اتصال دو نمونه است 
    print('bs_aug_1d shape = ', np.shape(bs_aug_1d))           #و در نتیجه نمونه ی دارای شکست تولید می کند
    i,j,z=0,0,0
    bs_aug_1d_Nrm=[]
    #Y=[]
    for i in range(0,len(bs_aug_1d)-window_size): #سیگنال نرمال شده حاوی همه نمونه ها(حسب نقطه اول پنجره)
        first = bs_aug_1d[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((bs_aug_1d[i + j] - first) / first)
        bs_aug_1d_Nrm.append(np.array(temp).reshape(window_size, 1))           #ردیف های نرمال شده
    print('bs_aug_1d_Nrm shape = ', np.shape(bs_aug_1d_Nrm))
    X_prd=np.reshape(bs_aug_1d_Nrm,(int(np.shape(bs_aug_1d_Nrm)[0]),1,int(np.shape(bs_aug_1d_Nrm)[1]),1))
    print('X_prd shape = ', np.shape(X_prd))     
    X_prdction=np.array(X_prd)            
    predicted  = model.predict(X_prdction)                # پیش بینی برچسب
    predicted = np.array(predicted[:,0]).reshape(-1)
    #بازگشت از فرم نورمال به واقعی برای انتهای سیگنال خام (محدوده ی تست و پیش بینی آن)
    pred_unrm=np.array(predicted)
    for z in range(len(predicted)):
        temp = bs_aug_1d[z]
        pred_unrm[z] = predicted[z] * temp + temp
    augment=[]
    augment.append(pred_unrm)
    augment=np.reshape(augment,(-1))
    #clear_output(wait=True)
    augment_fnl=[]
    augment_fnl=[]
    for k in range(int(len(augment)/smpl_rte)-1):
        augment_fnl.append(augment[3*window_size+k*smpl_rte:3*window_size+(k+1)*smpl_rte])
    cls_clm=np.zeros(len(augment_fnl))
    cls_clm=np.int16(cls_clm+cls)
    cls_clm=np.array([cls_clm]).T
    augment_fnl=np.concatenate((augment_fnl,cls_clm),axis=1)
    print('augmented shape for class ',c , np.shape(augment_fnl))
    return(augment_fnl)

def LSTM_Gnrt_PridcH(wndws,c,AgTyp):
    # در اینجا سیگنال تولید شده مشابه سیگنالی از اتصال سیگنال ها (حداکثر 60تا) در کلاس است
    #این سیگنال تولید شده در محل اتصال سیگنال ها (نمونه ها) دارای شکست خواهد بود
    # از سیگنال تولید شده تپش های قلب استخراج خواهد شد
    #و فاصله تا نزدیک ترین همسایه(موجود در تپش های داده ی اصلی)و واریانس بردار فاصله به انتهای تپش افزوده می شود##
    cwd = os.getcwd()
    print('bincount=',np.bincount(np.int16(wndws[:,-1])))
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    smpl_rte=len(wndws[0])-1  # decrease class lable -1
    points= int(smpl_rte)
    wdth=len(wndws[0])
    rte_strch=2                  # increase sample rate to 1x or 2x or ...
    window_size = int(rte_strch*smpl_rte/8)   #windows point in         50 or 100 or ...
    print('LSTM Window size=',window_size)
    aug_quantity=1200
    cls=int(c)
    print('CNN_LSTM Generation for Class = ', c)
    ecg_cls=np.array(wndws[np.where(wndws[:,-1]==c),:-1])             # حذف برچسب کلاس + جداسازی کلاس
    ecg_cls = signal.resample(ecg_cls,smpl_rte*rte_strch,axis=1)
    len_cls=len(ecg_cls)
    print('len_cls ', cls,'=',len_cls)
    epch=int(10*((mx_cls_smpl/len_cls)))                      
    print('epochs=',epch)
    epch=1                                       ####### !!!!!!!!!!    Delete   !!!!!!!!!!
    ecg_cls_1d=[]
    for s in range (len(ecg_cls)):
        ecg_cls_1d=np.append(ecg_cls_1d,ecg_cls[s])               #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'ecg_cls_1d.shape= ',np.shape(ecg_cls_1d))
    #آموزش شبکه با کل نمونه های کلاس 
    X = []
    Y = []
    sgnl= np.array(ecg_cls_1d)
    #print('sgnl shape = ', np.shape(sgnl))
    for i in range(0,len(sgnl) - window_size -rte_strch): #استخراج پنجره ونرمال سازی حسب نقطه اولش
        first = sgnl[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((sgnl[i + j] - first) / first)
        temp2.append((sgnl[i +window_size] - first) / first)
        X.append(np.array(temp).reshape(window_size, 1))
        Y.append(np.array(temp2).reshape(1,1))
    #print('shape X = ', np.shape(X))
    #print('shape Y = ', np.shape(Y))
    train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
    len_tr = len(train_X)
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_X = train_X.reshape(train_X.shape[0],1,window_size,1)
    test_X = test_X.reshape(test_X.shape[0],1,window_size,1)
    model = lstm_cnn(window_size)
    model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=epch,batch_size=64, shuffle = False)

    bs_aug_pls=pd.read_csv(cwd+'\AugEvlH\AugEvl'+str(cls)+str(AgTyp)+'0'+'.csv')#داده افزایی قبلی برای داده افزایی مجدد   
    os.chdir(cwd)    
    bs_aug_pls=np.array(bs_aug_pls)
    bs_aug_pls=np.array(bs_aug_pls[:,1:-6])
    print('bs_aug_pls shape=',np.shape(bs_aug_pls))
    print('bs_aug_pls[0]=',bs_aug_pls[0])
    bs_aug_pls = signal.resample(ecg_cls,smpl_rte*rte_strch,axis=1)
    bs_aug_1d=[]
    for s in range (len(bs_aug_pls)):
        bs_aug_1d=np.append(bs_aug_1d,bs_aug_pls[s])               #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'bs_aug_1d.shape= ',np.shape(bs_aug_1d))
    bs_aug_1d= np.array(bs_aug_1d)           #سیگنال حاوی همه ی نمونه ها که اشکالش در محل اتصال دو نمونه است 
    print('bs_aug_1d shape = ', np.shape(bs_aug_1d))           #و در نتیجه نمونه ی دارای شکست تولید می کند
    i,j,z=0,0,0
    bs_aug_1d_Nrm=[]
    #Y=[]
    for i in range(0,len(bs_aug_1d)-window_size): #سیگنال نرمال شده حاوی همه نمونه ها(حسب نقطه اول پنجره)
        first = bs_aug_1d[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((bs_aug_1d[i + j] - first) / first)
        bs_aug_1d_Nrm.append(np.array(temp).reshape(window_size, 1))           #ردیف های نرمال شده
    print('bs_aug_1d_Nrm shape = ', np.shape(bs_aug_1d_Nrm))
    X_prd=np.reshape(bs_aug_1d_Nrm,(int(np.shape(bs_aug_1d_Nrm)[0]),1,int(np.shape(bs_aug_1d_Nrm)[1]),1))
    print('X_prd shape = ', np.shape(X_prd))     
    X_prdction=np.array(X_prd)            
    predicted  = model.predict(X_prdction)                # پیش بینی برچسب
    predicted = np.array(predicted[:,0]).reshape(-1)
    #بازگشت از فرم نورمال به واقعی برای انتهای سیگنال خام (محدوده ی تست و پیش بینی آن)
    pred_unrm=np.array(predicted)
    for z in range(len(predicted)):
        temp = bs_aug_1d[z]
        pred_unrm[z] = predicted[z] * temp + temp
    augment=[]
    augment.append(pred_unrm)
    augment=np.reshape(augment,(-1))
    #clear_output(wait=True)
    augment_fnl=[]
    augment_fnl=[]
    for k in range(int(len(augment)/smpl_rte)-1):
        augment_fnl.append(augment[3*window_size+k*smpl_rte:3*window_size+(k+1)*smpl_rte])
    cls_clm=np.zeros(len(augment_fnl))
    cls_clm=np.int16(cls_clm+cls)
    cls_clm=np.array([cls_clm]).T
    augment_fnl=np.concatenate((augment_fnl,cls_clm),axis=1)
    print('augmented shape for class ',c , np.shape(augment_fnl))
    return(augment_fnl)

def LSTM_Gnrt_NtPridc(wndws,c,vlum_data):
    #file: lng_add_dis_var Code
    # در اینجا سیگنال تولید شده مشابه سیگنالی از اتصال سیگنال ها (حداکثر 60تا) در کلاس است
    #این سیگنال تولید شده در محل اتصال سیگنال ها (نمونه ها) دارای شکست خواهد بود
    # از سیگنال تولید شده تپش های قلب استخراج خواهد شد
    #و فاصله تا نزدیک ترین همسایه(موجود در تپش های داده ی اصلی)و واریانس بردار فاصله به انتهای تپش افزوده می شود##
    mx_cls_smpl=np.max(np.bincount(np.int16(wndws[:,-1])))
    rte_strch=2                  # increase sample rate to 1x or 2x or ...
    window_size = 50*rte_strch   #windows point in         50 or 100 or ...
    aug_quantity=1200
    epch=50
    points= int(len(ecg_cls[0])*rte_strch)
    cls=int(c)
    augment=[]
    print('\n\nPredicting Class = ', c)
    ecg_cls=np.array(wndws[np.where(wndws[:,-1]==c),:-1]) #np.array(wndws):اگر آخر ردیف شماره کلاس نباشد
    ecg_cls=np.array(ecg_cls[:,:-1])  #حذف برچسب کلاس برای یک بعدی کردن همه ی نمونه ها (ردیف ها)
    ecg_cls = signal.resample(ecg_cls,points,axis=1)
    ecg_cls_1d=[]
    for s in range (len(ecg_cls)):
        ecg_cls_1d=np.append(ecg_cls_1d,ecg_cls[s]) #سیگنال حاوی همه ی نمونه ها
    #print('ecg_cls.shape= ',np.shape(ecg_cls),'ecg_cls_1d.shape= ',np.shape(ecg_cls_1d))
    sgnl_1d= np.array(ecg_cls_1d)           #سیگنال حاوی همه ی نمونه ها که اشکالش در محل اتصال دو نمونه است 
    #print('sgnl shape = ', np.shape(sgnl))        #و در نتیجه نمونه ی دارای شکست تولید می کند
    i,j,z=0,0,0
    sgnl_1d_Nrm=[]
    #Y=[]
    for i in range(0,len(sgnl_1d)-window_size): #سیگنال کلی نرمال شده(حسب نقطه ی اول پنجره)حاوی همه نمونه ها
        first = sgnl_1d[i]
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append((sgnl_1d[i + j] - first) / first)
        sgnl_1d_Nrm.append(np.array(temp).reshape(window_size, 1))    #ردیف های نرمال شده
    #print('sgnl_1d_Nrm shape = ', np.shape(sgnl_1d_Nrm))
    X_prd=np.reshape(sgnl_1d_Nrm,(int(np.shape(sgnl_1d_Nrm)[0]),1,int(np.shape(sgnl_1d_Nrm)[1]),1))
    #print('X_prd shape = ', np.shape(X_prd))
    
    for r in range (0,len(ecg_cls)):  #len(ecg_cls)                  #آموزش شبکه هر بار با یک نمونه 
        X = []
        Y = []
        sgnl= np.array(ecg_cls[r])
        #print('sgnl shape = ', np.shape(sgnl))
        for i in range(0,len(sgnl) - window_size -rte_strch):#استخراج پنجره ونرمالسازی حسب نقطه اولش
            first = sgnl[i]
            temp = []
            temp2 = []
            for j in range(window_size):
                temp.append((sgnl[i + j] - first) / first)
            temp2.append((sgnl[i +window_size] - first) / first)
            X.append(np.array(temp).reshape(window_size, 1))
            Y.append(np.array(temp2).reshape(1,1))
        #print('shape X = ', np.shape(X))
        #print('shape Y = ', np.shape(Y))
        train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
        len_tr = len(train_X)
        train_X = np.array(train_X)
        test_X = np.array(test_X)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        train_X = train_X.reshape(train_X.shape[0],1,window_size,1)
        test_X = test_X.reshape(test_X.shape[0],1,window_size,1)
        model = lstm_cnn(window_size)
        model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=epch,batch_size=64,shuffle =False)
        if ((2*aug_quantity/len(ecg_cls))<len(ecg_cls)):
            sgnl_1d_Nrm_strt_indx=r*points
            sgnl_1d_Nrm_end_indx=(r+int(aug_quantity/len(ecg_cls))+1)*points
            if (sgnl_1d_Nrm_end_indx>(len(ecg_cls)*points)):
                sgnl_1d_Nrm_end_indx=int(len(ecg_cls))*points
                sgnl_1d_Nrm_strt_indx=(int(len(ecg_cls))-int(aug_quantity/len(ecg_cls))+1)*points
            X_prdction=np.array(X_prd[sgnl_1d_Nrm_strt_indx:sgnl_1d_Nrm_end_indx])            
        else:
            X_prdction=np.array(X_prd)            
        predicted  = model.predict(X_prdction)
        predicted = np.array(predicted[:,0]).reshape(-1)
        #بازگشت از فرم نورمال به واقعی برای انتهای سیگنال خام (محدوده ی تست و پیش بینی آن)
        pred_unrm=np.array(predicted)
        if ((2*aug_quantity/len(ecg_cls))<len(ecg_cls)):
            for z in range(len(predicted)):
                temp = sgnl_1d[z+sgnl_1d_Nrm_strt_indx]
                pred_unrm[z] = predicted[z] * temp + temp
        else:
            for z in range(len(predicted)):
                temp = sgnl_1d[z]
                pred_unrm[z] = predicted[z] * temp + temp
        augment.append(pred_unrm)
        #print('augmented shape for class ',c ,'Based on sample ', r, 'is = ', np.shape(augment))
        #clear_output(wait=True)
    print('befor add augmented shape for class ',c ,' = ', np.shape(augment))
    augment=wndwng(augment)
    cls_clm=np.zeros(len(augment))
    cls_clm=np.int16(cls_clm+cls)
    cls_clm=np.array([cls_clm]).T
    augment=np.concatenate((augment,cls_clm),axis=1)
    dis_var=dis_var_LSTM_lng(ecg_cls_1d,augment)
    augment=np.concatenate((augment,dis_var),axis=1)
    print('after add and Finall augmented shape for class ',c ,' = ', np.shape(augment))

    
def lstm_cnn(smpl_rt):
    model = Sequential()
    #add model layers
    model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu', input_shape=(None,smpl_rt,1))))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(256, kernel_size=1, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(400,return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(400,return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='RMSprop', loss='mse')
    return(model)

def LSTM_CNN_data(i,nmbr):    #دریافت داده از پنجره های آماده ی افزایشی که 6 مشخصه در انتها دارد برای کد کلاسیفای
    import glob               #3 end columns: 1-Class,2-distance,3-variance
    #from glob import glob
    cwd = os.getcwd() #
    adrs=cwd+'\LSTM_CNN_Aug_Data\LSTM_Cls'+str(i)+'.csv'
    files = glob.glob(adrs)
    alldata = []                                                      # List for storing all the data
    clstrs=len(files)
    nmbr_clstr=int(nmbr)
    for fname in files:                                               # Iterate mat files
        print('alldata shape = ', np.shape(alldata))
        data = pd.read_csv(adrs)                                     # Load mat file data into data.
        data=np.array(data)
        dt=data[data[:,-2].argsort()]                                 # Append data to the list
        '''print('data first row 2 last column = ', dt[0,-2:])
        print('data end row 2 last column = ', dt[-1,-2:])
        print('min dis = ', np.min(dt[:,-2]))
        print('min var = ', np.min(dt[:,-1]))
        print('csv shape = ', np.shape(dt))'''
        dt=list(dt[0:nmbr_clstr+1])
        alldata=list(alldata)
        alldata.append(dt)
    alldata=np.reshape(alldata,(np.shape(alldata)[0]*np.shape(alldata)[1],np.shape(alldata)[2]))
    alldata=np.array(alldata)
    print('final shape= ' , np.shape(alldata))
    return alldata[:nmbr,1:-5] 
          #6 end columns: 1-Class,2-Cluster,3-snsvty,gan epoch,4-epoch number,5-distance,6-variance
    
def dis_var_LSTM(dtst,P):
    ln_dtst=len(dtst[0])
    ln_P=len(P[0])
    strt_w=int(ln_dtst-ln_P)
    ds1 = dtst[:,strt_w:]-P      #.reshape(-1,1) #distance of every training instance
    dist_array=(ds1*10)**2       #معیار فاصله با حساسیت به توان 3 (حفظ علامت و تاثیر بیشتر تفاوت زیاد در نقاط)
    ds1=np.sum(dist_array,axis=1).reshape(-1,1)
    #print('dist2 = ', dist2)
    vr1=np.var((dist_array),axis=1).reshape(-1,1)
    var1=np.concatenate((ds1,vr1),axis=1)
    return(var1)

def dis_var_LSTM_lng(dtst,P): #dtst 2D and P is 1D
    ln_dtst=len(dtst)
    ln_P=len(P[0])
    strt_w=np.int16(ln_dtst-ln_P)
    ds1 = P-dtst[strt_w:]           #.reshape(-1,1) #distance of every training instance
    dist_array=(ds1*10)**2          #معیار فاصله با حساسیت به توان 3 (حفظ علامت و تاثیر بیشتر تفاوت زیاد در نقاط)
    ds1=np.sum(dist_array,axis=1).reshape(-1,1)
    #print('dist2 = ', dist2)
    vr1=np.var((dist_array),axis=1).reshape(-1,1)
    var1=np.concatenate((ds1,vr1),axis=1)
    return(var1)