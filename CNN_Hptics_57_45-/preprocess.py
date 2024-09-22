import numpy as np
from scipy import signal
from scipy.signal import find_peaks

def wndwng(x,snstvty=2):                   #روش تشخیص ایپاک ماتریس 2 بعدی
    smpl_rte=480
    i=0
    wndws=np.array([])
    #for i in range (1,2):                 #print(np.shape(x))
    pnts=len(x[0])-5                       #طول سیگنال منهای یک 
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


def wndwng0(x,snstvty=2):                   #روش تشخیص ایپاک آرایه 1 بعدی
    smpl_rte=480
    i=0
    wndws=np.array([])
    #for i in range (1,2):                 #print(np.shape(x))
    if x.ndim==1 :
        x=np.array([x])
    pnts=len(x[0])-1                       #طول سیگنال منهای یک 
    if snstvty==1 :                        #تشخیص ایپاک پایه
        for j in range (np.shape(x)[0]):
            xj=np.array(x[j])
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