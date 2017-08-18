def ts_to_time(ts):
    # Convert device timestamps to datetime objects
    utc_dt=datetime.datetime(year=2001,month=1,day=1)+datetime.timedelta(microseconds=np.float(ts))
    return utc_dt
######
# Rounding the device timestamp to the nearest 'T' minute interval
T_minutes=30
offset_T=T_minutes*60*1e9  # T minutes in nanoseconds 
df_test['time_rnd']=pd.to_datetime(((df_test.timestamp.apply(ts_to_time).astype(np.int64) // offset_T + 1 ) * offset_T))
###################

# Extract str content between two special characters
import re
re.findall(r'_(.*?)_',str)
#####################

# Helper functions for computing the top-K accuracy for keras
def topKlabels(p_Mat,k=2):
    top_k=np.apply_along_axis(lambda x: np.argpartition(x, -k)[-k:],1,p_Mat )
def acc_topK(p_Mat,y_test,k=2):
    top_k=np.apply_along_axis(lambda x: np.argpartition(x, -k)[-k:],1,p_Mat )
    n_correct=0
    for i in range(k):
        n_correct+=(y_test==top_k[:,i] ).sum()
    return np.float(n_correct)/len(y_test)
def acc_topK_Keras(X_test,y_test,model,k=2):
    p_Mat=model.predict_proba(X_test,verbose=0)
    top_k=np.apply_along_axis(lambda x: np.argpartition(x, -k)[-k:],1,p_Mat )
    n_correct=0
    for i in range(k):
        n_correct+=(y_test==top_k[:,i] ).sum()
    return np.float(n_correct)/len(y_test)

## Helper function for class-aware splits
def split_class_aware(X,y,n_splits=2):
    
    skf = StratifiedKFold(n_splits,shuffle=True)
    ind_split=skf.split(np.zeros(len(X)), y)
    train_index=ind_split.next()[0]
    test_index=ind_split.next()[1]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return (X_train, X_test,y_train, y_test)

## Helper functions for Keras-init (GPU/CPU control)
import tensorflow as tf
from keras.backend import tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.80), 
                                                 device_count={'CPU':4, 'GPU':1})))


## Parsing data from the sensor log app:
def parse_data(fileName,save_output=0):
    with open(fileName, 'r') as file:
        text = file.readlines()
        metaDataRegExp = re.match(r'(.*)\|(.*)', text[1], re.M|re.I)

    entries = len(text) - 3
    A = np.empty([entries, 4])
    c = 0
    #have to figure out how to change this according to how many labels there are
    for line in text[3:]:
        regExp = re.match(r'(.*)\|(.*)\|(.*)\|(.*)', line, re.M|re.I)
        array = regExp.group(3).strip(']').strip('[').split(',')
        timestamp = regExp.group(4)
        A[c][0] = float(array[0])
        A[c][1] = float(array[1])
        A[c][2] = float(array[2])
        A[c][3] = int(timestamp)
        c += 1
    nameRegExp = re.match(r'(.*)\.txt', fileName, re.M|re.I)
    name = nameRegExp.group(1)
    outputFileName = name + ".csv"
    df_raw=pd.DataFrame(data=A,columns=['x','y','z','time_stamp'])
    df_raw['time']=(df_raw.time_stamp-df_raw.time_stamp.min())/1e3
    df_out=df_raw.loc[:,['time','x','y','z']]
    f_s_raw=df_out.shape[0]/df_out.time.max()
    print('The average raw sampling rate is: '+str(f_s_raw)+' Hz')
    if(save_output):
        df_out.to_csv(outputFileName,index=False)
        print('File saved as: '+outputFileName)
    return df_out,f_s_raw

## Standard customized Sig-proc functions:

import numpy as np
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
%matplotlib inline
#############################################################
from IPython.display import Image
import os
#############################################################
from scipy.signal import argrelmin,butter, filtfilt, medfilt,freqz
from scipy.interpolate import interp1d
from scipy import array
from scipy.fftpack import fft as sp_fft
from scipy.stats import zscore as sp_zscore
#############################################################
from tqdm import tqdm
#############################################################


# x_median_filt = sp.signal.medfilt(x_uniform,median_filt_order)  
def median_filter(x_uniform,median_filt_order):
    x_median_filt=medfilt(x_uniform,median_filt_order) 
    return x_median_filt 

def lowpass_filter_butterworth(x_in, f_c, f_s, order):
    b, a = butter(order, f_c/ (0.5 * f_s), btype='low', analog=False)
    x_out = filtfilt(b, a, x_in)
    return x_out

def highpass_filter_butterworth(x_in, f_c, f_s, order):
    b, a = butter(order, f_c/ (0.5 * f_s), btype='high', analog=False)
    x_out = filtfilt(b, a, x_in)
    return x_out

def bandpass_filter_butterworth(x_in,f_c_low,f_c_high,f_s,order):
    b, a = butter(order, [f_c_low/ (0.5 * f_s), f_c_high/ (0.5 * f_s)], btype='band', analog=False)
    x_out = filtfilt(b, a, x_in)
    return x_out

def plot_spectrum(y,f_s):
    # Number of sample points
    N = len(y)
    # sample spacing
    T = 1.0 / f_s
    x = np.linspace(0.0, N*T, N)
    yf = sp_fft(y)
    freq_vec = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    Y_f=2.0/N * np.abs(yf[0:int(N/2)])
    plt.plot(freq_vec, Y_f)
    return Y_f,freq_vec
# Extrapolation code from: 
# http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike

def extrapolate_1d(time_in,values_in,time_uniform,kind='linear'):
    f_x = extrap1d(interp1d(time_in,values_in, kind))
    values_out =f_x(time_uniform)
    return values_out
#############################################################
# Simple toy experiment:
time_in=np.unique(np.sort(np.random.randint(4,50,100)))
values_in=np.sin(2*np.pi*0.15*time_in)
plt.plot(time_in,values_in,'-.*r')
time_out=np.linspace(0,51.0,100)
values_out=extrapolate_1d(time_in,values_in,time_out,kind='linear')
plt.plot(time_out,values_out,'--b')
# Toy example:
f1=3.7
f2=12
fs=100
t_vec=np.linspace(0,1,fs)
plt.figure()
N=len(t_vec)
x_in=3*np.sin(2*np.pi*f1*t_vec)+3*np.sin(2*np.pi*f2*t_vec)+np.random.randn(N)
_,_=plot_spectrum(x_in,fs)
f_c=5
x_lf=lowpass_filter_butterworth(x_in, f_c, fs, order=5)
_,_=plot_spectrum(x_lf,fs)
x_hf=highpass_filter_butterworth(x_in, f_c, fs, order=5)
_,_=plot_spectrum(x_hf,fs)

################
def interp_filt(df_out,f_s,filt,f_c,order=3):
    
    ts_max=max(df_out.time)
    time_uniform=np.linspace(0,ts_max,np.ceil(f_s*ts_max))
    df_out_if=pd.DataFrame(data=time_uniform,columns=['time'])
    axes=['x','y','z']
    for axis in axes:
        a_int=extrapolate_1d(df_out.time,df_out[axis],time_uniform,kind='linear')
        if(filt=='low'):
            df_out_if[axis]= lowpass_filter_butterworth(a_int,f_c[0], f_s, order=3)
        elif (filt=='band'):
            df_out_if[axis]= bandpass_filter_butterworth(a_int,f_c[0],f_c[1], f_s, order=3)
    df_out_if['mag']=np.sqrt(df_out_if.x**2+df_out_if.y**2+df_out_if.z**2)
    return df_out_if
def plt_specgram(t,x,NFFT = 1024,n_overlap=900):

    dt=t[1]-t[0] # the length of the windowing segments
    f_s = int(1.0/dt)  # the sampling frequency

    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    plt.figure(figsize=(30,8))
    ax1 = plt.subplot(211)
    plt.plot(t, x)
    plt.subplot(212, sharex=ax1)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=f_s, noverlap=n_overlap)
    plt.show()
