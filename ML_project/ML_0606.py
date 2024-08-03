# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 22:12:18 2023

@author: User
"""


import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from math import exp, sqrt, pi, ceil
import random
import winsound


Training_set_2 = np.load('ML_Training_set_2.npy')
Vad_set_2 = np.load('ML_Validation_set_2.npy')
#%%

# find R peak
def find_peak(data,distance):
    # default distance = 300
    mean = data.mean()
    std = data.std()
    peaks, _ = find_peaks(data, height = mean+std, distance = distance)
    return peaks

# find all troughs in data, the prerequisite for finding Q,S,...
def find_reverse(data,distance):
    # default distance = 20
    troughs,_= find_peaks(-filtedData, distance = distance)
    return troughs


# calculate R-R distance
def rr_distance(peaks):
    diffs = np.diff(peaks)                  # 第i+1個R_peak, 第i個R_peak 之間差幾個資料
    avg_rr_dis = np.round(np.mean(diffs),4)
    return avg_rr_dis



    
def Labeling_1(data,i):                         # For Lead I, V4~V6
    r_peaks = find_peak(data, 280)
    r_peaks = r_peaks[1:-2] # 去掉頭一筆尾兩筆資料
    avg_rr_dis = rr_distance(r_peaks)
    
    #for finding Q,S,J... points
    troughs = find_reverse(data, 20)
    
    #combine all the extrema
    points = np.concatenate((troughs,r_peaks))
    points = np.sort(points)
    
    # a set of Q point candidates
    mask = np.isin(points, r_peaks)
    q_index_c = np.argwhere(mask).flatten() - 1
    q_points_c = points[q_index_c]
    
    # a set of S point candidates
    s_index_c = np.argwhere(mask).flatten() + 1 
    s_points_c = points[s_index_c]
    

    # a set of J point candidates
    j_points_c = []
    for n in range(0,len(s_points_c)):
        currentpoint = s_points_c[n]
        temp = 0
        while temp < 250:
            if filtedData[currentpoint + temp] * filtedData[currentpoint + temp + 1] < 0:
                j_points_c.append(currentpoint + temp + 1)
                break
            temp += 1
    
    j_sum = 0
    for nn in range(0,len(j_points_c)):         #ST段上升/下降的feature
        j1 = data[j_points_c[nn]]
        j2 = data[j_points_c[nn] + 20]
        eachslope = (j2 - j1) / 0.04
        j_sum = j_sum + eachslope
    
    if len(j_points_c) == 0:                    # If cannot find J point, ST_slope = 0
        ST_slope = 0
    else:
        ST_slope = j_sum / len(j_points_c)



    # find T peaks
    t_peak = []
    if len(r_peaks) < 5 or avg_rr_dis == 0:     # If cannot find R peaks, T_peak = 0
        t_peaks = np.array([0])
        
    else:
        for j in range(len(r_peaks)):           # 從R_peak往右找0.15~0.5倍的rr_interval，其中最大的值    
            a = round(r_peaks[j] + 0.15* avg_rr_dis)
            b = round(r_peaks[j] + 0.5*avg_rr_dis)
            P = np.absolute(data)
            interval = P[a:b]
            t = np.argmax(interval)
            t_peak.append(int(t+a))
        t_peaks = np.array([t_peak])
        
    r_peaks = data[r_peaks].mean()
    if np.isnan(r_peaks):                       # If data[r_peaks] contains nan, return r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = nan
        r_peaks = np.nan
        q_points_c = np.nan
        s_points_c = np.nan
        t_peaks = np.nan
        ST_slope = np.nan
        
    else:
        q_points_c = data[q_points_c].mean()
        s_points_c = data[s_points_c].mean()
        t_peaks = data[t_peaks].mean()
        
    return r_peaks, q_points_c, s_points_c, t_peaks, ST_slope
        
       

def Labeling_2(data,i):                         # For Lead V1~V3 / MI: V4~V6
    mean = data.mean()
    s_points_c,_ = find_peaks(-data, height = mean, distance = 280)
    s_points_c = s_points_c [1:-2]
    
    y = np.absolute(data)
    troughs = find_peaks(y, distance = 10)
    troughs = troughs[0]
    points = np.concatenate((troughs,s_points_c))
    points = np.sort(points)
    mask = np.isin(points, s_points_c)
    
    r_index = np.argwhere(mask).flatten() - 1 
    r_index = np.sort(r_index)
    r_peaks = points[r_index]
    r_peaks = np.setdiff1d(r_peaks, s_points_c)     
    avg_rr_dis = rr_distance(r_peaks)
    
    q_index_c = np.argwhere(mask).flatten() - 2 
    q_points_c = points[q_index_c]
    q_points_c = np.setdiff1d(q_points_c, r_peaks)  
    
    # a set of J point candidates
    j_points_c = []
    for n in range(0,len(s_points_c)):
        currentpoint = s_points_c[n]
        temp = 0
        while temp < 250:
            if filtedData[currentpoint + temp] * filtedData[currentpoint + temp + 1] < 0:
                j_points_c.append(currentpoint + temp + 1)
                break

            temp += 1
    
    j_sum = 0
    for nn in range(0,len(j_points_c)):                   #ST段上升/下降的feature
        j1 = data[j_points_c[nn]]
        j2 = data[j_points_c[nn] + 20]
        eachslope = (j2 - j1) / 0.04
        j_sum = j_sum + eachslope
        
    if len(j_points_c) == 0:
        ST_slope = 0
    else:
        ST_slope = j_sum / len(j_points_c)
    
    
    t_peak = []
    if len(r_peaks) < 5 or avg_rr_dis == 0:
        t_peaks = np.array([0])
        
    else:
        for j in range(len(r_peaks)):
            a = round(r_peaks[j] + 0.15* avg_rr_dis)
            b = round(r_peaks[j] + 0.5*avg_rr_dis)
            P = np.absolute(data)
            interval = P[a:b]
            t = np.argmax(interval)
            t_peak.append(int(t+a))
        t_peaks = np.array([t_peak])
        
    r_peaks = data[r_peaks].mean() 
    if np.isnan(r_peaks):
        r_peaks = np.nan
        q_points_c = np.nan
        s_points_c = np.nan
        t_peaks = np.nan
        ST_slope = np.nan
    else:
        q_points_c = data[q_points_c].mean()
        s_points_c = data[s_points_c].mean()
        t_peaks = data[t_peaks].mean()
        
    return r_peaks, q_points_c, s_points_c, t_peaks, ST_slope
    

#%%
# Plot figure
print("Yes!")


#%%
# Training_set
'''
columns = ['I_r','I_q','I_s','I_t','I_st','V1_r','V1_q','V1_s','V1_t','V1_st','V2_r','V2_q','V2_s','V2_t','V2_st','V3_r','V3_q','V3_s','V3_t','V3_st','V4_r','V4_q','V4_s','V4_t','V4_st','V5_r','V5_q','V5_s','V5_t','V5_st','V6_r','V6_q','V6_s','V6_t','V6_st']
training_set_features = pd.DataFrame(np.zeros((Training_set_2.shape[0],35)),columns = columns)

for k in range(0,Training_set_2.shape[0]):
    Patient_ID_train = Training_set_2[k]
    Patient_ID_train = np.delete(Patient_ID_train, np.s_[1:6], 0)
    for i in range(0,7):
         Leads = Patient_ID_train[i,:]            # Lead number
         # Filter requirements.
         Wn = 20                                  # desired cutoff frequency of the filter, Hz
         b, a = signal.butter(4, Wn, 'lowpass',fs = 500)
         filtedData = signal.filtfilt(b, a, Leads )

         if i== 1 or i == 2 or i == 3:
             r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
         else:
             if filtedData.mean() < 0:            # V4~V6 have big S wave and small R wave
                 r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
                 
             else:
                 r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)


         if np.isnan(r_peaks):
             training_set_features.iloc[k,5*i] = (training_set_features.iloc[k-1,5*i] + training_set_features.iloc[k-2,5*i] + training_set_features.iloc[k-3,5*i] + training_set_features.iloc[k-4,5*i] + training_set_features.iloc[k-5,5*i])/5
         else:
             training_set_features.iloc[k,5*i] = r_peaks
             
         if np.isnan(q_points_c):
             training_set_features.iloc[k,5*i+1] = (training_set_features.iloc[k-1,5*i+1]+training_set_features.iloc[k-2,5*i+1]+training_set_features.iloc[k-3,5*i+1]+training_set_features.iloc[k-4,5*i+1]+training_set_features.iloc[k-5,5*i+1])/5
         else:
             training_set_features.iloc[k,5*i+1] = q_points_c
         
         if np.isnan(s_points_c):
             training_set_features.iloc[k,5*i+2] = (training_set_features.iloc[k-1,5*i+2]+training_set_features.iloc[k-2,5*i+2]+training_set_features.iloc[k-3,5*i+2]+training_set_features.iloc[k-4,5*i+2]+training_set_features.iloc[k-5,5*i+2])/5
         else:
             training_set_features.iloc[k,5*i+2] = s_points_c
         
         if np.isnan(t_peaks):
              training_set_features.iloc[k,5*i+3] = (training_set_features.iloc[k-1,5*i+3]+training_set_features.iloc[k-2,5*i+3]+training_set_features.iloc[k-3,5*i+3]+training_set_features.iloc[k-4,5*i+3]+training_set_features.iloc[k-5,5*i+3])/5
         else:
             training_set_features.iloc[k,5*i+3] = t_peaks
         
         if np.isnan(ST_slope):
             training_set_features.iloc[k,5*i+4] = (training_set_features.iloc[k-1,5*i+4]+training_set_features.iloc[k-2,5*i+4]+training_set_features.iloc[k-3,5*i+4]+training_set_features.iloc[k-4,5*i+4]+training_set_features.iloc[k-5,5*i+4])/5
         else:
             training_set_features.iloc[k,5*i+4] = ST_slope

# Validation_set

vad_features = pd.DataFrame(np.zeros((Vad_set_2.shape[0],35)),columns = columns)
for k in range(0,Vad_set_2.shape[0]):
    Patient_ID_test = Vad_set_2[k]
    Patient_ID_test = np.delete(Patient_ID_test, np.s_[1:6], 0)
    for i in range(0,7):
         Leads = Patient_ID_test[i,:]            # Lead number
         # Filter requirements.
         Wn = 20                                 # desired cutoff frequency of the filter, Hz
         b, a = signal.butter(4, Wn, 'lowpass',fs = 500)
         filtedData = signal.filtfilt(b, a, Leads )
         
         if i== 1 or i == 2 or i == 3:
             r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
         else:
             if filtedData.mean() < 0:           # V4~V6 have big S wave and small R wave
                 r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
                 
             else:
                 r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
     
        
         if np.isnan(r_peaks):
             vad_features.iloc[k,5*i] = (vad_features.iloc[k-1,5*i] + vad_features.iloc[k-2,5*i] + vad_features.iloc[k-3,5*i] + vad_features.iloc[k-4,5*i] + vad_features.iloc[k-5,5*i])/5
         else:
             vad_features.iloc[k,5*i] = r_peaks
             
         if np.isnan(q_points_c):
             vad_features.iloc[k,5*i+1] = (vad_features.iloc[k-1,5*i+1]+vad_features.iloc[k-2,5*i+1]+vad_features.iloc[k-3,5*i+1]+vad_features.iloc[k-4,5*i+1]+vad_features.iloc[k-5,5*i+1])/5
         else:
             vad_features.iloc[k,5*i+1] = q_points_c
         
         if np.isnan(s_points_c):
             vad_features.iloc[k,5*i+2] = (vad_features.iloc[k-1,5*i+2]+vad_features.iloc[k-2,5*i+2]+vad_features.iloc[k-3,5*i+2]+vad_features.iloc[k-4,5*i+2]+vad_features.iloc[k-5,5*i+2])/5
         else:
             vad_features.iloc[k,5*i+2] = s_points_c
         
         if np.isnan(t_peaks):
              vad_features.iloc[k,5*i+3] = (vad_features.iloc[k-1,5*i+3]+vad_features.iloc[k-2,5*i+3]+vad_features.iloc[k-3,5*i+3]+vad_features.iloc[k-4,5*i+3]+vad_features.iloc[k-5,5*i+3])/5
         else:
             vad_features.iloc[k,5*i+3] = t_peaks
         
         if np.isnan(ST_slope):
             vad_features.iloc[k,5*i+4] = (vad_features.iloc[k-1,5*i+4]+vad_features.iloc[k-2,5*i+4]+vad_features.iloc[k-3,5*i+4]+vad_features.iloc[k-4,5*i+4]+vad_features.iloc[k-5,5*i+4])/5
         else:
             vad_features.iloc[k,5*i+4] = ST_slope
             
'''