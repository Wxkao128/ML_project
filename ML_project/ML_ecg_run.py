# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:10:28 2023

@author: User
"""

#%% Feature labeling

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from math import exp, sqrt, pi

train = np.load('ML_Train.npy')
test = np.load('ML_Test.npy')

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
    #diffs = np.diff(time[peaks])
    #avg_diff = np.round(np.mean(diffs),4)
    diffs = np.diff(peaks)                  # Ri+1 - Ri in index
    avg_rr_dis = np.round(np.mean(diffs),4)
    return avg_rr_dis



    
def Labeling_1(data,i):                         # For Lead I~III, V4~V6
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
    for nn in range(0,len(j_points_c)):                   #ST段上升/下降的feature
        j1 = data[j_points_c[nn]]
        j2 = data[j_points_c[nn] + 20]
        eachslope = (j2 - j1) / 0.04
        j_sum = j_sum + eachslope
    
    if len(j_points_c) == 0:
        ST_slope = 0
    else:
        ST_slope = j_sum / len(j_points_c)



    # find T peaks
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
    q_points_c = data[q_points_c].mean()
    s_points_c = data[s_points_c].mean()
    t_peaks = data[t_peaks].mean()
    return r_peaks, q_points_c, s_points_c, t_peaks, ST_slope
        
       

def Labeling_2(data,i):                         # For Lead V1~V3
    mean = data.mean()
    s_points_c,_ = find_peaks(-data, height = mean, distance = 280)
    #s_points_c = find_reverse(data, 280)
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
    r_peaks = np.setdiff1d(r_peaks, s_points_c)     # 上面的部分會標到S點
    avg_rr_dis = rr_distance(r_peaks)
    
    q_index_c = np.argwhere(mask).flatten() - 2 
    q_points_c = points[q_index_c]
    q_points_c = np.setdiff1d(q_points_c, r_peaks)  # 上面的部分會標到R點
    #q_points_c = q_points_c[0:-2]
    
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
    q_points_c = data[q_points_c].mean()
    s_points_c = data[s_points_c].mean()
    t_peaks = data[t_peaks].mean()
    return r_peaks, q_points_c, s_points_c, t_peaks, ST_slope
    


def PCA(X, n_components):
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)     # Step 1: Standardize the data
    cov_matrix = np.cov(X_std.T)                             # Step 2: Compute the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]

    pca_matrix = eigenvectors[:, :n_components]              # Step 5: Select the principal components
    
    #X_pca = np.dot(X_std, pca_matrix)
    
    return pca_matrix


time  = np.arange(0,10,0.002)     # 10 sec
columns = ['I_r','I_q','I_s','I_t','I_st','V1_r','V1_q','V1_s','V1_t','V1_st','V2_r','V2_q','V2_s','V2_t','V2_st','V3_r','V3_q','V3_s','V3_t','V3_st','V4_r','V4_q','V4_s','V4_t','V4_st','V5_r','V5_q','V5_s','V5_t','V5_st','V6_r','V6_q','V6_s','V6_t','V6_st']
training_features = pd.DataFrame(np.zeros((train.shape[0],35)),columns = columns)


for k in range(0,train.shape[0]):
    Patient_ID_train = train[k]
    Patient_ID_train = np.delete(Patient_ID_train, np.s_[1:6], 0)
    for i in range(0,7):
         Leads = Patient_ID_train[i,:]            # Lead number
         # Filter requirements.
         Wn = 20         # desired cutoff frequency of the filter, Hz
         b, a = signal.butter(4, Wn, 'lowpass',fs = 500)
         filtedData = signal.filtfilt(b, a, Leads )
         if i== 1 or i == 2 or i == 3:
             r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
         else:
             r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
        
        
         if np.isnan(r_peaks):
             training_features.iloc[k,5*i] = (training_features.iloc[k-1,5*i] + training_features.iloc[k-2,5*i] + training_features.iloc[k-3,5*i] + training_features.iloc[k-4,5*i] + training_features.iloc[k-5,5*i])/5
         else:
             training_features.iloc[k,5*i] = r_peaks
             
         if np.isnan(q_points_c):
             training_features.iloc[k,5*i+1] = (training_features.iloc[k-1,5*i+1]+training_features.iloc[k-2,5*i+1]+training_features.iloc[k-3,5*i+1]+training_features.iloc[k-4,5*i+1]+training_features.iloc[k-5,5*i+1])/5
         else:
             training_features.iloc[k,5*i+1] = q_points_c
         
         if np.isnan(s_points_c):
             training_features.iloc[k,5*i+2] = (training_features.iloc[k-1,5*i+2]+training_features.iloc[k-2,5*i+2]+training_features.iloc[k-3,5*i+2]+training_features.iloc[k-4,5*i+2]+training_features.iloc[k-5,5*i+2])/5
         else:
             training_features.iloc[k,5*i+2] = s_points_c
         
         if np.isnan(t_peaks):
              training_features.iloc[k,5*i+3] = (training_features.iloc[k-1,5*i+3]+training_features.iloc[k-2,5*i+3]+training_features.iloc[k-3,5*i+3]+training_features.iloc[k-4,5*i+3]+training_features.iloc[k-5,5*i+3])/5
         else:
             training_features.iloc[k,5*i+3] = t_peaks
         
         if np.isnan(ST_slope):
             training_features.iloc[k,5*i+4] = (training_features.iloc[k-1,5*i+4]+training_features.iloc[k-2,5*i+4]+training_features.iloc[k-3,5*i+4]+training_features.iloc[k-4,5*i+4]+training_features.iloc[k-5,5*i+4])/5
         else:
             training_features.iloc[k,5*i+4] = ST_slope 


testing_features = pd.DataFrame(np.zeros((test.shape[0],35)),columns = columns)
for k in range(0,test.shape[0]):
    Patient_ID_test = test[k]
    Patient_ID_test = np.delete(Patient_ID_test, np.s_[1:6], 0)
    for i in range(0,7):
         Leads = Patient_ID_test[i,:]            # Lead number
         # Filter requirements.
         Wn = 20         # desired cutoff frequency of the filter, Hz
         b, a = signal.butter(4, Wn, 'lowpass',fs = 500)
         filtedData = signal.filtfilt(b, a, Leads )
         if i== 1 or i == 2 or i == 3:
             r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
         else:
             r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
        
        
         if np.isnan(r_peaks):
             testing_features.iloc[k,5*i] = (testing_features.iloc[k-1,5*i] + testing_features.iloc[k-2,5*i] + testing_features.iloc[k-3,5*i] + testing_features.iloc[k-4,5*i] + testing_features.iloc[k-5,5*i])/5
         else:
             testing_features.iloc[k,5*i] = r_peaks
             
         if np.isnan(q_points_c):
             testing_features.iloc[k,5*i+1] = (testing_features.iloc[k-1,5*i+1]+testing_features.iloc[k-2,5*i+1]+testing_features.iloc[k-3,5*i+1]+testing_features.iloc[k-4,5*i+1]+testing_features.iloc[k-5,5*i+1])/5
         else:
             testing_features.iloc[k,5*i+1] = q_points_c
         
         if np.isnan(s_points_c):
             testing_features.iloc[k,5*i+2] = (testing_features.iloc[k-1,5*i+2]+testing_features.iloc[k-2,5*i+2]+testing_features.iloc[k-3,5*i+2]+testing_features.iloc[k-4,5*i+2]+testing_features.iloc[k-5,5*i+2])/5
         else:
             testing_features.iloc[k,5*i+2] = s_points_c
         
         if np.isnan(t_peaks):
              testing_features.iloc[k,5*i+3] = (testing_features.iloc[k-1,5*i+3]+testing_features.iloc[k-2,5*i+3]+testing_features.iloc[k-3,5*i+3]+testing_features.iloc[k-4,5*i+3]+testing_features.iloc[k-5,5*i+3])/5
         else:
             testing_features.iloc[k,5*i+3] = t_peaks
         
         if np.isnan(ST_slope):
             testing_features.iloc[k,5*i+4] = (testing_features.iloc[k-1,5*i+4]+testing_features.iloc[k-2,5*i+4]+testing_features.iloc[k-3,5*i+4]+testing_features.iloc[k-4,5*i+4]+testing_features.iloc[k-5,5*i+4])/5
         else:
             testing_features.iloc[k,5*i+4] = ST_slope


pca_matrix = PCA(training_features,20)

#training_std = (training_features - np.mean(training_features, axis=0)) / np.std(training_features, axis=0)
#testing_std = (testing_features - np.mean(testing_features, axis=0)) / np.std(testing_features, axis=0)
#train_pca = np.dot(training_std, pca_matrix)
#test_pca = np.dot(testing_std, pca_matrix)
train_pca = np.dot(training_features, pca_matrix)
test_pca = np.dot(testing_features, pca_matrix)



#%% Classsifier

# NORM:0~7493, STTC:7494~9513, CD:9514~11020, MI:1021~12208

#def mean(data):
 #   return np.mean(data)

#def stdev(data):
 #   avg = log_mean(data)
    #x = data.values.tolist()
  #  variance = (sum([(x - avg)**2 for x in data]) / len(data))
   # return sqrt(variance)

# Normal PDF
def Gaussian_NB(train_pca,test_pca,priors):
    mean = np.zeros((4,20))
    mean[0] = np.array(np.mean(train_pca[0:7493], axis = 0))#.reshape(1,20)    # NORM_mean
    mean[1] = np.array(np.mean(train_pca[7494:9513], axis = 0))#.reshape(1,20)    # STTC_mean
    mean[2] = np.array(np.mean(train_pca[9514:11020], axis = 0))#.reshape(1,20)    # CD_mean
    mean[3] = np.array(np.mean(train_pca[11020:12208], axis = 0))#.reshape(1,20)    # MI_mean
    std = np.zeros((4,20))
    std[0] = np.array(np.std(train_pca[0:7493], axis = 0))           # NORM_std
    std[1] = np.array(np.std(train_pca[7494:9513], axis = 0))           # STTC_std
    std[2] = np.array(np.std(train_pca[9514:11020], axis = 0))           # CD_std
    std[3] = np.array(np.std(train_pca[11020:12208], axis = 0))           # MI_std
    #cov = np.zeros((4,1))
    #cov[0] = np.linalg.inv(np.diag(std[0]*std[0]))
    #cov[1] = np.linalg.inv(np.diag(std[1]*std[1]))
    #cov[2] = np.linalg.inv(np.diag(std[2]*std[2]))
    #cov[3] = np.linalg.inv(np.diag(std[3]*std[3]))


    P = np.ones((len(test_pca),len(mean)))
    
    for index in range(0,len(test_pca)):                # 6000 testing patients
        for i in range(0,len(mean)):                    # 4 categories
            count = 0
            for j in range (0,test_pca.shape[1]):       # 20 features
                den = sqrt(2*pi) * std[i,j]
                #temp = test_pca[index,j] - mean[i,j]
                N = (-0.5 * (test_pca[index,j] - mean[i,j])**2 ) / (std[i,j]**2)
                p = (1/den) * exp(N)
                count = count + np.log(p)
            P[index,i] = count + np.log(priors[i])

    return  P
    #return mean,std

# Classify
priors = np.array([7494/12209,2020/12209,1507/12209,1188/12209])
#mean, std =  Gaussian_NB(train_pca,test_pca,priors)
P_matrix =  Gaussian_NB(train_pca,test_pca,priors)

classfication = []


for index in range(0,len(test_pca)):
    temp = P_matrix[index]
    classfication.append(np.argmax(temp))



prediction = np.zeros((6000,2))
prediction[:,0] = np.arange(0,6000,1)
prediction[:,1] = classfication
ML14_prediction = pd.DataFrame(prediction, columns = ['SubjectId','Label'])
ML14_prediction = ML14_prediction.astype(int)
ML14_prediction.to_csv('ML14_prediction_0517.csv', encoding='utf-8', index = False)
