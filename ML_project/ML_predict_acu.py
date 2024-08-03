# -*- coding: utf-8 -*-
"""
Created on Sat May 20 12:29:20 2023

@author: User
"""

# Feature labeling

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from math import exp, sqrt, pi
from time import time as tt

time1 = tt()
#Training_set = np.load('ML_Training_set.npy')
#Vad_set = np.load('ML_Validation_set.npy')
Training_set = np.load('ML_Train.npy')
Vad_set = np.load('ML_Test.npy')
#test = np.load('ML_Test.npy')
time2 = tt()
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

    return pca_matrix, eigenvalues

#%%
time3 = tt()
time  = np.arange(0,10,0.002)     # 10 sec
columns = ['I_r','I_q','I_s','I_t','I_st','V1_r','V1_q','V1_s','V1_t','V1_st','V2_r','V2_q','V2_s','V2_t','V2_st','V3_r','V3_q','V3_s','V3_t','V3_st','V4_r','V4_q','V4_s','V4_t','V4_st','V5_r','V5_q','V5_s','V5_t','V5_st','V6_r','V6_q','V6_s','V6_t','V6_st']
training_set_features = pd.DataFrame(np.zeros((Training_set.shape[0],35)),columns = columns)
# now: training_set_features.shape=(12209, 35), Training_set.shape=(12209, 12, 5000)
 

for k in range(0,Training_set.shape[0]):
    Patient_ID_train = Training_set[k]
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
             #r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
             if i == 4 or i == 5:
                 lead_mean = filtedData.mean()
                 if lead_mean<0:
                    r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
                    
                 else:
                    r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
                    
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
             



vad_features = pd.DataFrame(np.zeros((Vad_set.shape[0],35)),columns = columns)
for k in range(0,Vad_set.shape[0]):
    Patient_ID_test = Vad_set[k]
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
             #r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
             if i == 4 or i == 5:
                 lead_mean = filtedData.mean()
                 if lead_mean<0:
                    r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_2(filtedData,i)
                    
                 else:
                    r_peaks, q_points_c, s_points_c, t_peaks, ST_slope = Labeling_1(filtedData,i)
                    
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



# 原始檔案: NORM:0~7493, STTC:7494~9513, CD:9514~11020, MI:10021~12208
# Training_set:  NORM:0~5244, STTC:5245~6658, CD:6659~7713, MI:7714~8544
# Vad_set:       NORM:0~2248, STTC:2249~2854, CD:2855~3307, MI:3308~3664

best_feature = 19 # number of choosing the most representative features
pca_matrix, eigenvalues = PCA(training_set_features,best_feature)
training_pca = np.dot(training_set_features, pca_matrix)
vad_pca = np.dot(vad_features, pca_matrix)
# now: pca_matrix.shape=(35,19),  training_pca.shape=(12209, 19)
# now: vad_pca.shape=(6000, 19)


time4 = tt()



'''  
# 確認要取PCA幾個feature，跑完d1 = 19
d = 0
for m in range(len(eigenvalues)):
    d = d + eigenvalues[m]
    count = d/np.sum(eigenvalues)
    if count > 0.9:
        d1 = m
        break
'''
'''
#training_std = (training_features - np.mean(training_features, axis=0)) / np.std(training_features, axis=0)
#testing_std = (testing_features - np.mean(testing_features, axis=0)) / np.std(testing_features, axis=0)
#train_pca = np.dot(training_std, pca_matrix)
#test_pca = np.dot(testing_std, pca_matrix)

#NORM = training_features.iloc[0:7493]
#STTC = training_features.iloc[7494:9513]
#CD = training_features.iloc[9514:11020]
#MI = training_features.iloc[11020:12208]

'''

#%% Naive Bayes Classsifier

# NORM:0~7493, STTC:7494~9513, CD:9514~11020, MI:11021~12208
# Training_set:  NORM:0~5244, STTC:5245~6658, CD:6659~7713, MI:7714~8544
# Vad_set:       NORM:0~2248, STTC:2249~2854, CD:2855~3307, MI:3308~3664


#NORM = training_set_features.iloc[0:5245]
#NORM_train = NORM.sample(frac = 0.5, random_state = 10)
#NORM_pca =  np.dot(NORM_train, pca_matrix)

# Normal PDF
def Gaussian_NB(train_pca,test_pca,priors):
    mean = np.zeros((4,best_feature))
    mean[0] = np.array(np.mean(train_pca[0:5245], axis = 0))        # NORM_mean
    #mean[0] = np.array(np.mean(NORM_pca, axis = 0))   
    mean[1] = np.array(np.mean(train_pca[7714:8545], axis = 0))     # MI_mean
    mean[2] = np.array(np.mean(train_pca[5245:6659], axis = 0))     # STTC_mean
    mean[3] = np.array(np.mean(train_pca[6659:7714], axis = 0))     # CD_mean
    
    std = np.zeros((4,best_feature))
    std[0] = np.array(np.std(train_pca[0:5245], axis = 0))          # NORM_std
    #std[0] = np.array(np.std(NORM_pca, axis = 0))          # NORM_std
    std[1] = np.array(np.std(train_pca[7714:8545], axis = 0))       # MI_std 
    std[2] = np.array(np.std(train_pca[5245:6659], axis = 0))       # STTC_std
    std[3] = np.array(np.std(train_pca[6659:7714], axis = 0))       # CD_std



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
#priors = np.array([7494/12209,1188/12209,2020/12209,1507/12209])

priors = np.array([5244/8544,831/8544,1414/8544,1055/8544])
#priors = np.array([2622/5922,831/5922,1414/5922,1055/5922])


#mean, std =  Gaussian_NB(train_pca,test_pca,priors)
#P_matrix =  Gaussian_NB(train_pca,test_pca,priors)
P_matrix =  Gaussian_NB(training_pca,vad_pca,priors)

classfication = []
validation =  np.zeros((len(vad_pca),1))
validation[2249:2855] = 2
validation[2855:3308] = 3
validation[3308:3665] = 1
count = 0
for index in range(0,len(vad_pca)):
    temp = P_matrix[index]
    #classfication.append(np.argmax(temp))
    
    if np.argmax(temp) == 0:
        rnd_num = np.random.rand()
        if rnd_num < 0.25:
            classfication.append(2)
        elif rnd_num > 0.75:
            classfication.append(3)
        else:
            classfication.append(np.argmax(temp))
    else:
        classfication.append(np.argmax(temp))
        
    if validation[index] == np.argmax(temp):
        count += 1
time5 = tt()    
print("Accuracy =",round((count/len(vad_pca)),6))

print(f"       Load file time = {round(time2-time1,5)}")
print(f"   Data analysis time = {round(time5-time2,5)}")
print(f"Total processing time = {round(time5-time1,5)}")
# NORM training data quantities not modified: Accuracy = 0.587176
# NORM training data quantities modified to 2622: Accuracy = 0.579263

# Modify labeling with original definition
#NORM training data quantities not modified: Accuracy = 0.589086


prediction = np.zeros((6000,2))
prediction[:,0] = np.arange(0,6000,1)
prediction[:,1] = classfication
ML14_prediction = pd.DataFrame(prediction, columns = ['SubjectId','Label'])
ML14_prediction = ML14_prediction.astype(int)
ML14_prediction.to_csv('ML14_prediction_0517_naive_bayes_f6.csv', encoding='utf-8', index = False)

#%% Gaussian Multivariate Classifier

#NORM = training_set_features.iloc[0:5245]
#NORM_train = NORM.sample(frac = 0.5, random_state = 10)
#NORM_pca =  np.dot(NORM_train, pca_matrix)

# Normal PDF
def Gaussian(train_pca,test_pca,priors):
    mean = np.zeros((4,best_feature))
    mean[0] = np.array(np.mean(train_pca[0:5245], axis = 0))        # NORM_mean
    #mean[0] = np.array(np.mean(NORM_pca, axis = 0))   
    mean[1] = np.array(np.mean(train_pca[7714:8545], axis = 0))     # MI_mean
    mean[2] = np.array(np.mean(train_pca[5245:6659], axis = 0))     # STTC_mean
    mean[3] = np.array(np.mean(train_pca[6659:7714], axis = 0))     # CD_mean
    
    std = np.zeros((4,best_feature))
    std[0] = np.array(np.std(train_pca[0:5245], axis = 0))          # NORM_std
    #std[0] = np.array(np.std(NORM_pca, axis = 0))          # NORM_std
    std[1] = np.array(np.std(train_pca[7714:8545], axis = 0))       # MI_std 
    std[2] = np.array(np.std(train_pca[5245:6659], axis = 0))       # STTC_std
    std[3] = np.array(np.std(train_pca[6659:7714], axis = 0))       # CD_std
    
    cov = []
    cov.append(np.cov(train_pca[0:5245].T))
    #cov.append(np.cov(NORM_pca.T))
    cov.append(np.cov(train_pca[7714:8545].T))
    cov.append(np.cov(train_pca[5245:6659].T))
    cov.append(np.cov(train_pca[6659:7714].T))
    
    #return cov, mean

    P = np.ones((len(test_pca),len(mean)))
    
    for index in range(0,len(test_pca)):                # 6000 testing patients
        for i in range(0,len(mean)):                    # 4 categories
            #for j in range (0,test_pca.shape[1]):       # 20 features
            den = (2*pi)**(test_pca.shape[1]/2) * np.linalg.det(cov[i])
            temp = (test_pca[index] - mean[i]).reshape(best_feature,1)
            N = -0.5 * np.dot( np.dot(temp.T,np.linalg.inv(cov[i])), temp)
            p = (1/den) * exp(N)
            P[index,i] = np.log(p) + np.log(priors[i])
            #P[index,i] = p + np.log(priors[i])

    return  P
    #return  den,temp,N

  

# Classify
#priors = np.array([7494/12209,2020/12209,1507/12209,1188/12209])
priors = np.array([5244/8544,831/8544,1414/8544,1055/8544])
#priors = np.array([2622/5922,831/5922,1414/5922,1055/5922])

#den,temp,N =  Gaussian_NB(train_pca,test_pca,priors)
P_matrix =  Gaussian(training_pca,vad_pca,priors)


classfication = []
validation =  np.zeros((len(vad_pca),1))
validation[2249:2855] = 2
validation[2855:3308] = 3
validation[3308:3665] = 1
count = 0
for index in range(0,len(vad_pca)):
    temp = P_matrix[index]
    classfication.append(np.argmax(temp))
    if validation[index] == np.argmax(temp):
        count += 1
time6 = tt()    
print("Accuracy =",round((count/len(vad_pca)),6))
print(f"Total processing time2= {round(time6-time1,5)}")

# Original definition: np.log(p) + np.log(priors[i])
# NORM training data quantities not modified: Accuracy = 0.624829, with warning: divide by zero encountered in log
# NORM training data quantities modified to 2622: Accuracy = 0.624011, with warning: divide by zero encountered in log

# Linear discirminant
# NORM training data quantities not modified: Accuracy = 0.623192
# NORM training data quantities modified to 2622: Accuracy = 0.62292


# Modify labeling with original definition
# NORM training data quantities not modified: Accuracy = 0.623465


prediction = np.zeros((6000,2))
prediction[:,0] = np.arange(0,6000,1)
prediction[:,1] = classfication 
ML14_prediction = pd.DataFrame(prediction, columns = ['SubjectId','Label'])
ML14_prediction = ML14_prediction.astype(int)
ML14_prediction.to_csv('ML14_prediction_0517_Gaussian_f6.csv', encoding='utf-8', index = False)
