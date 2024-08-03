# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:34:37 2023

@author: User
"""


#%% Feature labeling  05/21

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from math import exp, sqrt, pi, ceil


Training_set_2 = np.load('ML_Training_set_2.npy')
Vad_set_2 = np.load('ML_Validation_set_2.npy')

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
    

# Training_set

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



# 原始檔案: NORM:0~7493, STTC:7494~9513, CD:9514~11020, MI:10021~12208
# Training_set_2:  NORM:0~6818, STTC:6819~8232, CD:8233~9286, MI:9287~10117
# Reduced_NORM_Training_set_2:  NORM:0~1499, STTC:1500~2913, CD:2914~3967, MI:3968~4798
# Vad_set_2:       NORM:0~674, STTC:675~1280, CD:1281~1733, MI:1734~2090


NORM = training_set_features.iloc[0:6819]
NORM_train = NORM.sample(frac = 0.22, random_state = 905)
Train_set_features = pd.concat([NORM_train, training_set_features.iloc[6819:10118]]).to_numpy()     # Reduced_NORM_Training_set_2
 
# Delete used matrix
Training_set_2 = None
Vad_set_2 = None
Patient_ID_test  = None
Patient_ID_train = None
training_set_features = None
NORM = None
NORM_train = None

# Benchmark: 2min 10sec跑完

#%% PCA  05/21

def PCA(X, n_components):
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)     # Step 1: Standardize the data
    cov_matrix = np.cov(X_std.T)                             # Step 2: Compute the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]

    pca_matrix = eigenvectors[:, :n_components]              # Step 5: Select the principal components

    return pca_matrix


pca_matrix = PCA(Train_set_features,19)

training_pca = np.dot(Train_set_features, pca_matrix)
vad_pca = np.dot(vad_features, pca_matrix)

# Delete used matrix
#Train_set_features = None
#vad_features = None



#%% LDA  05/21

def LDA(X,n_components):
    label_num = [1500,1414,1054,831]
    mean = np.zeros((4,X.shape[1]))
    mean[0] = np.array(np.mean(X[0:1500], axis = 0))        # NORM_mean
    mean[1] = np.array(np.mean(X[1500:2914], axis = 0))     # STTC_mean
    mean[2] = np.array(np.mean(X[2914:3968], axis = 0))     # CD_mean
    mean[3] = np.array(np.mean(X[3968:4799], axis = 0))     # MI_mean
    index = 0
    Sw = np.zeros((X.shape[1],X.shape[1]))                  # Within-class scatter
    for i in range(0,mean.shape[0]):                        # 4 categories
        Si = np.zeros((X.shape[1],X.shape[1]))
                                    
        for k in range(0,label_num[i]):
            temp = (X[index] - mean[i]).reshape((1,X.shape[1]))
            Si += np.dot(temp.T, temp)
            index += 1
        Sw +=  Si * (1/label_num[i])

    Sb = np.zeros((X.shape[1],X.shape[1]))                  # Between-class scatter
    m = np.array(np.mean(X, axis = 0))                      # Total training data mean
    for j in range(0,mean.shape[0]):
        temp = (mean[j]-m).reshape((1,X.shape[1]))
        Sb += np.dot(temp.T, temp) * label_num[j]
    
    matrix = np.dot(np.linalg.inv(Sw), Sb)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]
    
    lda_matrix = eigenvectors[:, :n_components]


    return lda_matrix
    

lda_matrix = LDA(Train_set_features,4)
lda_matrix = lda_matrix.astype(float)

training_lda = np.dot(Train_set_features, lda_matrix)
vad_lda = np.dot(vad_features, lda_matrix)


# Delete used matrix
#Train_set_features = None
#vad_features = None

#%% Naive Bayes Classsifier  05/21

# New splitting
# training set:
# NORM: 1500, STTC: 1414, CD: 1054, MI:831
# vad_set:
# NORM: 675, STTC: 606, CD: 453, MI: 357 

# Normal PDF naive bayes
def NB(train,test,priors):
    mean = np.zeros((4,train.shape[1]))
    mean[0] = np.array(np.mean(train[0:1500], axis = 0))        # NORM_mean
    mean[1] = np.array(np.mean(train[1500:2914], axis = 0))     # STTC_mean
    mean[2] = np.array(np.mean(train[2914:3968], axis = 0))     # CD_mean
    mean[3] = np.array(np.mean(train[3968:4799], axis = 0))     # MI_mean
    
    std = np.zeros((4,train.shape[1]))
    std[0] = np.array(np.std(train[0:1500], axis = 0))          # NORM_std
    std[1] = np.array(np.std(train[1500:2914], axis = 0))       # STTC_std
    std[2] = np.array(np.std(train[2914:3968], axis = 0))       # CD_std
    std[3] = np.array(np.std(train[3968:4799], axis = 0))       # MI_std 


    P = np.ones((len(test),len(mean)))
    
    for index in range(0,len(test)):                # 6000 testing patients
        for i in range(0,len(mean)):                    # 4 categories
            count = 0
            for j in range (0,test.shape[1]):       # 20 features
                den = sqrt(2*pi) * std[i,j]
                N = (-0.5 * (test[index,j] - mean[i,j])**2 ) / (std[i,j]**2)
                p = (1/den) * exp(N)
                count = count + np.log(p)
            P[index,i] = count + np.log(priors[i])
    
    
    # Classify
    classfication = []
    validation =  np.zeros((len(test),1))
    validation[675:1281] = 2
    validation[1281:1734] = 3
    validation[1734:2091] = 1

    count = 0
    for index in range(0,len(test)):
        temp = np.argmax(P[index])
        if temp == 0:
            classfication.append(0)
        elif temp == 1:
            classfication.append(2)
        elif temp == 2:
            classfication.append(3)
        else:
            classfication.append(1)

        if validation[index] == classfication[index]:
            count += 1
    
    print("Accuracy =",round((count/len(test)),6))
    
    return P



# Train_vad_set_2
#priors = np.array([6819/10118,831/10118,1414/10118,1054/10118])    # Orginal training_set
priors = np.array([1500/4799,1414/4799,1054/4799,831/4799])         # Reduced NORM training_set

P_NB =  NB(training_pca,vad_pca,priors)
P_NB =  NB(training_lda,vad_lda,priors)


# using new labeling function:
# NORM training data quantities modified to 1500: 
#   1. Use PCA: Accuracy = 0.446676
#   2. Use LDA: Accuracy = 0.602104

    
#%% Gaussian Multivariate Classifier  05/21

# New splitting
# training set:
# NORM: 1500, STTC: 1414, CD: 1054, MI:831
# vad_set:
# NORM: 675, STTC: 606, CD: 453, MI: 357 


# Normal PDF
def Gaussian(train,test,priors):
    mean = np.zeros((4,train.shape[1]))
    mean[0] = np.array(np.mean(train[0:1500], axis = 0))        # NORM_mean
    mean[1] = np.array(np.mean(train[1500:2914], axis = 0))     # STTC_mean
    mean[2] = np.array(np.mean(train[2914:3968], axis = 0))     # CD_mean
    mean[3] = np.array(np.mean(train[3968:4799], axis = 0))     # MI_mean
    
    std = np.zeros((4,train.shape[1]))
    std[0] = np.array(np.std(train[0:1500], axis = 0))          # NORM_std
    std[1] = np.array(np.std(train[1500:2914], axis = 0))       # STTC_std
    std[2] = np.array(np.std(train[2914:3968], axis = 0))       # CD_std
    std[3] = np.array(np.std(train[3968:4799], axis = 0))       # MI_std 
    
    cov = []
    cov.append(np.cov(train[0:1500].T))
    cov.append(np.cov(train[1500:2914].T))
    cov.append(np.cov(train[2914:3968].T))
    cov.append(np.cov(train[3968:4799].T))
    
    P = np.ones((len(test),len(mean)))
    
    for index in range(0,len(test)):                # 6000 testing patients
        for i in range(0,len(mean)):                # 4 categories
            den = (2*pi)**(test.shape[1]/2) * np.linalg.det(cov[i])
            temp = (test[index] - mean[i]).reshape(train.shape[1],1)
            N = -0.5 * np.dot( np.dot(temp.T,np.linalg.inv(cov[i])), temp)
            p = (1/den) * exp(N)
            P[index,i] = np.log(p) + np.log(priors[i])
            #P[index,i] = p + np.log(priors[i])    # Linear discirminant
    
    classfication = []
    validation =  np.zeros((len(test),1))
    validation[675:1281] = 2
    validation[1281:1734] = 3
    validation[1734:2091] = 1

    count = 0
    for index in range(0,len(test)):
        temp = np.argmax(P[index])
        if temp == 0:
            classfication.append(0)
        elif temp == 1:
            classfication.append(2)
        elif temp == 2:
            classfication.append(3)
        else:
            classfication.append(1)

        if validation[index] == classfication[index]:
            count += 1
    
    print("Accuracy =",round((count/len(test)),6))
    
    return P  


# Train_vad_set_2
#priors = np.array([6819/10118,831/10118,1414/10118,1054/10118])    # Orginal training_set
priors = np.array([1500/4799,1414/4799,1054/4799,831/4799])         # Reduced NORM training_set

P_Gaussian =  Gaussian(training_pca,vad_pca,priors)
P_Gaussian =  Gaussian(training_lda,vad_lda,priors)


# Using train_vad_set_2
# NORM training data quantities not modified: Accuracy = 0.474892
# NORM training data quantities modified to 2045: Accuracy = 0.492109

# using new labeling function:
# NORM training data quantities modified to 2045: Accuracy = 0.499761
# NORM training data quantities modified to 1500: 
#   1. Use PCA: Accuracy = 0.50263
#   2. Use LDA: Accuracy = 0.553324

