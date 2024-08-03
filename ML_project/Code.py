# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:52:18 2023

@author: User
"""

#%%   Low pass filter
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
x = np.load('ML_Train.npy')

#%%
from scipy.signal import find_peaks

#====================================================================
# define some useful funciton

# find R peak
def find_r_peak(data,distance=280):
    # default distance = 300
    peaks, _ = find_peaks(data,distance=distance)
    return peaks, _

# find all troughs in data, the prerequisite for finding Q,S,...
def find_reverse(data,distance=20):
    # default distance = 20
    troughs,_= find_peaks(-filtedData,distance=distance)
    return troughs,_

# cut head and tail of the data
def cut_head_tail(data,peak):
    pass

# calculate R-R distance
def rr_distance(peaks,num):
    diffs = np.diff(time[peaks])
    avg_diff = np.round(np.mean(diffs),4)
    avg_rr_dis.append(avg_diff)
    avg_peak_height = filtedData[peaks].mean()
    avg_peak.append(avg_peak_height)
    num += 1 # for i th for loop
    #print(f'{num}_th Lead: average RR distance is {avg_diff}')
    pass

# random sampling for each type 
def rand_sample(num,type_num):
    # num is number of sampling 
    """
    type_normal = x[0:7494]     #type_num = 0
    type_sttc   = x[7494:9514]
    type_cd     = x[9514:11021]
    type_mi     = x[11021:]     #type_num = 3
    """
    if type_num == 0:
        return np.random.randint(0,7494,num)
    elif type_num == 1:
        return np.random.randint(7494,9514,num)
    elif type_num == 2:
        return np.random.randint(9514,11021,num)
    elif type_num == 3:
        return np.random.randint(11021,12209,num)
    else:
        print("Wrong input")
#====================================================================

"""
Data:
    0 -  7493: normal
 7494 -  9513: STTC
 9514 - 11020: CD 
11021 - 12208: MI

"""

time  = np.arange(0,10,0.002)     # 10 sec
index = np.random.randint(0,12208)
index = 100 #9561 #7480 #9560
#indexx= rand_sample(2020,1)
#indexx = np.arange(11021,12209)

print(f"This is {index}_th data")
Patient_ID = x[index]              # Patient ID 
Labels = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

avg_peak = []     # for R peak in each lead
avg_rr_dis = []   # for average RR distance in each lead
avg_qrs_int = []  # for average QRS interval in each lead

fig,ax = plt.subplots(12,1,figsize = (15,15))       # 12 subplots
"""
for k in range(len(indexx)):
    Patient_ID = x[indexx[k]] 
    print(f"This is {indexx[k]}_th data")
"""    
for k in range(1):  
    for i in range(0,12):
        Leads = Patient_ID[i,:]            # Lead number
        # Filter requirements.
        Wn = 50         # desired cutoff frequency of the filter, Hz
        b, a = signal.butter(4, Wn, 'lowpass',fs = 500)
        filtedData = signal.filtfilt(b, a, Leads )
        
        
        #R_peaks,_ = signal.find_peaks(filtedData,height = 0.3)        # R wave peaks
        #peaks, _ = find_peaks( filtedData,distance=300)
        peaks, _ = find_r_peak(filtedData) ### use function

        ########
        # 找出左右相鄰元素大於當前元素的索引
        mask = (filtedData[1:-1] > filtedData[:-2]) & (filtedData[1:-1] > filtedData[2:])
        indices = np.arange(1, filtedData.size-1)[mask]
        # 找出符合條件的元素
        result = filtedData[indices]
        #print(result)
        k = 50
        max_k = np.partition(result, -k)[-k:] #找出前k個高的peak
        # 查找是否存在
        mask = np.in1d(filtedData, max_k)
        # 找到位置
        indexes = np.where(mask)[0]
        
        # 設置條件移除過於近鄰的假峰值點
        #但假峰值點的成立條件: 在鄰近的峰值點中並非最大值
        # mask = diffs > 30 & max in the local region are those we want !
        # 以下的mask2需要修正成更general的版本
        diffs = np.diff(indexes)
        mask2 = diffs > 30
        indices2 = np.arange(1, indexes.size)[mask2]
        result2 = indexes[indices2]
        # 畫出這些峰值
        #ax[i].plot(time[indexes],filtedData[indexes],"+",lw=20,c='k')
        ax[i].plot(time[result2],filtedData[result2],"*",lw=60,c='r')
        ########
        
        
        peaks = peaks[1:-2] # 去掉頭一筆尾兩筆資料
        rr_distance(peaks,i)
        
        #for finding Q,S,J... points
        #troughs,_= find_peaks(-filtedData,distance=20) 
        troughs,_ = find_reverse(filtedData,20)
        
        #combine all the extrema
        points = np.concatenate((troughs,peaks))
        points = np.sort(points)
        
        # a set of Q point candidates
        mask = np.isin(points, peaks)
        q_index_c = np.argwhere(mask).flatten() - 1 #Q point is in front of R
        q_points_c = points[q_index_c]
        
        # a set of S point candidates
        s_index_c = np.argwhere(mask).flatten() + 1 #S point is in back of R
        s_points_c = points[s_index_c]
        
        currentlabel = Labels[i]
        
        # QRS interval
        qrs_interval = time[s_points_c] - time[q_points_c]
        avg_qrs_interval = qrs_interval.mean()
        avg_qrs_int.append(avg_qrs_interval)
        
        # find T peaks #need to improve !!!
        filtedData_left = np.setdiff1d(filtedData, filtedData[peaks])
        time_left = np.setdiff1d(time, time[peaks])
        peaks2, _2 = find_peaks(filtedData_left)
        
        
    #"""    
        ax[i].plot(time,filtedData,label = currentlabel)
        ax[i].legend(loc = 'right')
        """
        ax[i].plot(time[peaks], filtedData[peaks], "x")
        ax[i].plot(time[q_points_c], filtedData[q_points_c], "*",lw=10,c='g')
        ax[i].plot(time[s_points_c], filtedData[s_points_c], "+",lw=10,c='b')
        ax[i].plot(time_left[peaks2],filtedData_left[peaks2],"x",lw=10,c='r')
        """
    plt.show()
    #"""

avg1 = round(sum(avg_rr_dis) / len(avg_rr_dis),4)
avg_peak_rounded = [round(num, 4) for num in avg_peak]
avg2 = round(sum(avg_peak_rounded) / len(avg_peak_rounded),4)
avg_qrs_int_rounded = [round(num, 4) for num in avg_qrs_int]
avg3 = round(sum(avg_qrs_int_rounded) / len(avg_qrs_int_rounded),4)

print("==============================================================")
print("average RR distance in each lead:")
print(avg_rr_dis)
print("--------------------------------------------------------------")
print(f"average RR distance is {avg1}")
print("==============================================================")
print("average R peak height in each lead:")
print(avg_peak_rounded)
print("--------------------------------------------------------------")
print(f"average R peak height is {avg2}")
print("==============================================================")
print("average QRS interval is:")
print(avg_qrs_int_rounded)
print("--------------------------------------------------------------")
print(f"average QRS interval is {avg3}")
print("==============================================================")


#plotting.

#plt.plot(x0,y1)
#plt.plot(x0[R_peaks],filtedData[R_peaks],'x')

#%% 
"""  
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 生成數據
x = np.linspace(0, 6*np.pi, 5000)
y = np.sin(x) + np.random.rand(5000)*0.5

# 繪製數據
plt.plot(x, y)

# 找到峰值點
peaks, _ = find_peaks(y, distance=500)

# 在圖中標示峰值點
plt.plot(x[peaks], y[peaks], "x")

# 顯示圖形
plt.show()
"""


#%%
#QRS peaks detector using Pan Tompkins method

#%% QRS detection (lowpass + der + sqr + moving_win)
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

y = np.load('ML_Train.npy')
Patient_ID = y[0]                                   # Patient ID 
Patient_12_Leads = np.zeros((5000,13))
Patient_12_Leads[:,0] = np.arange(0,10,0.002)       # 10 sec, stored in column 0

Labels = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']


for i in range(0,12):
    Leads = Patient_ID[i,:]            # Lead number
    # Filter requirements.
    lowcut = 20                        # desired cutoff frequency of the filter, Hz
    fs = 500                           # sampling rate, Hz
    b, a = signal.butter(4, lowcut,'lowpass',fs = 500)
    Patient_12_Leads[:,i+1] = signal.filtfilt(b, a, Leads )

def derivative(data):
    '''
    Derivative Filter 
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The derivative of the input signal is taken to obtain the
    information of the slope of the signal. Thus, the rate of change
    of input is obtain in this step of the algorithm.

    The derivative filter has the recursive equation:
      y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)
    '''

    # Initialize result
    

    result = data.copy()
    T = 0.002
    # Apply the derivative filter using the equation given
    for i in range(1,13):
         for index in range(len(data)):
             result[index,i] = 0
             if (index >= 1):
                 result[index,i] -= 2*data[index-1,i]
             if (index >= 2):
                 result[index] -= data[index-2,i]
             if (index >= 2 and index <= len(data)-2):
                 result[index] += 2*data[index+1,i]
             if (index >= 2 and index <= len(data)-3):
                 result[index,i] += data[index+2,i]
             result[index,i] = (result[index,i])/(8*T)
    return result  

global der
der = derivative(Patient_12_Leads.copy())

def squaring(data):
    '''
    Squaring the Signal
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The squaring process is used to intensify the slope of the
    frequency response curve obtained in the derivative step. This
    step helps in restricting false positives which may be caused
    by T waves in the input signal.

    The squaring filter has the recursive equation:
      y(nT) = [x(nT)]^2
    '''

    # Initialize result
    result = data.copy()

    # Apply the squaring using the equation given
    for i in range(1,13):
        for index in range(len(data)):
            result[index,i] = data[index,i]**2

    return result
 

global sqr
sqr = squaring(der.copy())
    
def moving_window_integration(data):
    '''
    Moving Window Integrator
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The moving window integration process is done to obtain
    information about both the slope and width of the QRS complex.
    A window size of 0.15*(sample frequency) is used for more
    accurate results.

    The moving window integration has the recursive equation:
      y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N

      where N is the number of samples in the width of integration
      window.
    '''

    # Initialize result and window size for integration
    result = data.copy()
    win_size = round(0.025 * fs)
    sum = 0

    # Calculate the sum for the first N terms
    for i in range(1,13):
        for j in range(win_size):
            sum += data[j,i] / win_size
            result[j,i] = sum
    
    # Apply the moving window integration using the equation given
    for k in range(1,13):
        for index in range(win_size,len(data)):
            sum += data[index] / win_size
            sum -= data[index-win_size] / win_size
            result[index] = sum    
    
    return result  

global moving_win
moving_win = moving_window_integration(sqr.copy())




Labels = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
fig,ax = plt.subplots(12,1,figsize = (15,20))       # 12 subplots
for i in range(0,12):
    ax[i].plot(Patient_12_Leads[:,0], moving_win[:,i+1] , label = Labels[i])
    ax[i].legend(loc = 'right')


"""