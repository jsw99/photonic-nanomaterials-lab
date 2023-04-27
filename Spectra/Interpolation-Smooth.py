#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import glob


# In[ ]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def sliding_window(arr, k):
    for i in range(len(arr)-k+1):
        yield arr[i:i+k]


# In[ ]:


wavelength = np.loadtxt('930~1367_640_wavelength.txt', usecols=0)


# In[ ]:


# Files to be processed
fluorescence_intensity_file_list = glob.glob('*.txt')


# In[ ]:


for fluorescence_intensity_file in fluorescence_intensity_file_list:
    
    data = 0
    
    flag = 0
    
    if len(np.loadtxt(fluorescence_intensity_file, usecols=(0))) == 512:
        ### INTERPOLATION to 640 intensity ###
        wavelength_512 = np.loadtxt(fluorescence_intensity_file, usecols=(0))
        intensity_512 = np.loadtxt(fluorescence_intensity_file, usecols=(2))

        f = interpolate.interp1d(wavelength_512, intensity_512)
        intensity_640 = f(wavelength)
        
        ### Fitting condition ###
        for i, j in zip(sliding_window(wavelength, 5), sliding_window(intensity_640, 5)):
            if (((max(j)-min(j))/intensity_640.max() >= 0.1) and (np.gradient(j)>0).all() != True) and\
                ((np.gradient(j)>0).any() == True):
                    flag = 1
                    #print(y_train_file, i, '1')
                    
        if flag == 1:
            cubic_interpolation_model = interpolate.interp1d(wavelength, intensity_640, kind='cubic')

            wavelength_ = np.linspace(wavelength.min(), wavelength.max(), 54)

            intensity_CIM_54 = cubic_interpolation_model(wavelength_)

            f = interpolate.interp1d(wavelength_, intensity_CIM_54)

            intensity_CIM_640 = f(wavelength)

            data = np.stack((wavelength, intensity_640, intensity_CIM_640), axis=-1)
            
            np.savetxt('{}.txt'.format(fluorescence_intensity_file.rsplit('.', 1)[0]+'_new'), data)
        
        else: #flag==0
            data = np.stack((wavelength, np.zeros(640), intensity_640), axis=-1)
            
            np.savetxt('{}.txt'.format(fluorescence_intensity_file.rsplit('.', 1)[0]+'_new'), data)
            
    else: #number of intensity == 640
        intensity = np.loadtxt(fluorescence_intensity_file, usecols=(2))
        
        ### Fitting condition ###
        for i, j in zip(sliding_window(wavelength, 5), sliding_window(intensity, 5)):
            if (((max(j)-min(j))/intensity.max() >= 0.1) and (np.gradient(j)>0).all() != True) and\
                ((np.gradient(j)>0).any() == True):
                    flag = 1
                    #print(y_train_file, i, '3')
        
        if flag == 1:
            
            cubic_interpolation_model = interpolate.interp1d(wavelength, intensity, kind='cubic')

            wavelength_ = np.linspace(wavelength.min(), wavelength.max(), 54)

            intensity_CIM_54 = cubic_interpolation_model(wavelength_)

            f = interpolate.interp1d(wavelength_, intensity_CIM_54)

            intensity_CIM_640 = f(wavelength)

            data = np.stack((wavelength, intensity, intensity_CIM_640), axis=-1)
            
            np.savetxt('{}.txt'.format(fluorescence_intensity_file.rsplit('.', 1)[0]+'_new'), data)
        
        else: #flag==0
            data = np.stack((wavelength, np.zeros(640), intensity), axis=-1)
            np.savetxt('{}.txt'.format(fluorescence_intensity_file.rsplit('.', 1)[0]+'_new'), data)

