#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
import math


# In[ ]:


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[ ]:


def calculate_max_lim(max_intensity):
    maximum = max_intensity
    if maximum >= 10:
        return math.ceil(maximum*1.15)
    elif maximum >= 1:
        return math.ceil(maximum*1.15*10) / 10
    elif maximum >= 0.1:
        return math.ceil(maximum*1.15*100) / 100
    elif maximum >= 0.01:
        return math.ceil(maximum*1.15*1000) / 1000 
    elif maximum >= 0.001:
        return math.ceil(maximum*1.15*10000) / 10000
    elif maximum >= 0.0001:
        return math.ceil(maximum*1.15*100000) / 100000
    elif maximum >= 0.00001:
        return math.ceil(maximum*1.15*1000000) / 1000000
    else:
        return math.ceil(maximum*1.15*10000000) / 10000000

def calculate_min_lim(min_intensity):
    minimum = min_intensity
    if minimum >= 0:
        minimum = 0
    elif minimum <= -10:
        return math.floor(minimum*1.15*1) / 1
    elif minimum <= -1:
        return math.floor(minimum*1.15*10) / 10
    elif minimum <= -0.1:
        return math.floor(minimum*1.15*100) / 100
    elif minimum <= -0.01:
        return math.floor(minimum*1.15*1000) / 1000
    elif minimum <= -0.001:
        return math.floor(minimum*1.15*10000) / 10000
    elif minimum <= -0.0001:
        return math.floor(minimum*1.15*100000) / 100000
    elif minimum <= -0.00001:
        return math.floor(minimum*1.15*1000000) / 1000000
    else:
        return math.floor(minimum*1.15*10000000) / 10000000


# In[ ]:


def read_files_analyte():
    
    fileNum = 2
    intensityList = []
    intensityListRaw = []
    nameList = []
    colorList = ['red', 'blue']
    fileNameList = []
    
    for i in range(fileNum):
        
        fileName = input('file name? ')
        fileNameList.append(fileName)
        
        if i == 0:
            name = input('curve\'s name? ')
            nameList.append(name)
            wavelength = np.loadtxt(fileName, usecols=(0))
            
        intensity = np.loadtxt(fileName, usecols=(2))
        intensity_raw = np.loadtxt(fileName, usecols=(1))
        
        if intensity.min() < 0:
            if np.array_equal(intensity_raw, np.zeros(640)) == True:
                intensityList.append(intensity)
                intensityListRaw.append(intensity)
            else:
                intensityList.append(intensity_raw)
                intensityListRaw.append(intensity_raw)
        else: 
            intensityList.append(intensity)
            if np.array_equal(intensity_raw, np.zeros(640)) == False:
                intensityListRaw.append(intensity_raw)
            else:
                intensityListRaw.append(intensity)
    
    nameList.append('+100µM Cocaine')
    
    return [wavelength, intensityList, nameList, colorList, fileNameList, intensityListRaw]


# In[ ]:


def read_files_general():
    
    fileNum = int(input('How many files? '))
    intensityList = []
    intensityListRaw = []
    nameList = []
    colorList = []
    fileNameList = []
    
    for i in range(fileNum):
        
        fileName = input('file name? ')
        nameAndColor = input('curve\'s name and color? (name/color) ')
        fileNameList.append(fileName)
        nameList.append(nameAndColor.rsplit('/',1)[0])
        colorList.append(nameAndColor.rsplit('/',1)[1])
        
        if i == 0:
            wavelength = np.loadtxt(fileName, usecols=(0))
            
        intensity = np.loadtxt(fileName, usecols=(2))
        intensity_raw = np.loadtxt(fileName, usecols=(1))
        
        if intensity.min() < 0:
            if np.array_equal(intensity_raw, np.zeros(640)) == True:
                intensityList.append(intensity)
                intensityListRaw.append(intensity)
            else:
                intensityList.append(intensity_raw)
                intensityListRaw.append(intensity_raw)
        else: 
            intensityList.append(intensity)
            if np.array_equal(intensity_raw, np.zeros(640)) == False:
                intensityListRaw.append(intensity_raw)
            else:
                intensityListRaw.append(intensity)    
    
    return [wavelength, intensityList, nameList, colorList, fileNameList, intensityListRaw]


# In[ ]:


def plot_fluorescence_spectrum(wavelength, intensityList, nameList, colorList, fileNameList, intensityListRaw):
    
    flag = 0
    
    for k in range(len(intensityList)):
        if (intensityList[k] == intensityListRaw[k]).all() == True:
            flag += 1
    
    if flag == len(intensityList):
            
        fig = plt.figure(figsize=(25, 25/1.618))
        ax = fig.add_axes([0, 0, 1, 1])

        ax.set_xlim(930, 1370)

        min_intensity = 0
        max_intensity = 0
        for i in range(len(intensityList)):
            if max(intensityList[i]) > max_intensity:
                max_intensity = max(intensityList[i])
        for i in range(len(intensityList)):
            if min(intensityList[i]) < min_intensity:
                min_intensity = min(intensityList[i])      
        ax.set_ylim(calculate_min_lim(min_intensity), calculate_max_lim(max_intensity))

        lineColor = ['black', 'red', 'blue', 'lime', 'fuchsia', 'cyan', 'darkorange', 'blueviolet']

        for i in range(len(nameList)):
            ax.plot(wavelength, intensityList[i], colorList[i], linewidth=4, antialiased=True)
        ax.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
        ax.set_ylabel('Intensity (a.u.)', fontsize=60, labelpad=40)

        ax.minorticks_on()
        ax.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
        ax.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

        ax.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, right='on', direction='in', pad=20)
        ax.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, right='on', direction='in')

        ax.ticklabel_format(style='plain')

        #ax.set_aspect(440/((calculate_max_lim(max_intensity)-calculate_min_lim(min_intensity))*1.618))

        for i in ['right', 'left', 'top', 'bottom']:
            ax.spines[i].set_linewidth(4)

        ax.legend(labels=nameList, loc='best', fontsize=35, shadow=True)

        plt.show()

        fig.savefig('{}_1.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')
        
    else:
        
        fig1 = plt.figure(figsize=(25, 25/1.618))
        ax1 = fig1.add_axes([0, 0, 1, 1])
        
        fig2 = plt.figure(figsize=(25, 25/1.618))
        ax2 = fig2.add_axes([0, 0, 1, 1])

        ax1.set_xlim(930, 1370)
        ax2.set_xlim(930, 1370)

        min_intensity1 = 0
        max_intensity1 = 0
        for i in range(len(intensityList)):
            if max(intensityList[i]) > max_intensity1:
                max_intensity1 = max(intensityList[i])
        for i in range(len(intensityList)):
            if min(intensityList[i]) < min_intensity1:
                min_intensity1 = min(intensityList[i])
                
        min_intensity2 = 0
        max_intensity2 = 0
        for i in range(len(intensityListRaw)):
            if max(intensityListRaw[i]) > max_intensity2:
                max_intensity2 = max(intensityListRaw[i])
        for i in range(len(intensityListRaw)):
            if min(intensityListRaw[i]) < min_intensity2:
                min_intensity2 = min(intensityListRaw[i])
                
        ax1.set_ylim(calculate_min_lim(min_intensity1), calculate_max_lim(max_intensity1))
        ax2.set_ylim(calculate_min_lim(min_intensity2), calculate_max_lim(max_intensity2))

        lineColor = ['black', 'red', 'blue', 'lime', 'fuchsia', 'cyan', 'darkorange', 'blueviolet']

        for i in range(len(nameList)):
            ax1.plot(wavelength, intensityList[i], colorList[i], linewidth=4, antialiased=True)
            ax2.plot(wavelength, intensityListRaw[i], colorList[i], linewidth=4, antialiased=True)
        ax1.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=60, labelpad=40)
        ax2.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
        ax2.set_ylabel('Intensity (a.u.)', fontsize=60, labelpad=40)

        ax1.minorticks_on()
        ax1.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
        ax1.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

        ax1.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, right='on', direction='in', pad=20)
        ax1.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, right='on', direction='in')

        ax1.ticklabel_format(style='plain')
        
        ax2.minorticks_on()
        ax2.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
        ax2.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

        ax2.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, right='on', direction='in', pad=20)
        ax2.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, right='on', direction='in')

        ax2.ticklabel_format(style='plain')

        #ax.set_aspect(440/((calculate_max_lim(max_intensity)-calculate_min_lim(min_intensity))*1.618))

        for i in ['right', 'left', 'top', 'bottom']:
            ax1.spines[i].set_linewidth(4)
            ax2.spines[i].set_linewidth(4)
        ax1.legend(labels=nameList, loc='best', fontsize=35, shadow=True)
        ax2.legend(labels=nameList, loc='best', fontsize=35, shadow=True)

        plt.show()

        fig1.savefig('{}_1.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')
        fig2.savefig('{}_1_raw.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')


# In[ ]:


def plot_fluorescence_spectrum_normalized(wavelength, intensityList, nameList, colorList, fileNameList, intensityListRaw):
    
    flag = 0
    
    for k in range(len(intensityList)):
        if (intensityList[k] == intensityListRaw[k]).all() == True:
            flag += 1
    
    if flag == len(intensityList):
    
        fig = plt.figure(figsize=(25, 25/1.618))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(930, 1370)
        ax.set_ylim(0, 1)

        lineColor = ['black', 'red', 'blue', 'lime', 'fuchsia', 'cyan', 'darkorange', 'blueviolet']

        for i in range(len(nameList)):
            ax.plot(wavelength, normalize(intensityList[i]), colorList[i], linewidth=4, antialiased=True)
        ax.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
        ax.set_ylabel('Intensity (normalized)', fontsize=60, labelpad=40)

        ax.minorticks_on()
        ax.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
        ax.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

        ax.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, right='on', direction='in', pad=20)
        ax.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, right='on', direction='in')

        ax.ticklabel_format(style='plain')

        #ax.set_aspect(440/1.618)

        for i in ['right', 'left', 'top', 'bottom']:
            ax.spines[i].set_linewidth(4)

        ax.legend(labels=nameList, loc='best', fontsize=35, shadow=True)

        plt.show()

        fig.savefig('{}_2.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')
        
    else:
        
        fig1 = plt.figure(figsize=(25, 25/1.618))
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.set_xlim(930, 1370)
        ax1.set_ylim(0, 1)
        
        fig2 = plt.figure(figsize=(25, 25/1.618))
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.set_xlim(930, 1370)
        ax2.set_ylim(0, 1)

        lineColor = ['black', 'red', 'blue', 'lime', 'fuchsia', 'cyan', 'darkorange', 'blueviolet']

        for i in range(len(nameList)):
            ax1.plot(wavelength, normalize(intensityList[i]), colorList[i], linewidth=4, antialiased=True)
            ax2.plot(wavelength, normalize(intensityListRaw[i]), colorList[i], linewidth=4, antialiased=True)
        ax1.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
        ax1.set_ylabel('Intensity (normalized)', fontsize=60, labelpad=40)
        ax2.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
        ax2.set_ylabel('Intensity (normalized)', fontsize=60, labelpad=40)

        ax1.minorticks_on()
        ax1.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
        ax1.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

        ax1.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, right='on', direction='in', pad=20)
        ax1.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, right='on', direction='in')

        ax1.ticklabel_format(style='plain')
        
        ax2.minorticks_on()
        ax2.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
        ax2.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

        ax2.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, right='on', direction='in', pad=20)
        ax2.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, right='on', direction='in')

        ax2.ticklabel_format(style='plain')

        #ax.set_aspect(440/1.618)

        for i in ['right', 'left', 'top', 'bottom']:
            ax1.spines[i].set_linewidth(4)
            ax2.spines[i].set_linewidth(4)

        ax1.legend(labels=nameList, loc='best', fontsize=35, shadow=True)
        ax2.legend(labels=nameList, loc='best', fontsize=35, shadow=True)

        plt.show()

        fig1.savefig('{}_2.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')
        fig2.savefig('{}_2_raw.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')


# In[ ]:


def plot_fluorescence_response(wavelength, intensityList, nameList, colorList, fileNameList):
    
    pct_response = [] #∆F/F

    for i in range(len(intensityList)-1):
        for j in range(len(intensityList[i])):
            pct_response.append((float(intensityList[i+1][j])-float(intensityList[i][j]))/float(intensityList[i][j]))
            
    min_response = min(pct_response)
    max_response = max(pct_response)
    
    min_intensity = 0
    max_intensity = 0
    for i in range(len(intensityList)):
        if max(intensityList[i]) > max_intensity:
            max_intensity = max(intensityList[i])
    for i in range(len(intensityList)):
        if min(intensityList[i]) < min_intensity:
            min_intensity = min(intensityList[i])
    
    fig = plt.figure(figsize=(25, 25/1.618))
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax1.set_xlim(930, 1370)
    ax1.set_ylim(calculate_min_lim(min_intensity), calculate_max_lim(max_intensity))
    ax2 = ax1.twinx()
    ax2.set_ylim(calculate_min_lim(min_response), calculate_max_lim(max_response))

    lines = []

    for i in range(len(intensityList)):
        line1 = ax1.plot(wavelength, intensityList[i], colorList[i], linewidth=4, antialiased=True, label=nameList[i])
        lines += line1  

    ax1.set_xlabel('Wavelength (nm)', fontsize=60, labelpad=40)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=60, labelpad=40)
    
    ax1.minorticks_on()
    ax1.xaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, top='on', direction='in', pad=20)
    ax1.xaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, top='on', direction='in')

    ax1.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, direction='in', pad=20)
    ax1.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, direction='in')

    ax1.ticklabel_format(style='plain')

    line2 = ax2.plot(wavelength, pct_response,\
                       'lime', linewidth=4, antialiased=True, label='∆F/F$_{0}$')
    lines += line2
    ax2.set_ylabel('∆F/F$_{0}$', fontsize=60, labelpad=40)
    
    ax2.minorticks_on()
    ax2.yaxis.set_tick_params(which='major', labelsize=45, width=4, length=25, direction='in', pad=20)
    ax2.yaxis.set_tick_params(which='minor', labelsize=45, width=4, length=10, direction='in')

    #ax1.set_aspect(440/((calculate_max_lim(max_intensity)-calculate_min_lim(min_intensity))*1.618))
    #ax2.set_aspect(440/((calculate_max_lim(max_response)-calculate_min_lim(min_response))*1.618))

    for i in ['right', 'left', 'top', 'bottom']:
        ax1.spines[i].set_linewidth(4)
        
    labs = []
    for i in range(len(lines)):
        labs.append(lines[i].get_label())

    ax1.legend(handles=lines, labels=labs, loc='best', fontsize=35, shadow=True)
    
    plt.show()
    
    fig.savefig('{}_3.png'.format(fileNameList[0].split('/')[-1].split('_')[0]), bbox_inches='tight')


# In[ ]:


if __name__ == '__main__':
    parameters = read_files_general()
    
    plot_fluorescence_spectrum(*parameters)

    plot_fluorescence_spectrum_normalized(*parameters)

    #plot_fluorescence_response(*parameters[0:5])

