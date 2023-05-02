from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
import math
import numpy as np
import sys
import itertools
import os
import PySimpleGUI as sg


def is_valid_path(filepath):
    if filepath and Path(filepath).exists():
        return True
    sg.popup_error('Filepath is not valid', font=('Courier', 15))
    return False

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

def plot_fluorescence_spectrum(data_file_path_list, plot_color_list, legend_label_list):

    intensity_list = []

    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])

    wavelength = np.loadtxt(data_file_path_list[0], usecols=0) # Read wavelength
    for data_file_path in data_file_path_list: # Read intenisty
        intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    
    min_intensity = 0
    max_intensity = 0
    for i in range(len(intensity_list)):
        if max(intensity_list[i]) > max_intensity:
            max_intensity = max(intensity_list[i])
    for i in range(len(intensity_list)):
        if min(intensity_list[i]) < min_intensity:
            min_intensity = min(intensity_list[i])

    ax.set_xlim(930, 1370)     
    ax.set_ylim(calculate_min_lim(min_intensity), calculate_max_lim(max_intensity))


    for i in range(len(data_file_path_list)):
        ax.plot(wavelength, intensity_list[i], color=plot_color_list[i], linewidth=2.5, antialiased=True)

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Intensity (a.u.)', fontsize=25, labelpad=18)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, right='on', direction='in', pad=15)
    ax.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, right='on', direction='in')

    ax.ticklabel_format(style='plain')

    ax.legend(labels=legend_label_list, loc='best', fontsize=16, fancybox=True, framealpha=0.5)

    #ax.set_aspect(440/((calculate_max_lim(max_intensity)-calculate_min_lim(min_intensity))*1.618))

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    plt.show()

    #fig.savefig('.png', bbox_inches='tight', dpi=150)

def plot_fluorescence_spectrum_normalized(data_file_path_list, plot_color_list, legend_label_list):

    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])
    ax.set_xlim(930, 1370)
    ax.set_ylim(0, 1)

    intensity_list = []

    wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    for data_file_path in data_file_path_list:
        intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    for i in range(len(data_file_path_list)):
        ax.plot(wavelength, normalize(intensity_list[i]), color=plot_color_list[i], linewidth=2.5, antialiased=True)

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Intensity (normalized)', fontsize=25, labelpad=18)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, right='on', direction='in', pad=15)
    ax.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, right='on', direction='in')

    ax.ticklabel_format(style='plain')

    ax.legend(labels=legend_label_list, loc='best', fontsize=16, fancybox=True, framealpha=0.5)

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    plt.show()

def plot_fluorescence_response(data_file_path_list, plot_color_list, legend_label_list):
    
    intensity_list = []
    pct_response = [] #∆F/F
    
    wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    for data_file_path in data_file_path_list:
        intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    min_intensity = 0
    max_intensity = 0
    for i in range(len(intensity_list)):
        if max(intensity_list[i]) > max_intensity:
            max_intensity = max(intensity_list[i])
    for i in range(len(intensity_list)):
        if min(intensity_list[i]) < min_intensity:
            min_intensity = min(intensity_list[i])

    for i in range(len(intensity_list)-1):
        for j in range(len(intensity_list[i])):
            pct_response.append((float(intensity_list[i+1][j])-float(intensity_list[i][j]))/float(intensity_list[i][j]))
    min_response = min(pct_response)
    max_response = max(pct_response)
    
    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax1 = fig.add_axes([0.152, 0.15, 0.735, 0.735*13/(13-1.5)])
    ax1.set_xlim(930, 1370)
    ax1.set_ylim(calculate_min_lim(min_intensity), calculate_max_lim(max_intensity))
    ax2 = ax1.twinx()
    ax2.set_ylim(calculate_min_lim(min_response), calculate_max_lim(max_response))

    lines = []

    for i in range(len(intensity_list)):
        line1 = ax1.plot(wavelength, intensity_list[i], plot_color_list[i], linewidth=2.5, antialiased=True, label=legend_label_list[i])
        lines += line1  

    ax1.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=18)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=25, labelpad=20)
    
    ax1.minorticks_on()
    ax1.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax1.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax1.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, direction='in', pad=15)
    ax1.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, direction='in')

    ax1.ticklabel_format(style='plain')

    line2 = ax2.plot(wavelength, pct_response,\
                       'lime', linewidth=2.5, antialiased=True, label='∆F/F$_{0}$')
    lines += line2
    ax2.set_ylabel('∆F/F$_{0}$', fontsize=25, labelpad=20)
    
    ax2.minorticks_on()
    ax2.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax2.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    for i in ['right', 'left', 'top', 'bottom']:
        ax1.spines[i].set_linewidth(2.5)
        
    labs = []
    for i in range(len(lines)):
        labs.append(lines[i].get_label())

    ax1.legend(handles=lines, labels=labs, loc='best', fontsize=16, fancybox=True, framealpha=0.5)
    
    plt.show()
   
    

def main():

    available_color = ['red', 'orange', 'gold', 'limegreen', 'dodgerblue', 'blue', 'darkviolet', 'fuchsia', 'lime', 'darkorange', 'black']
    #headings = ['COLOR', 'LEGEND'] # Headings name

    #-----GUI Definition-----#
    sg.theme('LightBrown3')

    layout1 = [
        [sg.Text('Input File:', font='Courier 20', justification='r'), sg.Input(key='-FILES-', font='Courier 20'), sg.FilesBrowse(font='Courier 20', size=(6,1))],
        [sg.Push(), sg.Exit(button_color='tomato', font='Courier 20', size=(6,1)), sg.OK(font='Courier 20', size=(6, 1))]
    ]

    window1 = sg.Window('Select file(s).', layout1, resizable=True)

    while True:

        data_file_path_list = [] # Create an empty list to hold data files provided
        plot_color_list = [] # Create an empty list to hold colors provided
        legend_label_list = [] # Create an empty list to hold legend names provided

        event1, data_file_path_all = window1.read()

        if event1 in (sg.WIN_CLOSED, 'Exit'):
            exit()

        data_file_path_list = data_file_path_all['-FILES-'].split(';')
        num_files = len(data_file_path_list)

        # Unpack the list of lists, where each inner list represents a row in a GUI 
        # os.path.basename() returns the final component of a pathname
        # os.path.normpath() simplifies the path by removing any double slashes and replacing any backslashes with forward slashes
        layout2 = [
        [sg.Push(), sg.Text('-------------------------Supprots LaTeX math codes for labels.-------------------------', font='Courier 20', justification='c'), sg.Push()],
        *[
        [sg.Text('File: {}'.format(os.path.basename(os.path.normpath(file_path))), font='Courier 20', size=(40,1)), 
        sg.InputCombo(values=available_color, default_value='black', font='Courier 20', size=(13,1)),
        sg.InputText('Enter Label', font='Courier 20', size=(25,1))] for file_path in data_file_path_list],
        [
        sg.Push(),
        sg.Button('Plot Basic', font='Courier 20'),
        sg.Button('Plot Normalized', font='Courier 20'),
        sg.Button('Plot ∆F/F', font='Courier 20'),
        sg.Button('Reset Color', font='Courier 20'),
        sg.Button('Reset Label', font='Courier 20'),
        sg.Push(),
        sg.Exit(button_color='tomato', font='Courier 20')
        ]
        ]

        window2 = sg.Window('Plotting Fluorescence Spectra', layout2, resizable=True)

        while True:

            event2, value = window2.read()
 
            for i in range(2*num_files):
                if i % 2 == 0:
                    plot_color_list.append(value[i])
                else:
                    legend_label_list.append(value[i])

            if event2 in (sg.WIN_CLOSED, 'Exit'):
                break
            elif event2 == 'Plot Basic':
                plot_fluorescence_spectrum(data_file_path_list, plot_color_list, legend_label_list)
            elif event2 == 'Plot Normalized':
                plot_fluorescence_spectrum_normalized(data_file_path_list, plot_color_list, legend_label_list)
            elif event2 == 'Plot ∆F/F':
                plot_fluorescence_response(data_file_path_list, plot_color_list, legend_label_list)
            elif event2 == 'Reset Color':
                for i in range(0, 2*num_files, 2):
                    value.pop(i)
                    plot_color_list = []
                    window2[i].Update(value='Select a Color')
            elif event2 == 'Reset Label':
                for i in range(1, 2*num_files, 2):
                    value.pop(i)
                    legend_label_list = []
                    window2[i].Update('Enter Label')

        window2.close()

        window1['-FILES-'].Update('')
        #window1['Browse'].Update('')

if __name__ == '__main__':
    main()
