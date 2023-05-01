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

def plot_fluorescence_spectrum(data_file_path_list, plot_color_list, legend_label_list):

	
    fig = plt.figure(figsize=(13, 13/1.618))
    ax = fig.add_axes([0.16, 0.15, 0.835, 0.835])

    ax.set_xlim(930, 1370)

    min_intensity = 0
    max_intensity = 0

    intensity_list = []

    wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    for data_file_path in data_file_path_list:
    	intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    for i in range(len(data_file_path_list)):
    	ax.plot(wavelength, intensity_list[i], color=plot_color_list[i], linewidth=2.5, antialiased=True)
    

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Intensity (a.u.)', fontsize=25, labelpad=20)

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

    #fig.savefig('12445.png', bbox_inches='tight', dpi=150)

def plot_fluorescence_spectrum_normalized(data_file_path_list, plot_color_list, legend_label_list):

    fig = plt.figure(figsize=(13, 13/1.618))
    ax = fig.add_axes([0.16, 0.15, 0.835, 0.835])
    ax.set_xlim(930, 1370)
    ax.set_ylim(0, 1)

    intensity_list = []

    wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    for data_file_path in data_file_path_list:
    	intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    for i in range(len(data_file_path_list)):
    	ax.plot(wavelength, normalize(intensity_list[i]), color=plot_color_list[i], linewidth=2.5, antialiased=True)

    

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Intensity (normalized)', fontsize=25, labelpad=20)

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

    for i in range(len(intensity_list)-1):
        for j in range(len(intensity_list[i])):
            pct_response.append((float(intensity_list[i+1][j])-float(intensity_list[i][j]))/float(intensity_list[i][j]))
            
    min_response = min(pct_response)
    max_response = max(pct_response)
    
    min_intensity = 0
    max_intensity = 0
    for i in range(len(intensity_list)):
        if max(intensity_list[i]) > max_intensity:
            max_intensity = max(intensity_list[i])
    for i in range(len(intensity_list)):
        if min(intensity_list[i]) < min_intensity:
            min_intensity = min(intensity_list[i])
    
    fig = plt.figure(figsize=(13, 13/1.618))
    ax1 = fig.add_axes([0.16, 0.15, 0.835, 0.835])
    ax1.set_xlim(930, 1370)
    #ax1.set_ylim(calculate_min_lim(min_intensity), calculate_max_lim(max_intensity))
    ax2 = ax1.twinx()
    #ax2.set_ylim(calculate_min_lim(min_response), calculate_max_lim(max_response))

    lines = []

    for i in range(len(intensity_list)):
        line1 = ax1.plot(wavelength, intensity_list[i], plot_color_list[i], linewidth=2.5, antialiased=True, label=legend_label_list[i])
        lines += line1  

    ax1.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=25, labelpad=20)
    
    ax1.minorticks_on()
    ax1.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax1.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax1.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax1.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

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

	data_file_path_list = [] # Create an empty list to hold data files provided
	plot_color_list = [] # Create an empty list to hold colors provided
	legend_label_list = [] # Create an empty list to hold legend names provided
	
	available_color = ['red', 'orange', 'gold', 'limegreen', 'dodgerblue', 'blue', 'darkviolet', 'fuchsia', 'lime', 'darkorange', 'black']
	#headings = ['COLOR', 'LEGEND'] # Headings name

	#-----GUI Definition-----#
	sg.theme('LightBrown3')

	layout1 = [
		[sg.Text('Input File:', font='Courier 20'), sg.Input(key='-FILES-', font='Courier 20'), sg.FilesBrowse(font='Courier 20')],
		[sg.OK(font='Courier 20'), sg.Cancel(font='Courier 20')]
	]

	window1 = sg.Window('Select file(s).', layout1)

	event, data_file_path_all = window1.read()

	if event in (sg.WIN_CLOSED, 'Cancel'):
		exit()

	data_file_path_list = data_file_path_all['-FILES-'].split(';')
	num_files = len(data_file_path_list)

	layout2 = [
		#[sg.Text('								')] + [[sg.Text(h, font='Courier 20')] for h in headings],
		# Unpack the list of lists, where each inner list represents a row in a GUI 
		# os.path.basename() returns the final component of a pathname
		# os.path.normpath() simplifies the path by removing any double slashes and replacing any backslashes with forward slashes
		*[[sg.Text('File: {}'.format(os.path.basename(os.path.normpath(file_path))), font='Courier 20'), 
               sg.InputCombo(values=available_color, default_value='black', font='Courier 20'),
               sg.InputText('Enter Legend Label', font='Courier 20'),
              ] for file_path in data_file_path_list
             ],
		[sg.Button('Plot Basic', font='Courier 20'),
			sg.Button('Plot Normalized', font='Courier 20'),
			sg.Button('Plot ∆F/F', font='Courier 20'),
			sg.Button('Clear', font='Courier 20'),
			sg.Exit(button_color='tomato', font='Courier 20')]
	]

	window2 = sg.Window('Plotting Fluorescence Spectra', layout2)

	while True:
		event, value = window2.read()
		for i in range(2*num_files):
			if i % 2 == 0:
				plot_color_list.append(value[i])
			else:
				legend_label_list.append(value[i])

		if event in (sg.WIN_CLOSED, 'Exit'):
			break
		elif event == 'Plot Basic':
			#if is_valid_path(value['-FILES-']):
			plot_fluorescence_spectrum(data_file_path_list, plot_color_list, legend_label_list)
		elif event == 'Plot Normalized':
			plot_fluorescence_spectrum_normalized(data_file_path_list, plot_color_list, legend_label_list)
		elif event == 'Plot ∆F/F':
			plot_fluorescence_response(data_file_path_list, plot_color_list, legend_label_list)
		elif event == 'Clear':
			for i in range(2*num_files):
				window2[i]('')

	window2.close()

if __name__ == '__main__':
	main()
