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

def plot_fluorescence_spectrum(data_file_path_list):

	
    fig = plt.figure(figsize=(13, 13/1.618))
    ax = fig.add_axes([0.18, 0.18, 0.815, 0.815])

    ax.set_xlim(930, 1370)

    min_intensity = 0
    max_intensity = 0

    intensity_list = []

    wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    for data_file_path in data_file_path_list:
    	intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    for i in range(len(data_file_path_list)):
    	ax.plot(wavelength, intensity_list[i], linewidth=2.5, antialiased=True)
    

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Intensity (a.u.)', fontsize=25, labelpad=20)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, right='on', direction='in', pad=15)
    ax.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, right='on', direction='in')

    ax.ticklabel_format(style='plain')

    #ax.set_aspect(440/((calculate_max_lim(max_intensity)-calculate_min_lim(min_intensity))*1.618))

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    plt.show()

    #fig.savefig('1244.png', bbox_inches='tight', dpi=150)
    
    

def main():
	data_file_path_list = [] # Create an empty list to hold data files provide
	LegendLabelList = []
	#-----GUI Definition-----#
	sg.theme('LightBrown3')

	layout = [
		[sg.Text('Input File:', font=('Courier', 20)), sg.Input(key='-FILES-', font=('Courier', 20)), sg.FilesBrowse(font=('Courier', 20))],
		[sg.Exit(button_color='tomato', font=('Courier', 20)), sg.Button('Plot', font=('Courier', 20))]
	]

	window = sg.Window('Plotting Fluorescence Spectra', layout)

	while True:
		event, value = window.read()
		#print(event, value)
		if event in (sg.WIN_CLOSED, 'Exit'):
			break
		if event == 'Plot':
			#if is_valid_path(value['-FILES-']):
			data_file_path_list = value['-FILES-'].split(';')
			num_files = len(data_file_path_list)
			print(data_file_path_list)
			plot_fluorescence_spectrum(data_file_path_list)
			

			#plot_fluorescence_spectrum(data_file_path)
				#print(sys.argv[:])
	window.close()

if __name__ == '__main__':
	main()
