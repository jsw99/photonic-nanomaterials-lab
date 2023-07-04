from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
import math
import numpy as np
import sys
import os
import PySimpleGUI as sg
from scipy import interpolate
from nptdms import TdmsFile

def is_valid_path(data_file_path_list):
    for data_file_path in data_file_path_list:
        if data_file_path and Path(data_file_path).exists():
            return True
        #sg.popup_error('Filepath is not valid', font=('Courier', 15))
        return False

def add_files_in_folder(parent, dirname):
    files = os.listdir(dirname)
    for f in sorted(files):
        fullname = os.path.join(dirname, f)
        if os.path.isdir(fullname):            # if it's a folder, add folder and recurse
            treedata.Insert(parent, fullname, f, values=[], icon=folder_icon)
            add_files_in_folder(fullname, fullname)
        else:
            treedata.Insert(parent, fullname, f, values=[os.stat(fullname).st_size], icon=file_icon)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def interpolate_640(data_file_path_list):
    interpolate_intensity_list = []

    global wavelength
    wavelength = np.loadtxt('930~1367_640_wavelength.txt', usecols=0)

    for data_file_path in data_file_path_list:
        wavelength_512 = np.loadtxt(data_file_path, usecols=(0))
        intensity_512 = np.loadtxt(data_file_path, usecols=(2))
        f = interpolate.interp1d(wavelength_512, intensity_512)
        intensity_640 = f(wavelength)
        interpolate_intensity_list.append(intensity_640)

    return interpolate_intensity_list

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

def on_close(event):
    global should_close
    should_close = True
    

def plot_fluorescence_spectrum(data_file_path_list, plot_color_list, legend_label_list):

    intensity_list = interpolate_640(data_file_path_list)

    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])

    #wavelength = np.loadtxt(data_file_path_list[0], usecols=0) # Read wavelength
    #for data_file_path in data_file_path_list: # Read intenisty
    #    intensity_list.append(np.loadtxt(data_file_path, usecols=2))

    
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

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show(block=False)
    #plt.pause(3)
    #plt.close()
    global should_close

    #fig.savefig('.png', bbox_inches='tight', dpi=150)

def plot_fluorescence_spectrum_normalized(data_file_path_list, plot_color_list, legend_label_list):

    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])
    ax.set_xlim(930, 1370)
    ax.set_ylim(0, 1)

    intensity_list = interpolate_640(data_file_path_list)

    #wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    #for data_file_path in data_file_path_list:
    #    intensity_list.append(np.loadtxt(data_file_path, usecols=2))

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

    #fig.canvas.mpl_connect('close_event', on_close)

    plt.show(block=False)


def plot_fluorescence_response(data_file_path_list, plot_color_list, legend_label_list):
    
    intensity_list = interpolate_640(data_file_path_list)

    pct_response = [] #∆F/F
    
    #wavelength = np.loadtxt(data_file_path_list[0], usecols=0)
    #for data_file_path in data_file_path_list:
    #    intensity_list.append(np.loadtxt(data_file_path, usecols=2))

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
    min_response = min(pct_response[87:])
    max_response = max(pct_response[87:])
    
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
    
    #fig.canvas.mpl_connect('close_event', on_close)

    plt.show(block=False)

def plot_mean_std(data_file_path_list, plot_color_list, legend_label_list):

    global wavelength
    wavelength = np.loadtxt('930~1367_640_wavelength.txt', usecols=0)

    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])

    mean_intensity_512_list = []
    std_512_list = []

    for data_file_path in data_file_path_list:
        #root
        tdms_file = TdmsFile.read(data_file_path)
        #group
        para_group = tdms_file['Parameters']
        process_group = tdms_file['Processed Spectrum']
        raw_group = tdms_file['Raw Data']
        #channel
        wavelength_channel = para_group['Wavelength']
        #data
        wavelength_512 = wavelength_channel[:]

        intensity_list = []

        for channel in process_group.channels():
            intensity_512 = channel[:]
            intensity_list.append(intensity_512)

        intensity_matrix_lambda_row = np.transpose(np.array(intensity_list))

        mean_intensity_512 = np.mean(intensity_matrix_lambda_row, axis=1)

        std_512 = np.std(intensity_matrix_lambda_row, axis=1)

        mean_intensity_512_list.append(mean_intensity_512)
        std_512_list.append(std_512)

    mean_intensity_640_list = []
    std_640_list = []

    for (mean_intensity, std) in zip(mean_intensity_512_list, std_512_list):
        f_mean = interpolate.interp1d(wavelength_512, mean_intensity)
        f_std = interpolate.interp1d(wavelength_512, std)
        mean_intensity_640 = f_mean(wavelength)
        std_640 = f_std(wavelength)
        mean_intensity_640_list.append(mean_intensity_640)
        std_640_list.append(std_640)

    for i in range(len(data_file_path_list)):
        ax.plot(wavelength, mean_intensity_640_list[i], plot_color_list[i], label=legend_label_list[i], linewidth=2.5, antialiased=True)

        ax.fill_between(
            wavelength,
            mean_intensity_640_list[i] - std_640_list[i],
            mean_intensity_640_list[i] + std_640_list[i],
            color="tab:{}".format(plot_color_list[i]),
            alpha=0.2,
            label='$\pm \sigma$ interval'
        )

    min_intensity = 0
    max_intensity = 0
    for i in range(len(mean_intensity_640_list)):
        if max(mean_intensity_640_list[i]+std_640_list[i]) > max_intensity:
            print(mean_intensity_640_list[i]+std_640_list[i])
            max_intensity = max(mean_intensity_640_list[i]+std_640_list[i])
    for i in range(len(mean_intensity_640_list)):
        if min(mean_intensity_640_list[i]-std_640_list[i]) < min_intensity:
            min_intensity = min(mean_intensity_640_list[i]-std_640_list[i])

    print(max_intensity, min_intensity)

    ax.set_xlim(930, 1370)     
    ax.set_ylim(calculate_min_lim(min_intensity), calculate_max_lim(max_intensity))

    ax.set_xlabel('Wavelength (nm)', fontsize=25, labelpad=20)
    ax.set_ylabel('Intensity (a.u.)', fontsize=25, labelpad=18)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, top='on', direction='in', pad=15)
    ax.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, top='on', direction='in')

    ax.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15, right='on', direction='in', pad=15)
    ax.yaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6, right='on', direction='in')

    ax.legend(loc='best', fontsize=15, fancybox=True, framealpha=0.5)

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    plt.show(block=False)


def main():
    global folder_icon, file_icon
    folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
    file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'

    # Available colors in InputCombo
    available_color = ['red', 'orange', 'gold', 'limegreen', 'dodgerblue', 'blue', 'darkviolet', 'fuchsia', 'lime', 'darkorange', 'black']

    #-----GUI Definition-----#
    sg.theme('LightBrown3')

    while True:
        starting_path = sg.popup_get_folder('Select the folder of your name', font='Courier 20')
        if not starting_path:
            sg.popup_error('Filepath is not valid', font='Courier 15')
            sys.exit(0)
        else:
            pass

        global treedata
        treedata = sg.TreeData()

        add_files_in_folder('', starting_path)

        layout0 = [[sg.Text('Select file(s)', font='Courier 15')],
              [sg.Tree(data=treedata,
                       headings=['Size'],
                       auto_size_columns=True,
                       select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
                       num_rows=20,
                       col0_width=30,
                       key='-TREE-',
                       show_expanded=False,
                       enable_events=True,
                       ),],
              [sg.Button('Ok', font='Courier 20'), sg.Button('Cancel', button_color='tomato', font='Courier 20')]]

        window0 = sg.Window('Select file(s)', layout0, resizable=True, finalize=True)
        window0['-TREE-'].expand(True, True)     # Resize with the window


        while True:
            event0, values = window0.read()
            if event0 in (sg.WIN_CLOSED, 'Cancel'):
                window0.close()
                break
            elif event0 == 'Ok':
                data_file_path_list = values['-TREE-']
            else:
                continue

            while True:

                #data_file_path_list = [] # Create an empty list to hold data files provided
                plot_color_list = [] # Create an empty list to hold colors provided
                legend_label_list = [] # Create an empty list to hold legend names provided
                
                num_files = len(data_file_path_list)

                if not is_valid_path(data_file_path_list):
                    sg.popup_error('Filepath is not valid', font='Courier 15')
                    continue
                

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
                sg.Button('Plot Mean & Std', font='Courier 20'),
                sg.Button('Reset Color', font='Courier 20'),
                sg.Button('Reset Label', font='Courier 20'),
                sg.Push(),
                sg.Exit(button_color='tomato', font='Courier 20')
                ]
                ]

                window2 = sg.Window('Plotting Fluorescence Spectra', layout2, resizable=True)

                global should_close # Set up a flag

                should_close = False

                while True:

                    event2, values = window2.read()
                    
                    for i in range(2*num_files):
                        if i % 2 == 0:
                            plot_color_list.append(values[i])
                        else:
                            legend_label_list.append(values[i])

                    if event2 in (sg.WIN_CLOSED, 'Exit'):
                        break
                    elif event2 == 'Plot Basic':
                        plot_fluorescence_spectrum(data_file_path_list, plot_color_list, legend_label_list)
                        if should_close:
                            plt.close()
                            should_close = False # Reset the flag
                    elif event2 == 'Plot Normalized':
                        plot_fluorescence_spectrum_normalized(data_file_path_list, plot_color_list, legend_label_list)
                        #if should_close:
                            #plt.close()
                            #should_close = False # Reset the flag
                    elif event2 == 'Plot ∆F/F':
                        plot_fluorescence_response(data_file_path_list, plot_color_list, legend_label_list)
                        #if should_close:
                            #plt.close()
                            #should_close = False # Reset the flag
                    elif event2 == 'Plot Mean & Std':
                        plot_mean_std(data_file_path_list, plot_color_list, legend_label_list)
                    elif event2 == 'Reset Color':
                        for i in range(0, 2*num_files, 2):
                            values.pop(i)
                            plot_color_list = []
                            window2[i].Update(value='black')
                    elif event2 == 'Reset Label':
                        for i in range(1, 2*num_files, 2):
                            values.pop(i)
                            legend_label_list = []
                            window2[i].Update('Enter Label')

                window2.close()
                break

if __name__ == '__main__':
    main()
