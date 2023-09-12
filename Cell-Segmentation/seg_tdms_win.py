import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
from matplotlib import colors
import spe2py as spe
from cellpose import models
import cellpose
from nptdms import TdmsFile
import time
from skimage import draw
import os
import glob
import pandas as pd
from tkinter.filedialog import askopenfilename
import math
import PySimpleGUI as sg
import sys
from scipy import ndimage

def generate_slope_mat_from_tdms(tdms_file_path):
    #tdms_file = TdmsFile.read(r'{}.tdms'.format(tdms_file_path))
    tdms_file = TdmsFile.read(r'{}'.format(tdms_file_path))
    all_groups: list = tdms_file.groups()

    slope_matrix_b_list: list = []
    slope_matrix_f_list: list = []
    group_num_list: list = []

    for group_num in range(int(len(all_groups)/2)):
        group_b = tdms_file['BrightField_{}'.format(group_num)]
        group_f = tdms_file['Fluorescence_{}'.format(group_num)]

        all_channels_b = group_b.channels()
        all_channels_f = group_f.channels()

        slope_matrix_b = np.transpose(np.array(all_channels_b))
        slope_matrix_f = np.transpose(np.array(all_channels_f))

        #group_num_list.append(group_num)
        slope_matrix_b_list.append(slope_matrix_b)
        slope_matrix_f_list.append(slope_matrix_f)

    # return something like: [(mat_b_0, mat_f_0), (mat_b_1, mat_f_1)]
    return list(zip(slope_matrix_b_list[1:], slope_matrix_f_list[1:]))

def save_tiff(list_from_first_function, file_name, folder_name):
    b_file_list = []
    for num in range(len(list_from_first_function)):
        #group_num = list_from_first_function[i][0]
        slope_matrix_b = list_from_first_function[num][0]
        slope_matrix_f = list_from_first_function[num][1]
        plt.imsave(r'./{}/b_{}_{}.tiff'.format(folder_name, file_name, num), arr=slope_matrix_b, cmap='gray')
        plt.imsave(r'./{}/f_{}_{}.tiff'.format(folder_name, file_name, num), arr=slope_matrix_f, cmap='gray')
        b_file_list.append(r'./{}/b_{}_{}.tiff'.format(folder_name, file_name, num))

    return b_file_list

def run_model(b_tiff_folder_path, file_name, folder_name, diameter):
    model = models.Cellpose(gpu=True, model_type='cyto')

    # list of files
    files = glob.glob(os.path.join(b_tiff_folder_path, 'b_{}_*.tiff'.format(file_name)))
    #files = [os.path.join(b_tiff_folder_path, '/b_test_*.tiff')]
    files = sorted(files, key=lambda file: int(file.split('.tiff')[0].split('_')[-1]))

    imgs = [cellpose.io.imread(f) for f in files]
    nimg = len(imgs) # == number of BrightField images read

    channels = [[0,0]]

    global masks, flows
    masks, flows, styles, diams = model.eval(imgs, diameter=diameter, channels=channels)

    cellpose.io.masks_flows_to_seg(images=imgs,
                                   masks=masks,
                                   flows=flows,
                                   diams=diams,
                                   file_names=[r'./{}/b_{}'.format(folder_name, num) for num in range(nimg)],
                                   channels=None)

    seg_list = []
    for i in range(nimg):
        seg = np.load(r'./{}/b_{}_seg.npy'.format(folder_name, i), allow_pickle=True)
        seg_temp = seg.tolist()
        seg_list.append(seg_temp) #len(seg_list) == nimg

    return seg_list

def plot_segb(seg_list, list_from_first_function, folder_name):
    for i in range(len(seg_list)):
        outlines = np.zeros((640, 512))
        for j in range(0, 640):
            for k in range(0, 512):
                if seg_list[i]['outlines'][j][k] > 0:
                    outlines[j][k] = 1
                else:
                    pass
        fig = plt.figure(figsize=(3200/640, 3200/512))
        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(list_from_first_function[i][0], cmap='gray')
        ax.imshow(outlines*100, cmap='Reds', alpha=outlines, vmax=100, vmin=-100)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig(r'./{}/segb_{}.tiff'.format(folder_name, i))
        plt.close()

def plot_segf(seg_list, list_from_first_function, folder_name):
    for i in range(len(seg_list)):
        outlines = np.zeros((640, 512))
        for j in range(0, 640):
            for k in range(0, 512):
                if seg_list[i]['outlines'][j][k] > 0:
                    outlines[j][k] = 1
                else:
                    pass
        fig = plt.figure(figsize=(3200/640, 3200/512))
        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(list_from_first_function[i][1], cmap='gray')
        ax.imshow(outlines*100, cmap='Reds', alpha=outlines, vmax=100, vmin=-100)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #plt.show(block=False)

        plt.savefig(r'./{}/segf_{}.tiff'.format(folder_name, i))
        plt.close()

def save_masks(seg_list, b_file_list, folder_name):
    name = b_file_list[0].split('b_')[-1].split('_')[0:-1][0]
    cellpose.io.save_masks(b_file_list,
                          masks=masks,
                          flows=flows,
                          file_names=[r'./{}/{}_{}'.format(folder_name, name, num) for num in range(len(b_file_list))],
                          png=False,
                          tif=False,
                          save_txt=True)

def convert_outline(seg_list, b_file_list, folder_name):
    name = b_file_list[0].split('b_')[-1].split('_')[0:-1][0]
    all_boundaries: list = []
    for num in range(int(len(seg_list))):
        boundaries: list = []
        try:
            with open(r'./{}/{}_{}_cp_outlines.txt'.format(folder_name, name, num)) as f:
                lines = f.readlines()
                for line in lines:
                    pure_line: list = line.split('\n')[0]
                    pure_number: list = pure_line.split(',')
                    pixel_index = [(int(pure_number[i]), int(pure_number[i+1])) for i in range(0, len(pure_number), 2)]
                    boundaries.append(pixel_index)
            all_boundaries.append(boundaries)
        except FileNotFoundError:
            all_boundaries.append([])
            continue
    return all_boundaries

def calculate_f_sum_gating(seg_list, all_boundaries, list_from_first_function, folder_name):
    all_cell_mask: list = []
    all_label_mask: list = []
    all_cell_f_sum: list = []
    all_mask_distribution: list = []

    for num in range(int(len(all_boundaries))):
        boundaries = all_boundaries[num]
        cell_mask:list = []
        label_mask = np.full((640,512), fill_value=-1, dtype=int)
        count_pass = 0
        mask_distribution = np.zeros((640, 512))

        for cell_num in range(len(boundaries)):
            h_coord = [tup[1] for tup in boundaries[cell_num]]
            w_coord = [tup[0] for tup in boundaries[cell_num]]
            h_coord = np.array(h_coord)
            w_coord = np.array(w_coord)
            # if a cell is cut off by the edges, then we exclude it
            if np.count_nonzero(h_coord==639) > 2 or np.count_nonzero(h_coord==0) > 2\
                or np.count_nonzero(w_coord==511) > 2 or np.count_nonzero(w_coord==0) > 2:
                count_pass += 1
                continue
            else:
                modified_boundary = [(tup[1], tup[0]) for tup in boundaries[cell_num]]
                single_mask_temp = draw.polygon2mask((640, 512), np.array(modified_boundary))
                # if a cell is too small, then we exclude it, and cell_mask.append(np.zeros((640, 512)))
                # size of a cell is decided by single_mask_temp
                # which is a (640, 512) matrix, in which collected 1's represent a cell
                # cell_mask is a list containing np.zeros((640, 512))  and (1 - single_mask_temp)
                # (1 - single_cell_temp) is to make cell parts become 0. Otherwise we would mask out the cell parts.
                if np.count_nonzero(single_mask_temp) < 230:
                    cell_mask.append(np.zeros((640, 512)))
                    count_pass += 1
                else:
                # mask_distribution will be a (640, 512) matrix where 1's parts represent cells that are not too small
                    cell_mask.append(1-single_mask_temp)
                    #label_mask += single_mask_temp.astype(int)*(cell_num-count_pass+1)
                    mask_distribution += single_mask_temp
        all_cell_mask.append(cell_mask)
        #all_label_mask.append(label_mask)
        all_mask_distribution.append(mask_distribution)

    for i in range(int(len(all_cell_mask))):
        cell_mask = all_cell_mask[i]
        mask_distribution = all_mask_distribution[i]
        # labels will be a (640, 512) matrix, in which connected components (cells) are labeled from 1 to nb
        # so from left to right, top to bottom, we will see cluster of 1's, cluster of 2's, ...., cluster of nb's
        # each cluster represent one cell
        labels, nb = ndimage.label(mask_distribution) 
        cell_f_sum: list = []

        # here we investigate the size of the cluster
        # if one number appears more than 1500 times, which means that this cluster is larger than 1500, or say that this cell is larger than 1500
        # then the cell (or say connected component) is too large
        # which might result from aggregation and it is unwanted
        # so we will exclude this cluster
        # too_large_mask will be a (640, 512) matrix, in which each cluster of 1's represents a too large cell
        too_large_mask = np.zeros((640, 512))
        for j in range(1, nb+1):
            if np.where(labels==j)[0].shape[0] > 1500:
                too_large_mask += np.where(labels!=j, 0, 1)

        count = 0
        # _adj for adjusted
        label_mask_adj = np.full((640,512), fill_value=-1, dtype=int)
        cell_mask_adj: list = []
        count_pass_index: list = []
        for single_cell_mask in cell_mask:
            #-------if the cell size is not too big and not too small------#
            # just a reminder the single_cell_mask here is a (640, 512) matrix, in which cluster of 0's represents the single cell
            # The concept of the algorithm is listed in the following:
            # 1. 
            # if there's any 0 in (single_cell_mask + too_large_mask), which means that that single cell is covered up by the too_large mask,
            # or say that single cell is not part of aggregation
            # and 
            # 2.
            # if single_cell_mask is not np.zeros((640, 512)), which means that the cell is not too small (see above)
            # 3.
            # then we include the cell into cell_mask_adj
            if ((single_cell_mask + too_large_mask)==np.zeros((640, 512))).any() and (single_cell_mask==np.zeros((640, 512))).all() == False:
                count += 1
                cell_mask_adj.append(single_cell_mask)
                label_mask_adj += (1-single_cell_mask).astype(int)*(count)
            # if a single cell is part of aggregated cells or it is too small, then we exclude it.
            else:
                pass
        for single_cell_mask in cell_mask_adj:
            cell_fluorescence = np.ma.masked_array(list_from_first_function[i][1], single_cell_mask)
            cell_sum = np.sum(cell_fluorescence)
            cell_f_sum.append(cell_sum)
        all_cell_f_sum.append(cell_f_sum)
        all_label_mask.append(label_mask_adj)

    for i in range(int(len(all_cell_f_sum))):
        cell_f_sum = all_cell_f_sum[i]
        label_mask = all_label_mask[i]

        outlines = np.zeros((640, 512))
        for j in range(0, 640):
            for k in range(0, 512):
                if seg_list[i]['outlines'][j][k] > 0:
                    outlines[j][k] = 1
                else:
                    pass
        fig = plt.figure(figsize=(3200/640, 3200/512))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(list_from_first_function[i][1], cmap='gray')
        ax.imshow(outlines*100, cmap='Reds', alpha=outlines, vmax=100, vmin=-100)

        label_temp = 0
        for label_num in range(len(cell_f_sum)):
            try:
                coord = (np.where(label_mask==label_num)[0][0], np.where(label_mask==label_num)[1][0])
                ax.text(coord[1], coord[0], int(label_num), ha='center', va='center', c='white')
            except IndexError:
                continue

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig(r'./{}/segf2_{}.tiff'.format(folder_name, i))
        plt.close()

    return all_cell_f_sum

def calculate_f_sum_no_gating(seg_list, all_boundaries, list_from_first_function, folder_name):
    all_cell_mask: list = []
    all_label_mask: list = []
    all_cell_f_sum: list = []

    for num in range(int(len(all_boundaries))):
        boundaries = all_boundaries[num]
        cell_mask:list = []
        label_mask = np.full((640,512), fill_value=-1, dtype=int)
        count_pass = 0
        for cell_num in range(len(boundaries)):
            h_coord = [tup[1] for tup in boundaries[cell_num]]
            w_coord = [tup[0] for tup in boundaries[cell_num]]
            h_coord = np.array(h_coord)
            w_coord = np.array(w_coord)
            if np.count_nonzero(h_coord==639) > 2 or np.count_nonzero(h_coord==0) > 2\
                or np.count_nonzero(w_coord==511) > 2 or np.count_nonzero(w_coord==0) > 2:
                count_pass += 1
                continue
            else:
                modified_boundary = [(tup[1], tup[0]) for tup in boundaries[cell_num]]
                single_mask_temp = draw.polygon2mask((640, 512), np.array(modified_boundary))
                cell_mask.append(1-single_mask_temp)
                label_mask += single_mask_temp.astype(int)*(cell_num-count_pass+1)
        all_cell_mask.append(cell_mask)
        all_label_mask.append(label_mask)

    for i in range(int(len(all_cell_mask))):
        cell_mask = all_cell_mask[i]
        cell_f_sum: list = []
        for single_cell_mask in cell_mask:
            cell_fluorescence = np.ma.masked_array(list_from_first_function[i][1], single_cell_mask)
            cell_sum = np.sum(cell_fluorescence)
            cell_f_sum.append(cell_sum)
        all_cell_f_sum.append(cell_f_sum)

    for i in range(int(len(all_cell_f_sum))):
        cell_f_sum = all_cell_f_sum[i]
        label_mask = all_label_mask[i]

        outlines = np.zeros((640, 512))
        for j in range(0, 640):
            for k in range(0, 512):
                if seg_list[i]['outlines'][j][k] > 0:
                    outlines[j][k] = 1
                else:
                    pass
        fig = plt.figure(figsize=(3200/640, 3200/512))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(list_from_first_function[i][1], cmap='gray')
        ax.imshow(outlines*100, cmap='Reds', alpha=outlines, vmax=100, vmin=-100)

        label_temp = 0
        for label_num in range(len(cell_f_sum)):
            try:
                coord = (np.where(label_mask==label_num)[0][0], np.where(label_mask==label_num)[1][0])
                ax.text(coord[1], coord[0], int(label_num), ha='center', va='center', c='white')
            except IndexError:
                continue

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig(r'./{}/segf2_{}.tiff'.format(folder_name, i))
        plt.close()

    return all_cell_f_sum

def plot_hist(all_cell_f_sum, folder_name, file_name):
    fig = plt.figure(figsize=(13, (13-1.5)/1.618))
    ax = fig.add_axes([0.26, 0.15, 0.735, 0.735*13/(13-1.5)])

    p_low = math.floor(math.log10(np.min(all_cell_f_sum)))
    low = math.pow(10, p_low)
    p_high = math.ceil(math.log10(np.max(all_cell_f_sum)))
    high = math.pow(10, p_high)

    logbins = np.logspace(np.log10(low), np.log10(high), 30)

    # bins will be separated in this way
    ax.hist(all_cell_f_sum, logbins, histtype='bar',
            color='red', label='{}'.format(file_name),
            edgecolor='white') 

    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xscale('log')

    ax.set_xlabel('SWCNTs signal', fontsize=25, labelpad=10)
    ax.set_ylabel('Count', fontsize=25, labelpad=18)
    ax.legend(loc='best', fontsize=16, fancybox=True, framealpha=0.5)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15/1.5, top='on', direction='in', pad=15)
    ax.xaxis.set_tick_params(which='minor', labelsize=20, width=2.5, length=6/1.5, top='on', direction='in')

    ax.yaxis.set_tick_params(which='major', labelsize=20, width=2.5, length=15/2.5, right='on', direction='in', pad=15)
    ax.tick_params(axis='y', which='minor', left=False, right=False)

    for i in ['right', 'left', 'top', 'bottom']:
        ax.spines[i].set_linewidth(2.5)

    plt.savefig(r'./{}/hist_{}.png'.format(folder_name, file_name))

def add_files_in_folder(parent, dirname):
    files = os.listdir(dirname)
    for f in sorted(files):
        fullname = os.path.join(dirname, f)
        if os.path.isdir(fullname):            # if it's a folder, add folder and recurse
            treedata.Insert(parent, fullname, f, values=[], icon=folder_icon)
            add_files_in_folder(fullname, fullname)
        else:
            treedata.Insert(parent, fullname, f, values=[os.stat(fullname).st_size], icon=file_icon)

def main():
    #file_path = askopenfilename()
    global folder_icon, file_icon
    folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
    file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'

    options: list = ['YES', 'NO']
    #-----GUI Definition-----#
    sg.theme('BrightColors')

    while True:
        starting_path = sg.popup_get_folder('Select the folder', font='Courier 20')
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
            event0, values0 = window0.read()
            if event0 in (sg.WIN_CLOSED, 'Cancel'):
                window0.close()
                break
            elif event0 == 'Ok':
                data_file_path_list = values0['-TREE-']
            else:
                continue

            print(data_file_path_list)
            while True:

                layout1 = [
                [sg.Text('Gating or not?', font='Courier 20', size=(13,1)),
                sg.InputCombo(values=options, default_value='YES', font='Courier 20', size=(5,1))],
                [sg.Text('Diameter?', font='Courier 20', size=(13,1)),
                sg.InputText('', font='Courier 20', size=(5,1))],
                [sg.Button('Run', font='Courier 20'),
                sg.Push(),
                sg.Exit(button_color='tomato', font='Courier 20')],
                ]

                window1 = sg.Window('Setting Segmentation Parameters', layout1, resizable=True)

                while True:
                    event1, values1 = window1.read()
                    print(values1)
                    diameter = int(values1[1])

                    if event1 in (sg.WIN_CLOSED, 'Exit'):
                        break
                    elif event1 == 'Run' and values1[0] == 'YES':
                        t0 = time.perf_counter()
                        for i in range(len(data_file_path_list)):
                            file_path = data_file_path_list[i]
                            folder_name = file_path.split('.')[0].split('\\')[-1]
                            folder_path = os.path.join(os.getcwd(), folder_name)
                            os.mkdir(folder_path)

                            t0 = time.perf_counter()
                            list_from_first_function = generate_slope_mat_from_tdms(file_path)
                            #After reading tdms from path, we only need file name
                            file_name = file_path.split('.')[0].split('\\')[-1]
                            b_file_list = save_tiff(list_from_first_function, file_name, folder_name)
                            seg_list = run_model(r'./{}'.format(folder_name), file_name, folder_name, diameter)
                            plot_segb(seg_list, list_from_first_function, folder_name)
                            plot_segf(seg_list, list_from_first_function, folder_name)
                            save_masks(seg_list, b_file_list, folder_name)
                            all_boundaries = convert_outline(seg_list, b_file_list, folder_name)
                            all_cell_f_sum = calculate_f_sum_gating(seg_list, all_boundaries, list_from_first_function, folder_name)
                            all_cell_f_sum_dict: dict = {i: all_cell_f_sum[i] for i in range(len(all_cell_f_sum))}
                            df = pd.DataFrame.from_dict(all_cell_f_sum_dict, orient='index')
                            df = df.transpose()
                            df.to_csv(r'./{}/{}.csv'.format(folder_name, file_name), mode='a', index=True, header=True)
                            all_cell_f_sum = df.to_numpy().flatten()
                            all_cell_f_sum = all_cell_f_sum[~np.isnan(all_cell_f_sum)]
                            plot_hist(all_cell_f_sum, folder_name, file_name)
                            print('-----------------------------------------')
                            print('               FINISH {}                 '.format(file_name))
                            print('-----------------------------------------')
                        t1 = time.perf_counter()
                        print(t1-t0, 'seconds')
                        break

                    elif event1 == 'Run' and values1[0] == 'NO':
                        t0 = time.perf_counter()
                        for i in range(len(data_file_path_list)):
                            file_path = data_file_path_list[i]
                            folder_name = file_path.split('.')[0].split('\\')[-1]
                            folder_path = os.path.join(os.getcwd(), folder_name)
                            os.mkdir(folder_path)

                            t0 = time.perf_counter()
                            list_from_first_function = generate_slope_mat_from_tdms(file_path)
                            #After reading tdms from path, we only need file name
                            file_name = file_path.split('.')[0].split('\\')[-1]
                            b_file_list = save_tiff(list_from_first_function, file_name, folder_name)
                            seg_list = run_model(r'./{}'.format(folder_name), file_name, folder_name, diameter)
                            plot_segb(seg_list, list_from_first_function, folder_name)
                            plot_segf(seg_list, list_from_first_function, folder_name)
                            save_masks(seg_list, b_file_list, folder_name)
                            all_boundaries = convert_outline(seg_list, b_file_list, folder_name)
                            all_cell_f_sum = calculate_f_sum_no_gating(seg_list, all_boundaries, list_from_first_function, folder_name)
                            all_cell_f_sum_dict: dict = {i: all_cell_f_sum[i] for i in range(len(all_cell_f_sum))}
                            df = pd.DataFrame.from_dict(all_cell_f_sum_dict, orient='index')
                            df = df.transpose()
                            df.to_csv(r'./{}/{}.csv'.format(folder_name, file_name), mode='a', index=True, header=True)
                            all_cell_f_sum = df.to_numpy().flatten()
                            all_cell_f_sum = all_cell_f_sum[~np.isnan(all_cell_f_sum)]
                            plot_hist(all_cell_f_sum, folder_name, file_name)
                            print('-----------------------------------------')
                            print('               FINISH {}                 '.format(file_name))
                            print('-----------------------------------------')
                        t1 = time.perf_counter()
                        print(t1-t0, 'seconds')
                        break

                window1.close()
                break

if __name__ == '__main__':
    main()




