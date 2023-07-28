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

def generate_slope_mat_from_spe():
    spe_tools = spe.load()

    frame_count_b = 7
    frame_count_f = 15

    x_b = np.arange(0, frame_count_b)
    x_f = np.arange(0, frame_count_f)

    slope_matrix_b_list: list = []
    slope_matrix_f_list: list = []
    for num in range(int(len(spe_tools))):
        if num%2 == 0: #BrightField
            slope_matrix_b = np.zeros((640, 512))
            for i in range(0, 512):
                for j in range(0, 640):
                    intensity_frame = []
                    for k in range(0, frame_count_b):
                        intensity_frame.append(spe_tools[num].file.data[k][0][j][i])
                    coef = np.polyfit(x_b, intensity_frame, 1)
                    slope_matrix_b[j][i] = coef[0]
            slope_matrix_b_list.append(slope_matrix_b)

        else: #Fluorescence
            slope_matrix_f = np.zeros((640, 512))
            for i in range(0, 512):
                for j in range(0, 640):
                    intensity_frame = []
                    for k in range(0, frame_count_f):
                        intensity_frame.append(spe_tools[num].file.data[k][0][j][i])
                    coef = np.polyfit(x_f, intensity_frame, 1)
                    slope_matrix_f[j][i] = coef[0]
            slope_matrix_f_list.append(slope_matrix_f)

    # return something like: [(mat_b_0, mat_f_0), (mat_b_1, mat_f_1)]
    return list(zip(slope_matrix_b_list, slope_matrix_f_list))

def save_tiff(list_from_first_function, name, folder_name):
    b_file_list = []
    for num in range(len(list_from_first_function)):
        slope_matrix_b = list_from_first_function[num][0]
        slope_matrix_f = list_from_first_function[num][1]
        plt.imsave(r'./{}/b_{}_{}.tiff'.format(folder_name, name, num), arr=slope_matrix_b, cmap='gray')
        plt.imsave(r'./{}/f_{}_{}.tiff'.format(folder_name, name, num), arr=slope_matrix_f, cmap='gray')
        b_file_list.append(r'./{}/b_{}_{}.tiff'.format(folder_name, name, num))

    return b_file_list

def run_model(b_tiff_folder_path, name, folder_name):
    model = models.Cellpose(gpu=True, model_type='cyto')

    # list of files
    files = glob.glob(os.path.join(b_tiff_folder_path, 'b_{}_*.tiff'.format(name)))
    #files = [os.path.join(b_tiff_folder_path, '/b_test_*.tiff')]
    files = sorted(files, key=lambda file: int(file.split('.tiff')[0].split('_')[-1]))

    imgs = [cellpose.io.imread(f) for f in files]
    nimg = len(imgs) # == number of BrightField images read

    #print(nimg)

    channels = [[0,0]]

    global masks, flows
    masks, flows, styles, diams = model.eval(imgs, diameter=35, channels=channels)

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
        with open(r'./{}/{}_{}_cp_outlines.txt'.format(folder_name, name, num)) as f:
            lines = f.readlines()
            for line in lines:
                pure_line: list = line.split('\n')[0]
                pure_number: list = pure_line.split(',')
                pixel_index = [(int(pure_number[i]), int(pure_number[i+1])) for i in range(0, len(pure_number), 2)]
                boundaries.append(pixel_index)
        all_boundaries.append(boundaries)
    return all_boundaries

def calculate_f_sum(seg_list, all_boundaries, list_from_first_function, folder_name):
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
            if np.count_nonzero(h_coord==639) > 3 or np.count_nonzero(h_coord==1) > 3\
                or np.count_nonzero(w_coord==511) > 3 or np.count_nonzero(w_coord==1) > 3:
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

def main():
    file_name = input('Files will be named as: ')
    folder_name = input('Folder where files will be saved: ')

    t0 = time.perf_counter()
    list_from_first_function = generate_slope_mat_from_spe()
    b_file_list = save_tiff(list_from_first_function, file_name, folder_name)
    seg_list = run_model(r'./{}'.format(folder_name), file_name, folder_name)
    plot_segb(seg_list, list_from_first_function, folder_name)
    plot_segf(seg_list, list_from_first_function, folder_name)
    save_masks(seg_list, b_file_list, folder_name)
    all_boundaries = convert_outline(seg_list, b_file_list, folder_name)
    all_cell_f_sum = calculate_f_sum(seg_list, all_boundaries, list_from_first_function, folder_name)
    all_cell_f_sum_dict: dict = {i: all_cell_f_sum[i] for i in range(len(all_cell_f_sum))}
    df = pd.DataFrame.from_dict(all_cell_f_sum_dict, orient='index')
    df = df.transpose()
    df.to_csv(r'./{}/{}.csv'.format(folder_name, file_name), mode='a', index=True, header=True)
    t1 = time.perf_counter()

    print(t1-t0, 'seconds')

if __name__ == '__main__':
    main()




