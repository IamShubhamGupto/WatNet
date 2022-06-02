## author: xin luo, creat: 2021.8.11

'''
des: perform surface water mapping by using pretrained watnet
     through funtional api and command line, respectively.

example:
     funtional api:
        water_map = watnet_infer(rsimg) 
     command line: 
        python watnet_infer.py data/test-demo/*.tif
        python watnet_infer.py data/test-demo/*.tif -o data/test-demo/result
    note: 
        rsimg is np.array (row,col,band), value: [0,1]
        data/test-demo/*.tif is the sentinel-2 image path
        data/test-demo/result is output directory
'''

import os
import numpy as np
import tensorflow as tf
import argparse
from utils.imgPatch import imgPatch
from utils.geotif_io import readTiff,writeTiff
import cv2
## default path of the pretrained watnet model
path_watnet = 'model/pretrained/watnet.h5'
def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

def get_args():

    description = 'surface water mapping by using pretrained watnet'
    parser = argparse.ArgumentParser(description=description)

    # parser.add_argument(
    #     'ifile', metavar='ifile', type=str, nargs='+',
    #     help=('file(s) to process (.tiff)'))

    parser.add_argument('--B2', type=str, help="Path to the B2 GeoTIFF image")
    parser.add_argument('--B3', type=str, help="Path to the B3 GeoTIFF image")
    parser.add_argument('--B4', type=str, help="Path to the B4 GeoTIFF image")
    parser.add_argument('--B5', type=str, help="Path to the B5 GeoTIFF image")
    parser.add_argument('--B6', type=str, help="Path to the B6 GeoTIFF image")
    parser.add_argument('--B7', type=str, help="Path to the B7 GeoTIFF image")

    parser.add_argument(
        '-m', metavar='watnet', dest='watnet', type=str, 
        nargs='+', default=path_watnet, 
        help=('pretrained watnet model (tensorflow2, .h5)'))

    parser.add_argument('--save_path', type=str, help="Path where the output map will be saved")

    return parser.parse_args()


def watnet_infer(rsimg, path_model = path_watnet):

    ''' des: surface water mapping by using pretrained watnet
        arg:
            img: np.array, surface reflectance data (!!data value: 0-1), 
                 consist of 6 bands (blue,green,red,nir,swir-1,swir-2).
            path_model: str, the path of the pretrained model.
        retrun:
            water_map: np.array.
    '''
    ###  ----- load the pretrained model -----#
    model = tf.keras.models.load_model(path_model, compile=False)
    ### ------ apply the pre-trained model
    imgPatch_ins = imgPatch(rsimg, patch_size=512, edge_overlay=80)
    patch_list, start_list, img_patch_row, img_patch_col = imgPatch_ins.toPatch()
    result_patch_list = [model(patch[np.newaxis, :]) for patch in patch_list]
    result_patch_list = [np.squeeze(patch, axis = 0) for patch in result_patch_list]
    pro_map = imgPatch_ins.toImage(result_patch_list, img_patch_row, img_patch_col)
    water_map = np.where(pro_map>0.5, 1, 0)

    return water_map


if __name__ == '__main__':
    args = get_args()
    dtype = np.dtype('>u2')
    shape = (809,809,1)
    B2 = np.fromfile(open(args.B2, 'rb'), dtype).reshape(shape)
    B3 = np.fromfile(open(args.B3, 'rb'), dtype).reshape(shape)
    B4 = np.fromfile(open(args.B4, 'rb'), dtype).reshape(shape)
    B5 = np.fromfile(open(args.B5, 'rb'), dtype).reshape(shape)
    B6 = np.fromfile(open(args.B6, 'rb'), dtype).reshape(shape)
    B7 = np.fromfile(open(args.B7, 'rb'), dtype).reshape(shape)

    print(B2.shape)
    inter_1 = np.concatenate((B2, B3), axis=2)
    inter_2 = np.concatenate((inter_1, B4), axis=2)
    inter_3 = np.concatenate((inter_2, B5), axis=2)
    inter_4 = np.concatenate((inter_3, B6), axis=2)
    sen2_img = np.concatenate((inter_4, B7), axis=2)

    pad_r = find_padding(sen2_img.shape[0])
    pad_c = find_padding(sen2_img.shape[1])
    sen2_img = np.pad(sen2_img, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')
    # solve no-pad index issue after inference
    if pad_r[1] == 0:
        pad_r = (pad_r[0], 1)
    if pad_c[1] == 0:
        pad_c = (pad_c[0], 1)
    sen2_img = np.float32(np.clip(sen2_img/10000, a_min=0, a_max=1))  ## normalization
    ## surface water mapping by using watnet
    water_map = watnet_infer(rsimg=sen2_img)
    dwm = np.squeeze(water_map)
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

    # soft threshold
    dwm = 1./(1+np.exp(-(16*(dwm-0.5))))
    dwm = np.clip(dwm, 0, 1)

    # save the output water map
    cv2.imwrite(args.save_path, dwm * 255)
    # ifile = args.ifile
    # path_model = args.watnet
    # odir = args.odir
    # ## write path
    # if odir:
    #     if not os.path.exists(odir[0]):
    #         os.makedirs(odir[0])
    #     ofile = [os.path.splitext(file)[0] + '_water.tif' for file in ifile]
    #     ofile = [os.path.join(odir[0], os.path.split(file)[1]) for file in ofile]
    # else:
    #     ofile = [os.path.splitext(file)[0] + '_water.tif' for file in ifile]

    # for i in range(len(ifile)):
    #     print('file in -->', ifile[i])
    #     ## image reading and normalization
    #     sen2_img, img_info = readTiff(path_in=ifile[i])
    #     sen2_img = np.float32(np.clip(sen2_img/10000, a_min=0, a_max=1))  ## normalization
    #     ## surface water mapping by using watnet
    #     water_map = watnet_infer(rsimg=sen2_img)
    #     # write out the result
    #     print('write out -->', ofile[i])
    #     writeTiff(im_data = water_map.astype(np.int8), 
    #             im_geotrans = img_info['geotrans'], 
    #             im_geosrs = img_info['geosrs'], 
    #             path_out = ofile[i])


