#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ### mount on google drive if you running the code on colab
# from google.colab import drive
# drive.mount('/content/drive/')
# import os
# os.chdir("/content/drive/My Drive/WatNet/notebooks")


# In[3]:


import os
# os.chdir('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.geotif_io import readTiff, writeTiff
from utils.acc_pixel import acc_matrix
from watnet_infer import watnet_infer
from PIL import Image as im

# In[6]:

def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2
## test image and test sample

dtype = np.dtype('>u2')
shape = (809,809,1)
B2 = np.fromfile(open('data/test-demo/B2.img', 'rb'), dtype).reshape(shape)
B3 = np.fromfile(open('data/test-demo/B3.img', 'rb'), dtype).reshape(shape)
B4 = np.fromfile(open('data/test-demo/B4.img', 'rb'), dtype).reshape(shape)
B5 = np.fromfile(open('data/test-demo/B8.img', 'rb'), dtype).reshape(shape)
B6 = np.fromfile(open('data/test-demo/B11.img','rb'), dtype).reshape(shape)
B7 = np.fromfile(open('data/test-demo/B12.img','rb'), dtype).reshape(shape)

inter_1 = np.concatenate((B2, B3), axis=2)
inter_2 = np.concatenate((inter_1, B4), axis=2)
inter_3 = np.concatenate((inter_2, B5), axis=2)
inter_4 = np.concatenate((inter_3, B6), axis=2)
sen2_img = np.concatenate((inter_4, B7), axis=2)
# path_S2_img = 'data/test-demo/T49QGF_20191017_6Bands_Urban_Subs.tif'
# pre-trained model
path_model = 'model/pretrained/watnet.h5'
path_result_map = r'data/test-demo/result_water_map.png'
## super parameters
patch_size = 512
overlay = 80  # the overlay area between neighboring patches


# In[7]:


# -----Load and prepare the satellite image data -----#
# sen2_img, img_info = readTiff(path_in=path_S2_img)
pad_r = find_padding(sen2_img.shape[0])
pad_c = find_padding(sen2_img.shape[1])
sen2_img = np.pad(sen2_img, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')
# solve no-pad index issue after inference
if pad_r[1] == 0:
    pad_r = (pad_r[0], 1)
if pad_c[1] == 0:
    pad_c = (pad_c[0], 1)
sen2_img = np.float32(np.clip(sen2_img/10000, a_min=0, a_max=1))  ## normalization
# plt.figure(figsize=(6,6))
# plt.imshow(sen2_img[:, :, (2,1,0)]*5)


# In[10]:


### ---- surface water mapping by using pretrained watnet.

water_map = watnet_infer(rsimg=sen2_img)
water_map = np.squeeze(water_map)
water_map = water_map[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

# soft threshold
water_map = 1./(1+np.exp(-(16*(water_map-0.5))))
water_map = np.clip(water_map, 0, 1)

# save the output water map
cv2.imwrite(path_result_map, water_map * 255)

# In[11]:


### show the result
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.imshow(sen2_img[1500:2000, 1000:1500, (3,2,1)]*6.5)
# plt.subplot(1,2,2)
# plt.imshow(water_map[1500:2000, 1000:1500, 0])


# In[18]:


## show the accuracy
# path_sam = 'data/test-data(demo)/val_sam.csv'
# df_sam = pd.read_csv(path_sam, header=0)
# acc_oa, acc_prod, acc_user, confus_mat = acc_matrix(water_map, df_sam.to_numpy(), id_label=1)
# print('OA:', acc_oa)
# print('Producer acc:', acc_prod)
# print('user acc:', acc_user)
# print('Confusion matrix:', np.around(confus_mat,4))
# plt.matshow(confus_mat,cmap=plt.cm.Greys, fignum=0)
# plt.colorbar()


# ### Save the result.

# In[19]:


# img_info


# In[20]:


# # # save the result
# writeTiff(im_data = water_map.astype(np.int8), 
#           im_geotrans = img_info['geotrans'], 
#           im_geosrs = img_info['geosrs'], 
#           path_out = path_result_map)

