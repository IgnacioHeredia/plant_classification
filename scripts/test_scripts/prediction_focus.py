#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Prediction focus

Author: Ignacio Heredia
Date: August 2017

Implement sliding (black) square to seek the most interesting parts of the 
image by occluding them and looking at the changes in the prediction confidence.

Parameters
----------
square_size : positive int
    Size of the occluding square. Useful to see importance across scales.
stride : positive int
    Step size when moving the square along the image. If different than 1 then 
    you interpolate for values in between. If equal to 1 then you do the detailed 
    prediction. The higher the value the faster the prediction is made.
square_RGBcolor : numpy array. Shape (1,3)
    RGB color of the occluding square
    
"""

import numpy as np
import os
import sys
import json
homedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(homedir, 'scripts'))
from model_utils import load_model
from PIL import Image
import matplotlib.pylab as plt
import scipy as sp


#Parameters
img = '/home/ignacio/cortaderia_selloana.jpg' #path to image
square_size = 50
stride = 4 # if =1 -> exact calculation (no interpolation)
square_RGBcolor = np.array([0,0,0]) #black

#==============================================================================
# Load the model
#==============================================================================

modelname = 'resnet50_6182classes_100epochs'
                          
# Load training info
info_file = os.path.join(homedir, 'scripts', 'training_info', modelname + '.json')
with open(info_file) as datafile:
    train_info = json.load(datafile)
mean_RGB = train_info['augmentation_params']['mean_RGB']
output_dim = train_info['training_params']['output_dim']

# Load net weights
test_func = load_model(os.path.join(homedir, 'scripts', 'training_weights', modelname + '.npz'), output_dim=output_dim)

#==============================================================================
# Processing the occluded image
#==============================================================================

X, Y = 224, 224
batchsize = 64
ori = Image.open(img)
ori = ori.resize((X,Y))
black_square = np.ones((square_size,square_size,3)) * square_RGBcolor

print 'Processing ocluded images...'
im_list, pred = [], np.array([]).reshape(0, output_dim)
ind = 0
x_range = np.arange(0, X-square_size, stride)
y_range = np.arange(0, Y-square_size, stride)
tot = x_range.shape[0] * y_range.shape[0]
num_batch = np.ceil(1.*tot/batchsize)
if mean_RGB is None:
    mean_RGB = np.array([107.59348955,  112.1047813,   80.9982362])

for y in y_range:
    for x in x_range:
        img_occ = np.array(ori)
        img_occ[y:y+square_size, x:x+square_size, :] = black_square
        im_list.append(img_occ)
        ind += 1
        if (ind % batchsize)==0 or ind==tot:
            print 'Batch {:.0f}/{:.0f}'.format(np.ceil(1.*ind/batchsize), num_batch)
            
            #Preprocess the batch
            im_list = np.array(im_list) - mean_RGB[None, None, None, :]  # mean centering
            im_list = im_list.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
            im_list = im_list[:, ::-1, :, :]  # switch from RGB to BGR   
            im_list = im_list.astype(np.float32)

            #Predict
            pred_raw = test_func(im_list)
            pred = np.vstack((pred, pred_raw))
            im_list = []

# Main prediction (not ocludded)
img = np.array(ori)
im_list = [img]
im_list = np.array(im_list) - mean_RGB[None, None, None, :]  # mean centering
im_list = im_list.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
im_list = im_list[:, ::-1, :, :]  # switch from RGB to BGR   
im_list = im_list.astype(np.float32)
pred_raw = test_func(im_list)
arg_max = np.argmax(pred_raw)

# Select probabilities corresponding to that prediction
pred_prob = pred[:, arg_max]
pred_prob = pred_prob.reshape(len(y_range), len(x_range))
pred_prob = 1 - pred_prob #the more important the occluded part is, the lower the probability it has 

#Interpolate between the values
f = sp.interpolate.RectBivariateSpline(x_range, y_range, pred_prob)
pred_prob = f(np.arange(X-square_size), np.arange(0, Y-square_size))

#==============================================================================
# Plot results
#==============================================================================

ori_patch = np.array(ori)[square_size/2:-square_size/2, square_size/2:-square_size/2]
new_prob = pred_prob - np.amin(pred_prob)
new_prob /= np.amax(new_prob)

# Plot mask and original image
fig, ax = plt.subplots(1,2)
cf = ax[0].imshow(pred_prob, clim=[np.amin(pred_prob), np.amax(pred_prob)])
fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
ax[1].imshow(ori_patch)
ax[0].set_title('Importance mask')
ax[1].set_title('Original image (patch)')

# Plot demo of occluding
plt.figure(2)
plt.imshow(img_occ)
plt.title('Occluding demo')

# Plot RGBA image (soft mask)
plt.figure(3)
alpha_channel = np.round(new_prob*255)
rgba_arr = np.dstack((ori_patch, alpha_channel)).astype(np.uint8)
plt.imshow(rgba_arr)
plt.title('RGBA image')

# Plot contour plots over the image
plt.figure(4)
plt.imshow(ori_patch)
CS = plt.contour(np.arange(X-square_size), np.arange(0, Y-square_size), pred_prob)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour plot')

## Plot hard mask
#plt.figure(5)
#mask = (new_prob < 0.7)
#ori_masked = np.copy(ori_im)
#ori_masked[mask, :] = np.array([0,0,0])
#plt.imshow(ori_masked)
