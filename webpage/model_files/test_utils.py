# -*- coding: utf-8 -*-
"""
Miscellanous functions used to evaluate image classification in the demo webpage.

Author: Ignacio Heredia
Date: December 2016
"""

import numpy as np
import os
import sys
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
sys.path.append(os.path.dirname(os.path.realpath(__file__))) 

import theano
import theano.tensor as T
import lasagne

theano.config.floatX = 'float32'


def load_model(modelweights, output_dim):
    """
    Loads a model with some trained weights and returns the test function that
    gives the deterministic predictions.

    Parameters
    ----------
    modelweights : str
        Name of the weights file
    outputdim : int
        Number of classes to predict

    Returns
    -------
    Test function
    """
    print 'Loading the model...'
    input_var = T.tensor4('X', dtype=theano.config.floatX)
    from models.resnet50 import build_model
    net = build_model(input_var, output_dim)
    # Load pretrained weights
    with np.load(modelweights) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['prob'], param_values)
    # Define test function
    test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
    test_fn = theano.function([input_var], test_prediction)
    return test_fn


def data_augmentation(im_list, mode='standard', tags=None, params=None, im_size=224,
                      filemode='local', mean_RGB=None):
    """
    Perform data augmentation on some image list using PIL.

    Parameters
    ----------
    im_list : array of strings
        Array where the first column is image_path (or image_url). Optionally
        a second column can be the tags of the image.
        Shape (N,) or (N,2)
    tags : array of strings, None
        If existing, you can the manually modify the data_augmentation function
        (by adding an additional condition to the if, like tags[i]=='fruit')
        to choose which transformations are to be performed to each tag.
    params : dict or None
        Mandatory keys:
        - mirror (bool): allow 50% random mirroring.
        - rotation (bool): allow rotation of [0, 90, 180, 270] degrees.
        - stretch ([0,1] float): randomly stretch image.
        - crop ([0,1] float): randomly take an image crop.
        - zoom ([0,1] float): random zoom applied to crop_size.
          --> Therefore the effective crop size at each iteration will be a 
              random number between 1 and crop*(1-zoom). For example:
                  * crop=1, zoom=0: no crop of the image
                  * crop=1, zoom=0.1: random crop of random size between 100% image and 90% of the image
                  * crop=0.9, zoom=0.1: random crop of random size between 90% image and 80% of the image
                  * crop=0.9, zoom=0: random crop of always 90% of the image
                  Image size refers to the size of the shortest side.
        - pixel_noise (bool): allow different pixel transformations like gaussian noise,
          brightness, color jittering, contrast and sharpness modification.
    mode : {'standard', 'minimal', 'test', None}
        We overwrite the params dict with some defaults augmentation parameters
        - 'minimal': no data augmentation, just resizing
        - 'standard': tipical parameters for data augmentation during training
        - 'test': minimized data augmentation for validation/testing
        - None: we do not overwrite the params dict variable
    im_size : int
        Final image size to feed the net's input (eg. 224 for Resnet).
    filemode : {'local','url'}
        - 'local': filename is absolute path in local disk.
        - 'url': filename is internet url.
    mean_RGB : array, None
        Mean RGB values for your dataset. If not provided, we use some default values.

    Returns
    -------
    Array of shape (N,3,im_size,im_size) containing the augmented images.

    """
    if mean_RGB is None:
        mean_RGB = np.array([107.59348955,  112.1047813,   80.9982362])
    else:
        mean_RGB = np.array(mean_RGB)   

    if mode == 'minimal':
        params = {'mirror':False, 'rotation':False, 'stretch':False, 'crop':False, 'pixel_noise':False}
    if mode == 'standard':
        params = {'mirror':True, 'rotation':True, 'stretch':0.3, 'crop':1., 'zoom':0.3, 'pixel_noise':False}
    if mode == 'test':
        params = {'mirror':True, 'rotation':True, 'stretch':0.1, 'crop':.9, 'zoom':0.1, 'pixel_noise':False}
    
    batch = []
    for i, filename in enumerate(im_list):
        
        if filemode == 'local':
            im = Image.open(filename)
            im = im.convert('RGB')
        elif filemode == 'url':
            filename = BytesIO(requests.get(filename).content)
            im = Image.open(filename)
            im = im.convert('RGB')
                
        if params['stretch']:
            stretch = params['stretch']
            stretch_factor = np.random.uniform(low=1.-stretch/2, high=1.+stretch/2, size=2)
            im = im.resize((im.size * stretch_factor).astype(int))
            
        if params['crop']:
            effective_zoom = np.random.rand() * params['zoom']
            crop = params['crop'] - effective_zoom
            
            ly, lx = im.size
            crop_size = crop * min([ly, lx]) 
            rand_x = np.random.randint(low=0, high=lx-crop_size + 1)
            rand_y = np.random.randint(low=0, high=ly-crop_size + 1)
                
            min_yx = np.array([rand_y, rand_x])
            max_yx = min_yx + crop_size #square crop
            im = im.crop(np.concatenate((min_yx, max_yx)))
            
        if params['mirror']:
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
        
        if params['rotation']:
            rot = np.random.choice([0, 90, 180, 270])
            if rot == 90:
                im = im.transpose(Image.ROTATE_90)
            if rot == 180:
                im = im.transpose(Image.ROTATE_180)
            if rot == 270:
                im = im.transpose(Image.ROTATE_270)            

        if params['pixel_noise']:
            
            #not used by defaul as it does not seem to improve much the performance,
            #but more than DOUBLES the data augmentation processing time.
            
            # Color
            color_factor = np.random.normal(1, 0.3)  #1: original
            color_factor = np.clip(color_factor, 0., 2.)
            im = ImageEnhance.Color(im).enhance(color_factor)
            
            # Brightness
            brightness_factor = np.random.normal(1, 0.2)  #1: original
            brightness_factor = np.clip(brightness_factor, 0.5, 1.5)
            im = ImageEnhance.Brightness(im).enhance(brightness_factor)
            
            # Contrast
            contrast_factor = np.random.normal(1, 0.2)  #1: original
            contrast_factor = np.clip(contrast_factor, 0.5, 1.5)
            im = ImageEnhance.Contrast(im).enhance(contrast_factor)
            
            # Sharpness
            sharpness_factor = np.random.normal(1, 0.4)  #1: original
            sharpness_factor = np.clip(sharpness_factor, 0., 1.)
            im = ImageEnhance.Sharpness(im).enhance(sharpness_factor)

#            # Gaussian Noise #severely deteriorates learning 
#            im = np.array(im)
#            noise = np.random.normal(0, 15, im.shape)
#            noisy_image = np.clip(im + noise, 0, 255).astype(np.uint8)
#            im = Image.fromarray(noisy_image)

        im = im.resize((im_size, im_size))
        batch.append(np.array(im))  # shape (N, 224, 224, 3)

    batch = np.array(batch) - mean_RGB[None, None, None, :]  # mean centering
    batch = batch.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
    batch = batch[:, ::-1, :, :]  # switch from RGB to BGR
    return batch.astype(np.float32)


def standard_tencrop_batch(filename, filemode='local', crop_prop=0.8, im_size=224):
    """
    Returns an ordered ten crop batch of images from an original image (corners,
    center + mirrors).

    Parameters
    ----------
    filename : str
        Image path
    crop_size : float
        Size of the crop with respect to the original image.
    im_size : int
        Size of the output image to feed the net.
            filemode : str
    filemode : {'local','url'}
        * 'local' -- filename is absolute path in local disk.
        * 'url' -- filename is internet url.

    Returns
    -------
    Array of shape (10,3,im_size,im_size) containing the augmented images.

    """
    batch = []
    mean_RGB = np.array([107.59348955, 112.1047813, 80.9982362])
    if filemode == 'local':
        im = Image.open(filename)
    elif filemode == 'url':
        filename = BytesIO(requests.get(filename).content)
        im = Image.open(filename)
        im = im.convert('RGB')
    min_side = min(im.size)
    im = im.resize((min_side, min_side))  # resize to shorter border
    h, w = min_side, min_side  # height, width (square)
    crop_size = int(crop_prop * min_side)

    # Crops
    c1 = im.crop((0, 0, crop_size, crop_size))          # top-left
    c2 = im.crop((0, h-crop_size, crop_size, h))        # bottom-left
    c3 = im.crop((w-crop_size, 0, w, crop_size))        # top-right
    c4 = im.crop((w-crop_size, h-crop_size, w, h))      # bottom-right
    c5 = im.crop(((w-crop_size)/2, (h-crop_size)/2,
                  (w+crop_size)/2, (h+crop_size)/2))    # center

    # Save crop and its mirror
    for image in [c1, c2, c3, c4, c5]:
        image = image.resize((im_size, im_size))
        batch.append(np.array(image))
        batch.append(np.array(image.transpose(Image.FLIP_LEFT_RIGHT)))

    batch = (np.array(batch) - mean_RGB)  # mean centering
    batch = batch.transpose(0, 3, 1, 2)  # shape(10, 3, 224, 224)
    batch = batch[:, ::-1, :, :]  # switch from RGB to BGR
    return batch.astype(np.float32)


def single_prediction(test_func, im_list, aug_params=None, crop_mode='random'):
    """
    Function for identying a SINGLE plant with one or more images and to
    combine the predictions for all the images to output the best possible labels.

    Parameters
    ----------
    test_func : theano function
        Function to make predictions
    im_list : list
        List of image filepaths or urls.
    aug_params : dict, None, optional
        Parameters for data augmentation.
    crop_mode : {'random','standard'}
        Modality of croppping. Random usually works better.

    Returns
    -------
    Arrays with top 5 predicted labels numbers and their corresponding probabilities.

    """
    if aug_params is None:
        aug_params = {}
    aug_params.pop('mode', None)
    pred = []
    for i, im in enumerate(im_list):
        print 'Image number: {}'.format(i)
        try:
            if crop_mode == 'random':
                batch = data_augmentation([im]*10, mode='test', **aug_params)
            if crop_mode == 'standard':
                batch = standard_tencrop_batch(im, **aug_params)
        except Exception:
            print 'Error at Image {}'.format(i)
            continue
        pred_raw = test_func(batch)  # probabilities for all labels for all 10 crops
        pred_tmp = np.sum(pred_raw, axis=0) / 10.  # mean probabilities across crops
        pred.append(pred_tmp)
    pred_prob = np.sum(pred, axis=0) / len(im_list)  # mean probabilities across images
    args = pred_prob.argsort()[-5:][::-1]  # top5 predicted labels
    pred_lab = args
    return np.array(pred_lab), np.array(pred_prob[args])
