#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Saliency maps using gradients
Author: Ignacio Heredia
Date: November 2017

Computed saliency maps using vanilla/guided backprop gradients + smoothgrad. 
Some pieces of code have been shamelessly borrowed from Jan Schluter tutorial 
(https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb).

The configurations I have found to work the best HERE:
    1) Use modified backprop and vanilla_map(magnitude=False)
    2) Use vanilla gradient and smoothgrad(magnitude=True)
Other configurations are very noisy or produce a lot of checkerboard patterns.
"""

import numpy as np
import sys
import json
import os.path as op
homedir = op.abspath(op.join(__file__, op.pardir, op.pardir, op.pardir, op.pardir))
sys.path.append(op.join(homedir, 'scripts'))
from PIL import Image
import matplotlib.pylab as plt

import theano
import theano.tensor as T
import lasagne


# Load the image
img_path = '/home/ignacio/castanus.jpg' #path to image
#img = os.path.join(homedir, 'data', 'demo-images', 'image1.jpg')
X, Y = 224, 224 #network's input size
img = Image.open(img_path)
img = img.resize((X,Y))

# Load the model
modelname = 'resnet50_6182classes_100epochs'
metadata = np.genfromtxt(op.join(homedir, 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
                       
# Load training info
info_file = op.join(homedir, 'scripts', 'training_info', modelname + '.json')
with open(info_file) as datafile:
    train_info = json.load(datafile)
output_dim = train_info['training_params']['output_dim']
mean_RGB = np.array(train_info['augmentation_params']['mean_RGB'])
if np.any(mean_RGB) == None:
    mean_RGB = np.array([107.59348955, 112.1047813, 80.9982362])


def preprocess_batch(im_list):
    """
    Minimal preprocess of the image before feeding to network
    """
    im_list = np.array(im_list) - mean_RGB[None, None, None, :]  # mean centering
    im_list = im_list.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
    im_list = im_list[:, ::-1, :, :]  # switch from RGB to BGR   
    return im_list.astype(np.float32)


def guided_backprop(net):
    """
    Modify the gradient of the relu to implement guided backpropagation
    """
    
    class ModifiedBackprop(object):

        def __init__(self, nonlinearity):
            self.nonlinearity = nonlinearity
            self.ops = {}  # memoizes an OpFromGraph instance per tensor type
    
        def __call__(self, x):
            # OpFromGraph is oblique to Theano optimizations, so we need to move
            # things to GPU ourselves if needed.
            if theano.gpuarray.pygpu_activated:
                ctx = theano.gpuarray.basic_ops.infer_context_name(x)
                x = theano.gpuarray.as_gpuarray_variable(x, ctx)
            # We note the tensor type of the input variable to the nonlinearity
            # (mainly dimensionality and dtype); we need to create a fitting Op.
            tensor_type = x.type
            # If we did not create a suitable Op yet, this is the time to do so.
            if tensor_type not in self.ops:
                # For the graph, we create an input variable of the correct type:
                inp = tensor_type()
                # We pass it through the nonlinearity (and move to GPU if needed).
                if theano.gpuarray.pygpu_activated:
                    ctx = theano.gpuarray.basic_ops.infer_context_name(self.nonlinearity(inp))
                    outp = theano.gpuarray.as_gpuarray_variable(self.nonlinearity(inp), ctx)
                # Then we fix the forward expression...
                op = theano.OpFromGraph([inp], [outp])
                # ...and replace the gradient with our own (defined in a subclass).
                op.grad = self.grad
                # Finally, we memoize the new Op
                self.ops[tensor_type] = op
            # And apply the memoized Op to the input we got.
            return self.ops[tensor_type](x)
    
    class GuidedBackprop(ModifiedBackprop):
        
        def grad(self, inputs, out_grads):
            (inp,) = inputs
            (grd,) = out_grads
            dtype = inp.dtype
            return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)
    
    relu = lasagne.nonlinearities.rectify
    relu_layers = [layer for layer in lasagne.layers.get_all_layers(net['prob'])
                   if getattr(layer, 'nonlinearity', None) is relu]
    modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
    for layer in relu_layers:
        layer.nonlinearity = modded_relu

    return net    


def load_model(modelweights, output_dim, use_guided_backprop=True):
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
    
    if use_guided_backprop:
        net = guided_backprop(net)
        
    return net


def compile_saliency_function(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = net['input'].input_var
    outp = lasagne.layers.get_output(net['fc1000'], deterministic=True)
    max_outp = T.max(outp, axis=1) #we select the first prediction
    saliency = theano.grad(max_outp.sum(), wrt=inp) #we sum because cost must be scalar
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])


def vanilla_map(img, saliency_func, magnitude=False):
    im_list = [np.array(img)]
    im_list = preprocess_batch(im_list)
    saliency_map, max_class = saliency_func(im_list) #shape (N,3,224,224)

    if magnitude:
        saliency_map = saliency_map ** 2    
    saliency_map = np.mean(saliency_map, axis=0) #mean across batch (1 image)
    saliency_map = np.abs(saliency_map)
    saliency_map = np.sum(saliency_map, axis=0) #sum accross channels
    
    vmax = np.percentile(saliency_map, 99)
    vmin = np.amin(saliency_map)
    saliency_map = np.clip((saliency_map - vmin)/(vmax - vmin), 0, 1)

    return saliency_map


def smoothgrad_map(img, saliency_func, N=40, noise=0.2, magnitude=False):
    """
    Smoothgrad function
    It averages over several saliency maps created with different noise to obtain
    a cleaner image.
    
    Parameters
    ----------
    N : int
        Number of maps to average over
    noise : float in range [0,1]
        Relative noise level. Values beteween 10-30% ussually work well.
    magnitude : bool
        Compute the square of the grads. Default to False although in the original 
        paper was set to True but here seem to favour checkerboard patterns.
    """
    img_arr = np.array(img)
    mean = np.array([0, 0, 0])
    std = noise * (np.amax(img_arr, axis=(0,1)) - np.amin(img_arr, axis=(0,1))) #noise adapted to channel variance
    
    im_list = [np.array(img)] * N
    im_list = np.array(im_list) + np.random.normal(mean, std, (N, 224, 224, 3))
    im_list = np.clip(im_list, 0, 255)
    im_list = preprocess_batch(im_list)
    saliency_map, max_class = saliency_func(im_list)

    if magnitude:
        saliency_map = saliency_map ** 2    
    saliency_map = np.mean(saliency_map, axis=0) #mean across batch
    saliency_map = np.abs(saliency_map)
    saliency_map = np.sum(saliency_map, axis=0) #sum accross channels
    
    vmax = np.percentile(saliency_map, 99)
    vmin = np.amin(saliency_map)
    saliency_map = np.clip((saliency_map - vmin)/(vmax - vmin), 0, 1)
                           
    return saliency_map
    

def plots(saliency_map):
    """
	Different plots for saliency maps
    
    Parameters
    ----------
    saliency_map : np.array shape(x,y)
        Map to be plotted
    """
    
    # Plot mask and original image
    fig, ax = plt.subplots(1,2, figsize=(12, 4))
    cf = ax[0].imshow(saliency_map, clim=[np.amin(saliency_map), np.amax(saliency_map)])
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title('Saliency map')
    x,y = saliency_map.shape
    ori_patch = np.array(img)[(X-x)/2:(X+x)/2, (Y-y)/2:(Y+y)/2] #central patch
    ax[1].imshow(ori_patch)
    ax[1].set_title('Original image')
    
    # Plot contour plots of the mask over the image
    plt.figure()
    plt.imshow(ori_patch)
    CS = plt.contour(np.arange(saliency_map.shape[0]), np.arange(saliency_map.shape[1]), saliency_map)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour plot')    
    
    # Plot RGBA image (soft mask)
    plt.figure()
    saliency_map -= np.amin(saliency_map)
    saliency_map /= np.amax(saliency_map)
    alpha_channel = np.round(saliency_map*255)
    rgba_arr = np.dstack((ori_patch, alpha_channel)).astype(np.uint8)
    plt.imshow(rgba_arr)
    plt.title('RGBA image')


# Load net weights
weights_path = op.join(homedir, 'scripts', 'training_weights', modelname + '.npz')
net = load_model(weights_path, output_dim=output_dim, use_guided_backprop=True)
saliency_func = compile_saliency_function(net)  

if __name__ == "__main__":
    
	##Plot single method
    #saliency_map = smoothgrad_map(img, saliency_func)
    #plots(saliency_map)
    
    # Compare all the methods
    img_path = '/home/ignacio/castanus.jpg' #path to image
    img = Image.open(img_path)
    img = img.resize((224, 224))
    
    fig, ax = plt.subplots(2,4)
    ax = ax.flatten()
    plt_count = 0
    for i in ['vanilla grad', 'guided bp']:
        use_guided_backprop = {'vanilla grad': False, 'guided bp': True}[i]
        net = load_model(weights_path, output_dim=output_dim, use_guided_backprop=use_guided_backprop)
        saliency_func = compile_saliency_function(net)
        for j in ['no smooth', 'smooth']:
            f = {'no smooth': vanilla_map, 'smooth': smoothgrad_map}[j]
            for k in ['no_mag', 'mag']:
                use_magnitude = {'no_mag': False, 'mag':True}[k]
                saliency_map = f(img, saliency_func, magnitude=use_magnitude)
                ax[plt_count].imshow(saliency_map)
                ax[plt_count].set_title('{} | {} | {}'.format(i,j,k))
                plt_count += 1
        
