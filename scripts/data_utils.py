# -*- coding: utf-8 -*-
"""
Miscellanous functions manage data in image recognition.
network.

Author: Ignacio Heredia
Date: September 2016
"""

import os
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import threading
import Queue


def data_splits(im_dir='/media/ignacio/Datos/plant_net/images_ori', tag=False):
    """
    Load the training and validation arrays from the train.txt and val.txt files.
    Lines of txt files have the following format:
    'relative_path_to_image' 'image_tag'[optional] 'image_label_number'

    Parameters
    ----------
    im_dir : str
        Absolute path to the image folder.
    tag : bool
        Presence or absence of tag in txt files.

    Returns
    -------
    X_train, X_val : array of strs
        First colunm: Contains 'absolute_path_to_file' to images.
        Second column [optional]: Contains tag of image
    y_train, y_val : array of int32
        Image label number
    metadata : array of strs
        Label names array.

    """
    homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    splits_dir = os.path.join(homedir, 'data', 'data_splits')
    print("Loading data...")
    file_list = os.listdir(splits_dir)

    # Metadata labels
    metadata = np.genfromtxt(os.path.join(splits_dir, 'synsets.txt'), dtype='str', delimiter='/n')

    # Training splits
    train = np.genfromtxt(os.path.join(splits_dir, 'train.txt'), dtype='str', delimiter=' ')
    y_train = train[:, -1].astype(np.int32)
    if tag:
        X_train = train[:, 0:2].astype(object)
        X_train[:, 0] = np.array([os.path.join(im_dir, i) for i in X_train[:, 0]])
    else:
        X_train = np.array([os.path.join(im_dir, i) for i in train[:, 0]])

    # Validation splits
    if 'val.txt' in file_list:
        val = np.genfromtxt(os.path.join(splits_dir, 'val.txt'), dtype='str', delimiter=' ')
        y_val = val[:, -1].astype(np.int32)
        if tag:
            X_val = val[:, 0:2].astype(object)
            X_val[:, 0] = np.array([os.path.join(im_dir, i) for i in X_val[:, 0]])
        else:
            X_val = np.array([os.path.join(im_dir, i) for i in val[:, 0]])
    else:
        print 'Training with no validation data.'
        X_val, y_val = None, None

    return X_train, y_train, X_val, y_val, metadata


def data_augmentation(im_list, mode='standard', tag=False, params=None, im_size=224,
                      filemode='local', mean_RGB=None):
    """
    Perform data augmentation on some image list using PIL.

    Parameters
    ----------
    im_list : array of strings
        Array where the first column is image_path (or image_url). Optionally
        a second column can be the tags of the image.
        Shape (N,) or (N,2)
    tag : bool
        If True then im_list is an array with 1st column is filepath
        and 2nd column is image_tag (eg. for plants we had habit, fruit,
        flower, bark). You can the manually modify the data_augmentation
        function to choose which transformations are to be performed to each tag.
        In a future we could pass a params dict along with each tag.
    params : dict or None
        Mandatory keys:
        - mirror (bool): allow 50% random mirroring.
        - rescale ([0,1] float): randomly rescale image parameter.
        - crop_size ([0,1] float): random crop of size crop_size*original_size.
        - zoom ([0,1] float): the zoom will be implicitely by rescaling the crop_size.
    mode : {'standard', 'minimal', 'test', None}
        We overwrite the params dict with some defaults augmentation parameters
        - 'minimal': no data augmentation, just resizing
        - 'standard': tipical parameters for data augmentation during training
        - 'test': minimized data augmentation for testing
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
    rot_ang = [0, 90, 180, 270]
    batch = []
    if tag:
        tag_list = im_list[:, 1]
        im_list = im_list[:, 0]
    if mode == 'minimal':
        params = {'mirror': False, 'rescale': False, 'crop_size': False}
    if mode == 'standard':
        params = {'mirror': True, 'rescale': 0.3, 'zoom': 0.3, 'crop_size': 1.}
    if mode == 'test':
        params = {'mirror': True, 'rescale': 0.1, 'zoom': 0.1, 'crop_size': .9}
    for i, filename in enumerate(im_list):
        if filemode == 'local':
            im = Image.open(filename)
            im = im.convert('RGB')
        elif filemode == 'url':
            filename = BytesIO(requests.get(filename).content)
            im = Image.open(filename)
            im = im.convert('RGB')
        if params['mirror'] and np.random.random() > 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if params['mirror'] and tag and tag_list[i] != 'habit':
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            rot = np.random.choice(rot_ang)
            if rot == 90:
                im = im.transpose(Image.ROTATE_90)
            if rot == 180:
                im = im.transpose(Image.ROTATE_180)
            if rot == 270:
                im = im.transpose(Image.ROTATE_270)
        if params['rescale']:
            rescale = params['rescale']
            new_scale = np.random.uniform(low=1.-rescale, high=1.+rescale, size=2)
            im = im.resize((im.size * new_scale).astype(int))
        if params['crop_size']:
            zoom = np.random.rand() * params['zoom']
            crop_size = params['crop_size'] * (1.-zoom)
            ly, lx = im.size
            min_side = min([ly, lx])
            if crop_size == 1:
                crop_size -= 1e-10  # avoid low=high problem of randint generator
            if ly > lx:
                rand_x = np.random.randint(low=0, high=lx*(1.-crop_size))
                rand_y = np.random.randint(low=0, high=ly-lx*crop_size)
            else:
                rand_x = np.random.randint(low=0, high=lx-ly*crop_size)
                rand_y = np.random.randint(low=0, high=ly*(1.-crop_size))
            rand_xy = np.array([rand_y, rand_x])
            im = im.crop(np.concatenate((rand_xy, rand_xy+crop_size*min_side)))
        im = im.resize((im_size, im_size))
        batch.append(np.array(im))  # shape (N, 224, 224, 3)

    batch = np.array(batch) - mean_RGB[None, None, None, :]  # mean centering
    batch = batch.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
    batch = batch[:, ::-1, :, :]  # switch from RGB to BGR
    return batch.astype(np.float32)


#def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#    """
#    Returns generator of batches of inputs and targets.
#    """
#    assert len(inputs) == len(targets)
#    assert len(inputs) >= batchsize
#    if shuffle:
#        indices = np.arange(len(inputs))
#        np.random.shuffle(indices)
#    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#        if shuffle:
#            excerpt = indices[start_idx:start_idx + batchsize]
#        else:
#            excerpt = slice(start_idx, start_idx + batchsize)
#        if targets.shape < 2: targets = targets.reshape(-1,1)
#        yield inputs[excerpt], targets[excerpt]


def buffered_gen_threaded(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate thread. Beware of the GIL!
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    Author: Benanne (github-kaggle/benanne/ndsb)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = Queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.

    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator

    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, **augmentation_params):
    """
    Returns generator of batches of inputs and targets via buffer.
    Therefore we perform dataaugmnetaion for the next batch on CPU while the GPU is
    computing the current batch.
    """
    assert len(inputs) == len(targets)
    assert len(inputs) >= batchsize
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    def gen(inputs, targets, batchsize, **augmentation_params):
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if targets.shape < 2:
                targets = targets.reshape(-1, 1)
            X, y = data_augmentation(inputs[excerpt], **augmentation_params), targets[excerpt]
            yield X, y
    return buffered_gen_threaded(gen(inputs, targets, batchsize, **augmentation_params))


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


def meanRGB(im_list):
    """
    Returns the mean and std RGB values for the whole dataset.
    For the plantnet we have:
        mean_RGB = np.array([ 107.59348955,  112.1047813 ,   80.9982362 ])
        std_RGB = np.array([ 52.78326119,  50.56163087,  50.86486131])

    Parameters
    ----------
    im_list : array of strings
        Array where the first column is image_path (or image_url). Shape (N,).

    """
    print 'Computing mean RGB pixel ...'
    mean, std = np.zeros(3), np.zeros(3)
    for i, filename in enumerate(im_list):
        # Write completion bar
        n = 1. * i / len(im_list)
        sys.stdout.write('\r')
        sys.stdout.write("[{:20}] {}%".format('='*int(n/0.05), int(100*n)))
        sys.stdout.flush()
        # Process image
        im = np.array(Image.open(filename)).reshape(-1, 3)
        mean += np.mean(im, axis=0)
        std += np.std(im, axis=0)
    print ''
    mean, std = mean / len(im_list), std / len(im_list)
    return mean, std
