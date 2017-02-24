# -*- coding: utf-8 -*-
"""
Miscellanous functions load model for image recognition.

Author: Ignacio Heredia
Date: November 2016
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))) 

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
