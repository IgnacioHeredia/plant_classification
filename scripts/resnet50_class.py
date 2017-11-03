"""
Resnet50_class

Author: Ignacio Heredia
Date: October 2016

Description:
Class for training a resnet50 for a new dataset by finetuning the weights
already pretrained with ImageNet.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import json
import collections
import inspect
import os
import sys
homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(homedir, 'scripts'))
from data_utils import iterate_minibatches, data_augmentation

import theano
import theano.tensor as T
import lasagne
from models.resnet50 import build_model

theano.config.floatX = 'float32'


class prediction_net(object):

    def __init__(self, output_dim=3680, lr=1e-3, lr_decay=0.1,
                 lr_decay_rate=None, lr_decay_schedule=[0.7, 0.9],
                 finetuning=1e-3, reg=1e-4, num_epochs=50,
                 batchsize=32):
        """
        Parameters
        ----------
        output_dim : int
            output dimension (number of possible output classes)
        lr : float
            Base learning rate (1e-3 is the default for Adam update rule)
        lr_decay : float
            It's the ratio (new_lr / old_lr)
        lr_decay_rate : float, None
            Update the lr after this number of epochs
        lr_decay_schedule : list, numpy array, None
            Update at this % of training.
            Eg. [0.7,0.9] and 50 epochs --> update at epochs 35 and 45
            This variable overwrites lr_decay_rate.
        finetuning : float
            Finetuning coefficient for learning the first layers
            Eg. layer_lr = finetuning * lr
        reg : float
            Regularization parameter
        num_epochs : int
            Number of epochs for training
        batchsize: int
            Size of each training batch (should fit in GPU)

        """
        self.output_dim = output_dim
        self.lr_init = lr
        self.lr = theano.shared(np.float32(lr))
        self.lr_decay = np.float32(lr_decay)
        if lr_decay_schedule is not None:
            self.lr_decay_schedule = (np.array(lr_decay_schedule) * num_epochs).astype(np.int)
        else:
            self.lr_decay_schedule = np.arange(0, num_epochs, lr_decay_rate)[1:].astype(np.int)
        self.reg = reg
        self.num_epochs = num_epochs
        self.batchsize = batchsize
        self.finetuning = finetuning

    def build_and_train(self, X_train, y_train, X_val=None, y_val=None,
                        display=False, save_model=True, aug_params=None):
        """
        Builds the model and runs the training loop.

        Parameters
        ----------
        X_train : numpy array
            Training data
        y_train : numpy array
            Training targets.
        X_val : numpy array, None, optional
            Validation data
        y_val : numpy array, None, optional
            Validation targets
        Display : bool, optional
            Display on-fly plots of training and validation results.
        Save_model : bool, optional
            Save model weights.
        aug_params : dict, None, optional
            Dict containing the data augmentation parameters.

        Returns
        -------
        Test function of the net.

        """
        # ======================================================================
        # Model compilation
        # ======================================================================

        print("Building model and compiling functions...")

        # Create Theano variables for input and target minibatch
        input_var = T.tensor4('X', dtype=theano.config.floatX)  # shape (batchsize,3,224,224)
        target_var = T.ivector('y')  # shape (batchsize,)

        # Load model weights and metadata
        d = pickle.load(open(os.path.join(homedir, 'data', 'pretrained_weights', 'resnet50.pkl')))

        # Build the network and fill with pretrained weights except for the last fc layer
        net = build_model(input_var, self.output_dim)
        lasagne.layers.set_all_param_values(net['pool5'], d['values'][:-2])

        # create loss function and accuracy
        prediction = lasagne.layers.get_output(net['prob'])
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean() + self.reg * lasagne.regularization.regularize_network_params(
                             net['prob'], lasagne.regularization.l2)
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)

        # Create parameter update expressions with fine tuning
        updates = {}
        for name, layer in net.items():
            layer_params = layer.get_params(trainable=True)
            if name == 'fc1000' or name == 'prob':
                layer_lr = self.lr
            else:
                layer_lr = self.lr * self.finetuning
            layer_updates = lasagne.updates.adam(loss, layer_params, learning_rate=layer_lr)
            updates.update(layer_updates)
        updates = collections.OrderedDict(updates)

        # Create a loss expression for validation/testing.
        test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        # Compile training and validation functions
        train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        test_fn = theano.function([input_var], test_prediction)

        # ======================================================================
        # Training routine
        # ======================================================================

        print("Starting training...")
        track = {'train_err': [], 'train_acc': [], 'val_err': [], 'val_acc': []}

        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            line1, = ax1.plot([], [], 'r-')
            line2, = ax2.plot([], [], 'r-')

            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Training loss')
            ax1.set_yscale('log')
            ax1.set_title('Training loss')

            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Validation loss')
            ax2.set_yscale('log')
            ax2.set_title('Validation loss')

        # Batchsize and augmentation parameters
        if aug_params is None:
            aug_params = {}
        train_batchsize = min(len(y_train), self.batchsize)
        train_aug_params = aug_params.copy()
        train_aug_params.update({'mode': 'standard'})
        if X_val is not None:
            val_batchsize = min(len(y_val), self.batchsize)
            val_aug_params = aug_params.copy()
            val_aug_params.update({'mode': 'minimal', 'tags': None})

        for epoch in range(self.num_epochs):

            start_time = time.time()

            # Learning rate schedule decay
            if epoch in self.lr_decay_schedule:
                self.lr.set_value(self.lr.get_value() * self.lr_decay)
                print('############# Leaning rate: {} ####################').format(self.lr.get_value())

            # Full pass over training data
            train_err, train_batches = 0, 0
            for batch in iterate_minibatches(X_train, y_train, train_batchsize, shuffle=True, **train_aug_params):
                inputs, targets = batch[0], batch[1]
                tmp_train_err, tmp_train_acc = train_fn(inputs, targets)
                track['train_err'].append(tmp_train_err)
                track['train_acc'].append(tmp_train_acc)
                train_err += tmp_train_err
                train_batches += 1
                print 'Training epoch {} - {:.1f}% completed | Loss: {:.4f} ; Accuracy: {:.1f}%'.format(epoch, train_batches*self.batchsize*100./len(y_train), float(tmp_train_err), float(tmp_train_acc)*100)
                if np.isnan(train_err):
                    print('Your net exploded, try decreasing the learning rate.')
                    return None

            # Full pass over the validation data (if any)
            if X_val is not None:
                val_err, val_batches = 0, 0
                for batch in iterate_minibatches(X_val, y_val, val_batchsize, shuffle=False, **val_aug_params):
                    inputs, targets = batch[0], batch[1]
                    tmp_val_err, tmp_val_acc = val_fn(inputs, targets)
                    track['val_err'].append(tmp_val_err)
                    track['val_acc'].append(tmp_val_acc)
                    val_err += tmp_val_err
                    val_batches += 1
                    print 'Validation epoch {} - {:.1f}% completed | Loss: {:.4f} ; Accuracy: {:.1f}%'.format(epoch, val_batches*self.batchsize*100./len(y_val), float(tmp_val_err), float(tmp_val_acc)*100)

            # Print the results for this epoch
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            if X_val is not None:
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            # Display training and validation accuracy in plot
            if display:

                line1.set_xdata(np.append(line1.get_xdata(), epoch))
                line1.set_ydata(np.append(line1.get_ydata(), train_err / train_batches))
                ax1.relim(), ax1.autoscale_view()

                if X_val is not None:
                    line2.set_xdata(np.append(line2.get_xdata(), epoch))
                    line2.set_ydata(np.append(line2.get_ydata(), val_err / val_batches))
                    ax2.relim(), ax2.autoscale_view()

                fig.canvas.draw()

        # Save training information and net parameters
        print("Saving the model parameters and training information ...")

        train_info = {'training_params': {'output_dim': self.output_dim,
                                          'lr_init': self.lr_init,
                                          'lr_decay': float(self.lr_decay),
                                          'lr_schedule': self.lr_decay_schedule.tolist(),
                                          'reg': self.reg,
                                          'num_epochs': self.num_epochs,
                                          'batchsize': self.batchsize,
                                          'finetuning': self.finetuning}}

        a = inspect.getargspec(data_augmentation)
        augmentation_params = dict(zip(a.args[-len(a.defaults):], a.defaults))  # default augmentation params
        augmentation_params.update(aug_params)  # update with user's choice
        for k, v in augmentation_params.items():
            if type(v) == np.ndarray:
                augmentation_params[k] = np.array(v).tolist()
        train_info.update({'augmentation_params': augmentation_params})

        for k, v in track.items():
            track[k] = np.array(v).tolist()
        train_info.update(track)

        if save_model:
            filename = 'resnet50_' + str(self.output_dim) + 'classes_' + str(self.num_epochs) + 'epochs'
            with open(os.path.join(homedir, 'scripts', 'training_info', filename + '.json'), 'w') as outfile:
                json.dump(train_info, outfile)
            np.savez(os.path.join(homedir, 'scripts', 'training_weights', filename + '.npz'), *lasagne.layers.get_all_param_values(net['prob']))

        return test_fn
