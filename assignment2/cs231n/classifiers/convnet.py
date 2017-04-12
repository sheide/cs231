import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, use_batchnorm=False, use_dropout=False,
                 hidden_dim=100, num_classes=10, crp_n=3, aff_n=3, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.input_dim = input_dim
        self.num_filter = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.crp_n = crp_n
        self.aff_n = aff_n
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        stride = 1
        pad = (filter_size - 1) / 2
        pool_height = 2
        pool_width = 2
        pool_stride = 2

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        #crpN - number of [conv-relu-pool] layers
        #affN - number of affine layers

        (C, H, W) = input_dim
        depth = C
        for i in np.arange(crp_n) + 1:
            W_i = 'W' + str(i)
            b_i = 'b' + str(i)
            gamma_i = 'gamma' + str(i)
            beta_i = 'beta' + str(i)
            self.params[W_i] = np.random.normal(0.0, weight_scale, (
                num_filters, depth, filter_size, filter_size))  # prepend number of filters to shape
            self.params[b_i] = np.zeros(num_filters)
            self.params[gamma_i] = np.zeros(num_filters) + 1
            self.params[beta_i] = np.zeros(num_filters)

            depth = num_filters

            #after conv layer
            H = 1 + (H - filter_size + 2 * pad) / stride
            W = 1 + (W - filter_size + 2 * pad) / stride

            if i%2 == 0:
                #apply pool layer every two conv layers
                H = 1 + (H - pool_height) / pool_stride
                W = 1 + (W - pool_width) / pool_stride

            print 'H: ', H


        #volume of last convolutional layer
        input_affine = num_filters * H * W
        for i in np.arange(aff_n) + crp_n + 1:
            W_i = 'W' + str(i)
            b_i = 'b' + str(i)
            gamma_i = 'gamma' + str(i)
            beta_i = 'beta' + str(i)
            self.params[W_i] = np.random.normal(0.0, weight_scale, (input_affine, hidden_dim))
            self.params[b_i] = np.zeros(hidden_dim)
            self.params[gamma_i] = np.zeros(hidden_dim) + 1
            self.params[beta_i] = np.zeros(hidden_dim)
            input_affine = hidden_dim


        lastIndex = crp_n + aff_n + 1
        self.params['W' + str(lastIndex)] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b' + str(lastIndex)] = np.zeros(num_classes)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in np.arange(lastIndex) + 1]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        # pass conv_param to the forward pass for the convolutional layer
        #filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        cache = {}
        for i in np.arange(self.crp_n) + 1:
            index = str(i)
            W_i = 'W' + index
            b_i = 'b' + index
            gamma_i = 'gamma' + str(i)
            beta_i = 'beta' + str(i)
            if i % 2 == 0:
                X, cache[i] = conv_bn_relu_pool_forward(X, self.params[W_i], self.params[b_i], conv_param, pool_param,
                                                    self.params[gamma_i], self.params[beta_i], self.bn_params[i])
            else:
                X, cache[i] = conv_bn_relu_forward(X, self.params[W_i], self.params[b_i], conv_param,
                                                    self.params[gamma_i], self.params[beta_i], self.bn_params[i])


        for i in np.arange(self.aff_n) + self.crp_n + 1:
            index = str(i)
            W_i = 'W' + index
            b_i = 'b' + index
            gamma_i = 'gamma' + str(i)
            beta_i = 'beta' + str(i)
            X, cache[i] = affine_bn_relu_forward(X, self.params[W_i], self.params[b_i],
                                                 self.params[gamma_i], self.params[beta_i], self.bn_params[i])
            #X, cache[i] = affine_relu_forward(X, self.params[W_i], self.params[b_i])

        lastIndex = str(self.aff_n + self.crp_n + 1)
        scores, cache[lastIndex] = affine_forward(X, self.params['W' + lastIndex], self.params['b' + lastIndex])


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)

        if self.reg > 0:
            for name, W in self.params.iteritems():
                if name.startswith('W'):
                    loss += 0.5 * self.reg * (np.sum(W * W))


        dout, grads['W' + lastIndex], grads['b' + lastIndex] = affine_backward(dout, cache[lastIndex])

        for i in reversed(np.arange(self.aff_n) + self.crp_n + 1):
            W_i = 'W' + str(i)
            b_i = 'b' + str(i)
            gamma_i = 'gamma' + str(i)
            beta_i = 'beta' + str(i)
            dout, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = affine_bn_relu_backward(dout, cache[i])

        for i in reversed(np.arange(self.crp_n) + 1):
            W_i = 'W' + str(i)
            b_i = 'b' + str(i)
            gamma_i = 'gamma' + str(i)
            beta_i = 'beta' + str(i)
            if (i % 2 == 0):
                dout, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = conv_bn_relu_pool_backward(dout, cache[i])
            else:
                dout, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = conv_bn_relu_backward(dout, cache[i])


        if self.reg > 0:
            for name, W in grads.iteritems():
                if name.startswith('W'):
                    grads[name] += self.reg * W


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads