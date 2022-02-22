from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        output_shape[0] = input_size[0]
        output_shape[1] = int(1.0 * (input_size[1] + self.padding * 2 - self.kernel_size) / self.stride) + 1
        output_shape[2] = int(1.0 * (input_size[2] + self.padding * 2 - self.kernel_size) / self.stride) + 1
        output_shape[3] = self.number_filters
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        #pad the input image according to self.padding (see np.pad)
        input = np.pad(img, pad_width=((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))

        #iterate over output dimensions, moving by self.stride to create the output
        W = np.transpose(self.params[self.w_name], axes=[3,2,0,1])
        b = self.params[self.b_name]
        X = np.transpose(input, axes=[0, 3, 1, 2])
        batch_size, input_height, input_width, _ = input.shape

        shape = (self.input_channels, self.kernel_size, self.kernel_size, batch_size, output_height, output_width)
        strides = (input_height * input_width, input_width, 1, self.input_channels * input_height * input_width, self.stride * input_width, self.stride)
        strides = input.itemsize * np.array(strides)
        input_stride = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
        input_cols = np.ascontiguousarray(input_stride)
        input_cols.shape = (self.input_channels * self.kernel_size * self.kernel_size, batch_size * output_height * output_width)

        z = W.reshape(self.number_filters, -1).dot(input_cols) + b.reshape(-1, 1)
        z.shape = (self.number_filters, batch_size, output_height, output_width)
        output = np.transpose(z, axes=[1, 2, 3, 0])
        output = np.ascontiguousarray(output)

        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        input = np.pad(img, pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        _, output_height, output_width, _ = dprev.shape
        batch_size, input_height, input_width, _ = input.shape
        W = np.transpose(self.params[self.w_name], axes=[3, 2, 0, 1])

        shape = (self.input_channels, self.kernel_size, self.kernel_size, batch_size, output_height, output_width)
        strides = (input_height * input_width, input_width, 1, self.input_channels * input_height * input_width,
                   self.stride * input_width, self.stride)
        strides = img.itemsize * np.array(strides)
        input_stride = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
        input_cols = np.ascontiguousarray(input_stride)
        input_cols.shape = (self.input_channels * self.kernel_size * self.kernel_size, batch_size * output_height * output_width)

        self.grads[self.b_name] = np.sum(dprev, axis=(0, 1, 2))
        dprev_reshaped = np.transpose(dprev, axes=[3, 0, 1, 2]).reshape(self.number_filters, -1)
        dW = dprev_reshaped.dot(input_cols.T).reshape((self.number_filters, self.input_channels, self.kernel_size, self.kernel_size))
        self.grads[self.w_name] = np.transpose(dW, axes=[2, 3, 1, 0])

        dinput_cols = W.reshape(self.number_filters, -1).T.dot(dprev_reshaped)
        dinput_cols.shape = shape
        dimg = np.zeros(input.shape)
        for n in range(batch_size):
            for c in range(self.input_channels):
                for kernel_h in range(self.kernel_size):
                    for kernel_w in range(self.kernel_size):
                        for out_h in range(output_height):
                            for out_w in range(output_width):
                                dimg[n, self.stride * out_h + kernel_h, self.stride * out_w + kernel_w, c] += dinput_cols[c, kernel_h, kernel_w, n, out_h, out_w]
        dimg = dimg[:, self.padding:-self.padding, self.padding: -self.padding, :]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        output_shape = [img.shape[0], int(1.0 + (img.shape[1] - self.pool_size) / self.stride),
                        int(1.0 + (img.shape[2] - self.pool_size) / self.stride), img.shape[3]]
        output = np.zeros(output_shape)
        for i in range(img.shape[0]):
            for h in range(output_shape[1]):
                for w in range(output_shape[2]):
                    local_segment = img[i, h * self.stride:h * self.stride + self.pool_size,
                                    w * self.stride:w * self.stride + self.pool_size, :]
                    for c in range(img.shape[3]):
                        output[i, h, w, c] = np.max(local_segment[:,:,c])
        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        for i in range(img.shape[0]):
            for h in range(h_out):
                for w in range(w_out):
                    local_segment = img[i, h * self.stride:h * self.stride + h_pool,
                                    w * self.stride:w * self.stride + w_pool, :]
                    for c in range(img.shape[3]):
                        slice = local_segment[:,:, c]
                        max_mask = slice == np.max(slice)
                        dimg[i, h * self.stride:h * self.stride + h_pool,
                            w * self.stride:w * self.stride + w_pool, c] += np.multiply(max_mask, dprev[i, h, w, c])
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
