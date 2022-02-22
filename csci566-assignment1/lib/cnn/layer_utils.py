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
        input = np.pad(img.copy(), pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        #iterate over output dimensions, moving by self.stride to create the output
        output = np.zeros(output_shape)
        for i in range(input.shape[0]):
            for h in range(output_height):
                for w in range(output_width):
                    local_segment = input[i, h*self.stride:h*self.stride+self.kernel_size, w*self.stride:w*self.stride+self.kernel_size, :]
                    for c in range(self.number_filters):
                        output[i, h, w, c] = np.sum(local_segment * self.params[self.w_name][:,:,:,c]) + self.params[self.b_name][c]
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
        _, output_height, output_width, _ = self.get_output_size(img.shape)
        input = np.pad(img.copy(), pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        dimg = np.zeros_like(input)
        self.grads[self.w_name] = np.zeros_like(self.params[self.w_name])
        self.grads[self.b_name] = np.zeros_like(self.params[self.b_name])
        for i in range(input.shape[0]):
            for h in range(output_height):
                for w in range(output_width):
                    local_segment = input[i, h * self.stride:h * self.stride + self.kernel_size,
                                    w * self.stride:w * self.stride + self.kernel_size, :]
                    for c in range(self.number_filters):
                        dimg[i, h * self.stride : h * self.stride + self.kernel_size, w * self.stride : w * self.stride + self.kernel_size, :] += self.params[self.w_name][:, :, :, c] * dprev[i, h, w, c]
                        self.grads[self.w_name][:, :, :, c] += local_segment * dprev[i, h, w, c]
                        self.grads[self.b_name][c] += dprev[i, h, w, c]
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
