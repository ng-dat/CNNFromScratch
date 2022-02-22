from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 3, stride=2, padding=0, name="conv"),
            MaxPoolingLayer(pool_size=1, stride=1, name="pool"),
            flatten(name="flat"),
            fc(27, 5, 2e-2, name="fc"),
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 5, 8, stride=1, padding=0, name="conv1"),
            MaxPoolingLayer(pool_size=2, stride=2, name="pool1"),
            ConvLayer2D(8, 3, 8, stride=1, padding=0, name="conv2"),
            MaxPoolingLayer(pool_size=2, stride=2, name="pool2"),
            flatten(name="flat"),
            fc(288, 128, 2e-2, name="fc1"),
            leaky_relu(name="relu"),
            fc(128, 10, 2e-2, name="fc2"),
            ########### END ###########
        )
