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
            #Convolutional --> Maxpool --> flatten --> fc 
            #input_channels, kernel_size, number_filters, stride=1, padding=0, init_scale=.02, name="conv"
            ConvLayer2D(input_channels=3,kernel_size=3,number_filters=3,name="conv"),
            MaxPoolingLayer(pool_size=2,stride=2,name="maxpool"),
            flatten(name="flat"),
            fc(27, 5, 0.02,name="fc")
            
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            # ConvLayer2D(input_channels=3,kernel_size=3,padding=1,number_filters=12,name="conv"),
            # MaxPoolingLayer(pool_size=2,stride=2,name="maxpool"),
            # ConvLayer2D(input_channels=12,kernel_size=3,padding=1,number_filters=6,name="conv2"),
            # MaxPoolingLayer(pool_size=2,stride=2,name="maxpool2"),
            # flatten(name="flat"),
            # fc(384, 10, 0.02,name="fc")


            ConvLayer2D(input_channels=3,kernel_size=3,number_filters=16,name="conv"),
            leaky_relu(name="relu1"),
            MaxPoolingLayer(pool_size=2,stride=2,name="maxpool"),
            ConvLayer2D(input_channels=16,kernel_size=3,number_filters=32,name="conv2"),
            leaky_relu(name="relu2"),
            MaxPoolingLayer(pool_size=2,stride=2,name="maxpool2"),
            flatten(name="flat"),
            fc(1152, 164, 0.02,name="fc"),
            leaky_relu(name="relu3"),
            fc(164, 10, 0.02,name="fc2")

            # ConvLayer2D(input_channels=3,kernel_size=5,padding=2,number_filters=6,name="conv"),
            # leaky_relu(name="relu1"),
            # MaxPoolingLayer(pool_size=2,stride=2,name="maxpool"),
            # ConvLayer2D(input_channels=12,kernel_size=5,number_filters=16,name="conv2"),
            # leaky_relu(name="relu2"),
            # MaxPoolingLayer(pool_size=2,stride=2,name="maxpool2"),
            # flatten(name="flat"),
            # fc(1024, 120, 0.02,name="fc1"),
            # fc(120, 84, 0.02,name="fc2"),
            # fc(84, 10, 0.02,name="fc3")

            ########### END ###########
        )
