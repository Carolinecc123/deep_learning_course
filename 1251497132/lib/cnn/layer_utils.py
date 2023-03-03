from __future__ import division
from __future__ import print_function
import imghdr

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
        output_height = (input_size[1] + 2*self.padding -self.kernel_size) / self.stride + 1
        output_width = (input_size[2] + 2*self.padding -self.kernel_size) / self.stride + 1
        output_shape[0] = int(input_size[0])
        output_shape[1] = int(output_height)
        output_shape[2] = int(output_width)
        output_shape[3] = int(self.number_filters)

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
        img2 = np.pad(img,((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))
        #iterate over output dimensions, moving by self.stride to create the output
        output = np.zeros([output_shape[0],output_shape[1],output_shape[2],output_shape[3]])
        input_blocks = []
        curr_height = img2.shape[1] #height after padding

        for i in range(output_shape[1]): #height
            for j in range(output_shape[2]): #width
                for f in range(output_shape[3]):  #filter number
                    start_h = i * self.stride
                    end_h = start_h + self.kernel_size
                    start_w = j * self.stride
                    end_w = start_w + self.kernel_size
                    #current projection onto current image
                    current_proj = img2[:,start_h:end_h,start_w:end_w,:]
                    #print("current proj", current_proj.shape)
                    w =self.params[self.w_name][:, :, :, f] #current w on filter
                    b = self.params[self.b_name][f] #current b on filter
                    multi = np.multiply(current_proj,w)
                    # print("w", w.shape)
                    # print("b", b)
                    # print("multi shape", multi.shape)
                    curr_sum = np.sum(multi, axis=(1, 2, 3))
                   # print("sum", curr_sum.shape)
                    curr_sum = curr_sum + b.astype(float)
                    #print("curr_sum", curr_sum)
                    output[:,i,j,f] = curr_sum

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        self.meta = img
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
       
        dimg = np.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
        
        self.grads[self.w_name] = np.zeros((self.params[self.w_name].shape))
        self.grads[self.b_name] = np.zeros((self.params[self.b_name].shape))
        #padding
        img2 = np.pad(img,((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))
        dimg2 = np.pad(dimg,((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))

        for i in range(dprev.shape[1]): #height
            for j in range(dprev.shape[2]): #width
                for f in range(dprev.shape[3]): #filter
                    start_h = i * self.stride
                    end_h = start_h + self.kernel_size
                    start_w = j * self.stride
                    end_w = start_w + self.kernel_size
                    #current projection onto current image
                    current_proj = img2[:,start_h:end_h,start_w:end_w,:]
                    #print("current_proj", current_proj.shape)
                    ww =np.expand_dims(self.params[self.w_name][:,:,:,f], axis=3)
                    dprevv = np.expand_dims(dprev[:, i, j, f], axis=0)
                    # print("ww", ww.shape)
                    # print("dprevv", dprevv.shape)
                    # print("dot", np.dot(ww,dprevv).shape)
                    # print(np.dot(ww,dprevv).transpose(3,0,1,2).shape)
                    dimg2_plus = np.dot(ww,dprevv).transpose(3,0,1,2)                  
                    dimg2[:,start_h:end_h,start_w:end_w,:] += dimg2_plus                
                    current_proj2 = current_proj.transpose(1,2,3,0)
                    w_mul = current_proj2 * dprev[:, i, j, f]
                    w_sum = np.sum(w_mul,axis=3)
                    
                    self.grads[self.w_name][:,:,:,f] += w_sum
                    self.grads[self.b_name][f] += np.sum(dprev[:, i, j, f], axis=0)
            dimg_n = dimg[:, :, :, :]
            dimg_n = dimg2[:,self.padding:-self.padding, self.padding:-self.padding, :]

        # for i in range(dprev.shape[0]):  #batch
        #     curr_img = img2[i]
        #     curr_dimg = dimg2[i]
        #     for j in range(dprev.shape[1]): #height
        #         for k in range(dprev.shape[2]): #width
        #             for f in range(dprev.shape[3]): #fiilter
        #                 start_h = j * self.stride
        #                 end_h = start_h + self.kernel_size
        #                 start_w = k * self.stride
        #                 end_w = start_w + self.kernel_size
        #                 #current projection onto current image
        #                 current_proj = curr_img[start_h:end_h,start_w:end_w,:]
        #                 curr_dimg[start_h:end_h,start_w:end_w,:] += self.params[self.w_name][:,:,:,f] * dprev[i, j, k, f]
        #                 self.grads[self.w_name][:,:,:,f] += current_proj * dprev[i, j, k, f]
        #                 self.grads[self.b_name][f] += dprev[i, j, k, f]
        #     dimg_n = dimg[i, :, :, :]
        #     dimg_n = curr_dimg[self.padding:-self.padding, self.padding:-self.padding, :]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg_n


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
        output_height = (int) ((img.shape[1] - self.pool_size)/self.stride + 1)
        output_width = (int) ((img.shape[2] - self.pool_size)/self.stride + 1)
        shape0 = img.shape[0]
        shape3 = img.shape[3]

        output = np.zeros((shape0, output_height, output_width, shape3))
        current_img = img[:,:,:,:]
        for j in range(output_height): #height
            for k in range(output_width): #width
                for f in range(img.shape[3]):  #filter number
                    start_h = j * self.stride
                    end_h = start_h + self.pool_size
                    start_w = k * self.stride
                    end_w = start_w + self.pool_size

                    curr_proj = current_img[:,start_h:end_h,start_w:end_w,f]
                    #print("curr_proj", curr_proj.shape)
                    output[:,j,k,f] = np.max(curr_proj,axis=(1,2))

        # for i in range(img.shape[0]): #batch
        #     current_img = img[i,:,:,:]
        #     for j in range(output_height): #height
        #         for k in range(output_width): #width
        #             for f in range(img.shape[3]):  #filter number
        #                 start_h = j * self.stride
        #                 end_h = start_h + self.pool_size
        #                 start_w = k * self.stride
        #                 end_w = start_w + self.pool_size

        #                 curr_proj = current_img[start_h:end_h,start_w:end_w,f]
        #                 print("curr_proj", curr_proj.shape)
        #                 output[i][j][k][f] = np.max(curr_proj)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
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
        #for i in range(dprev.shape[0]):  #batch  
        current_img = img[:,:,:,:]
        for j in range(dprev.shape[1]): #height
            for k in range(dprev.shape[2]): #width
                for f in range(dprev.shape[3]): #filter
                    start_h = j * self.stride
                    end_h = start_h + self.pool_size
                    start_w = k * self.stride
                    end_w = start_w + self.pool_size

                    curr_proj = current_img[:,start_h:end_h,start_w:end_w,f]
                    curr_grad = np.zeros((curr_proj.shape))
                    max_value = np.max(curr_proj)
                    #print("curr_proj", curr_proj.shape)
                    #curr_grad[:,:,:] = curr_proj[:,:,:] == np.max(curr_proj, axis=(1,2))
                    curr_grad = curr_proj.max(axis=(1,2),keepdims=1) == curr_proj
                    #print("curr_grad", curr_grad.shape)
                    #print(curr_grad)
                    dimg[:, start_h:end_h, start_w:end_w, f] += np.multiply(curr_grad.transpose(1,2,0), dprev[:, j, k, f]).transpose(2,0,1)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dimg
