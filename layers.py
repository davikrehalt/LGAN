from __future__ import print_function,division
import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.tensor.nnet import conv2d
from ops import tmax,tmin,tbox

class Lipshitz_Layer(lasagne.layers.Layer):
    def __init__(self, incoming, n_out, W=None, b=None,init=0,nonlinearity=None,**kwargs):
        super(Lipshitz_Layer,self).__init__(incoming,**kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        n_max=2
        n_in=self.input_shape[1]
        self.n_out=n_out
        self.n_params=n_max*n_out*(1+n_in)
        if W is None:
            if init == 0:
                min_value = -1.0 / n_in
                max_value = 1.0 / n_in
            elif init == 1:
                min_value = -np.sqrt(3.0/n_in)
                max_value = np.sqrt(3.0/n_in)
            elif init == 2:
                min_value = 0.5 / n_in
                max_value = 1.5 / n_in
            W = lasagne.init.Uniform(range=(min_value,max_value))

        if b is None:
            b = lasagne.init.Uniform(range=(-0.5,0.5))

        self.W = self.add_param(W,(n_max,n_in,n_out))
        self.b = self.add_param(b,(n_max,n_out))
        self.pre_gradient_norms=T.sum(abs(self.W),axis=1)
        self.gradient_norms=tmax(self.pre_gradient_norms,1.0)
        self.rescale_W = self.W / self.gradient_norms.dimshuffle(0,'x',1)
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        self.rescale.append((self.W,self.rescale_W))
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0
        self.max_gradient*=T.max(self.pre_gradient_norms)

    def get_output_for(self,input,**kwargs):
        return self.nonlinearity((T.dot(input,self.W) + self.b).max(axis=1))
    def get_output_shape_for(self,input_shape):
        return (input_shape[0],self.n_out)

class LipConvLayer(lasagne.layers.Layer):
    def __init__(self,incoming,n_out,filter_size,W=None,b=None,init=0,nonlinearity=None,**kwargs):
        #shape =(
        #        height(0),width(1),
        #        filter height(2), filter width(3)
        #        num input feature maps(4),
        #        num intermediate filters per output map(5), 
        #        num output feature maps(6),

        #only implemented "valid" filter
        super(LipConvLayer,self).__init__(incoming,**kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        shape=[self.input_shape[2],self.input_shape[3],
               filter_size[0],filter_size[1],
               self.input_shape[1],2,n_out]
        n_in = shape[2]*shape[3]*shape[4]
        self.n_out=n_out
        self.filter_shape = [shape[5],shape[6],shape[4],shape[2],shape[3]]
        self.bias_shape   = [shape[5],shape[6]]
        self.shape=shape
        self.n_params=np.prod(self.filter_shape)+np.prod(self.bias_shape)
        if W is None:
            if init == 0:
                min_value = -1.0 / n_in
                max_value = 1.0 / n_in
            elif init == 1:
                min_value = -np.sqrt(3.0/n_in)
                max_value = np.sqrt(3.0/n_in)
            elif init == 2:
                min_value = 0.5 / n_in
                max_value = 1.5 / n_in
            W = lasagne.init.Uniform(range=(min_value,max_value))
        if b is None:
            b = lasagne.init.Uniform(range=(-0.5,0.5))

        self.W = self.add_param(W,self.filter_shape)
        self.b = self.add_param(b,self.bias_shape)
        self.pre_gradient_norms=T.sum(abs(self.W),axis=(2,3,4))
        self.gradient_norms=tmax(self.pre_gradient_norms,1.0)
        self.rescale_W = self.W / self.gradient_norms.dimshuffle(0,1,'x','x','x')
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        self.rescale.append((self.W,self.rescale_W))
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0
        self.max_gradient*=T.max(self.pre_gradient_norms)

    def get_output_for(self,input,**kwargs):
        intermediate=[]
        for i in range(self.shape[5]):
            conv_out=conv2d(
                input=input,
                filters=self.W[i],
                border_mode='valid',
                filter_shape=self.filter_shape[1:],
                input_shape=self.input_shape
            )
            intermediate.append(conv_out+self.b[i].dimshuffle('x',0,'x','x'))
        return self.nonlinearity(T.max(intermediate,axis=0))
    def get_output_shape_for(self,input_shape):
        return (input_shape[0],self.shape[6],
                self.shape[0]-self.shape[2]+1,
                self.shape[1]-self.shape[3]+1)

class Subpixel_Layer(lasagne.layers.Layer):
    def __init__(self,incoming,n_out,filter_size,multiplier,W=None,b=None,init=1,nonlinearity=None,**kwargs):
        #shape =(
        #        height(0),width(1)
        #        filter height(2), filter width(3)
        #        multiplier(4),
        #        num input feature maps(5),
        #        num intermediate filters per output map(6), 
        #        num output feature maps(7),

        #only implements "valid" filter

        super(Subpixel_Layer,self).__init__(incoming,**kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        shape=[self.input_shape[2],self.input_shape[3],
               filter_size[0],filter_size[1],multiplier,
               self.input_shape[1],2,n_out]
        n_in = shape[3]*shape[4]*shape[6]
        self.filter_shape = [shape[6],shape[4]*shape[4]*shape[7],shape[5],shape[2],shape[3]]
        self.bias_shape   = [shape[6],shape[4]*shape[4]*shape[7]]
        self.shape=shape
        self.n_params=np.prod(self.filter_shape)+np.prod(self.bias_shape)
        if W is None:
            if init == 0:
                min_value = -1.0 / n_in
                max_value = 1.0 / n_in
            elif init == 1:
                min_value = -np.sqrt(3.0/n_in)
                max_value = np.sqrt(3.0/n_in)
            elif init == 2:
                min_value = 0.5 / n_in
                max_value = 1.5 / n_in
            W = lasagne.init.Uniform(range=(min_value,max_value))
        if b is None:
            b = lasagne.init.Uniform(range=(-0.5,0.5))

        self.W = self.add_param(W,self.filter_shape)
        self.b = self.add_param(b,self.bias_shape)
        self.pre_gradient_norms=T.sum(abs(self.W),axis=(2,3,4))
        self.gradient_norms=tmax(self.pre_gradient_norms,1.0)
        self.rescale_W = self.W / self.gradient_norms.dimshuffle(0,1,'x','x','x')
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        self.rescale.append((self.W,self.rescale_W))
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0
        self.max_gradient*=T.max(self.pre_gradient_norms)

    def get_output_for(self,input,**kwargs):
        image_shape  = [None,self.shape[5],self.shape[0],self.shape[1]]
        output_shape = [input.shape[0],self.shape[7],
                        self.shape[4]*(self.shape[0]-self.shape[2]+1),
                        self.shape[4]*(self.shape[1]-self.shape[3]+1)]
        intermediate=[]
        for i in range(self.shape[6]):
            conv_out=conv2d(
                input=input,
                filters=self.W[i],
                border_mode='valid',
                filter_shape=self.filter_shape[1:],
                input_shape=image_shape
            )
            intermediate.append(conv_out+self.b[i].dimshuffle('x',0,'x','x'))
        pre_output=T.max(intermediate,axis=0)
        #reshape the output
        output=T.zeros(output_shape)
        r = self.shape[4]
        for x in range(r): 
            for y in range(r):
                output=T.set_subtensor(
                    output[:,:,x::r,y::r],pre_output[:,r*x+y::r*r,:,:])
        return self.nonlinearity(output)

    def get_output_shape_for(self,input_shape):
        return (input_shape[0],self.shape[7],
                self.shape[4]*(self.shape[0]-self.shape[2]+1),
                self.shape[4]*(self.shape[1]-self.shape[3]+1))
