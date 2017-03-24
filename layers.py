from __future__ import print_function,division
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import Layer
from theano.tensor.nnet import conv2d
from ops import tmax,tmin,tbox

class FlattenLayer(Layer):
    #Copied from Source but added rescale and max_gradient
    #Copied on March 15,2017
    """
    A layer that flattens its input. The leading ``outdim-1`` dimensions of
    the output will have the same shape as the input. The remaining dimensions
    are collapsed into the last dimension.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    outdim : int
        The number of dimensions in the output.
    See Also
    --------
    flatten  : Shortcut
    """
    def __init__(self, incoming, outdim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.outdim = outdim
        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0

    def get_output_shape_for(self, input_shape):
        to_flatten = input_shape[self.outdim - 1:]

        if any(s is None for s in to_flatten):
            flattened = None
        else:
            flattened = int(np.prod(to_flatten))

        return input_shape[:self.outdim - 1] + (flattened,)

    def get_output_for(self, input, **kwargs):
        return input.flatten(self.outdim)


class ReshapeLayer(Layer):
    #Copied from Source but added rescale and max_gradient
    """
    A layer reshaping its input tensor to another tensor of the same total
    number of elements.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    shape : tuple
        The target shape specification. Each element can be one of:
        * ``i``, a positive integer directly giving the size of the dimension
        * ``[i]``, a single-element list of int, denoting to use the size
          of the ``i`` th input dimension
        * ``-1``, denoting to infer the size for this dimension to match
          the total number of elements in the input tensor (cannot be used
          more than once in a specification)
        * TensorVariable directly giving the size of the dimension
    Examples
    --------
    >>> from lasagne.layers import InputLayer, ReshapeLayer
    >>> l_in = InputLayer((32, 100, 20))
    >>> l1 = ReshapeLayer(l_in, ((32, 50, 40)))
    >>> l1.output_shape
    (32, 50, 40)
    >>> l_in = InputLayer((None, 100, 20))
    >>> l1 = ReshapeLayer(l_in, ([0], [1], 5, -1))
    >>> l1.output_shape
    (None, 100, 5, 4)
    Notes
    -----
    The tensor elements will be fetched and placed in C-like order. That
    is, reshaping `[1,2,3,4,5,6]` to shape `(2,3)` will result in a matrix
    `[[1,2,3],[4,5,6]]`, not in `[[1,3,5],[2,4,6]]` (Fortran-like order),
    regardless of the memory layout of the input tensor. For C-contiguous
    input, reshaping is cheap, for others it may require copying the data.
    """

    def __init__(self, incoming, shape, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int):
                if s == 0 or s < - 1:
                    raise ValueError("`shape` integers must be positive or -1")
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int) or s[0] < 0:
                    raise ValueError("`shape` input references must be "
                                     "single-element lists of int >= 0")
            elif isinstance(s, T.TensorVariable):
                if s.ndim != 0:
                    raise ValueError(
                        "A symbolic variable in a shape specification must be "
                        "a scalar, but had %i dimensions" % s.ndim)
            else:
                raise ValueError("`shape` must be a tuple of int and/or [int]")
        if sum(s == -1 for s in shape) > 1:
            raise ValueError("`shape` cannot contain multiple -1")
        self.shape = shape
        # try computing the output shape once as a sanity check
        self.get_output_shape_for(self.input_shape)
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0

    def get_output_shape_for(self, input_shape, **kwargs):
        # Initialize output shape from shape specification
        output_shape = list(self.shape)
        # First, replace all `[i]` with the corresponding input dimension, and
        # mask parts of the shapes thus becoming irrelevant for -1 inference
        masked_input_shape = list(input_shape)
        masked_output_shape = list(output_shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                if o[0] >= len(input_shape):
                    raise ValueError("specification contains [%d], but input "
                                     "shape has %d dimensions only" %
                                     (o[0], len(input_shape)))
                output_shape[dim] = input_shape[o[0]]
                masked_output_shape[dim] = input_shape[o[0]]
                if (input_shape[o[0]] is None) \
                   and (masked_input_shape[o[0]] is None):
                        # first time we copied this unknown input size: mask
                        # it, we have a 1:1 correspondence between out[dim] and
                        # in[o[0]] and can ignore it for -1 inference even if
                        # it is unknown.
                        masked_input_shape[o[0]] = 1
                        masked_output_shape[dim] = 1
        # Secondly, replace all symbolic shapes with `None`, as we cannot
        # infer their size here.
        for dim, o in enumerate(output_shape):
            if isinstance(o, T.TensorVariable):
                output_shape[dim] = None
                masked_output_shape[dim] = None
        # From the shapes, compute the sizes of the input and output tensor
        input_size = (None if any(x is None for x in masked_input_shape)
                      else np.prod(masked_input_shape))
        output_size = (None if any(x is None for x in masked_output_shape)
                       else np.prod(masked_output_shape))
        del masked_input_shape, masked_output_shape
        # Finally, infer value for -1 if needed
        if -1 in output_shape:
            dim = output_shape.index(-1)
            if (input_size is None) or (output_size is None):
                output_shape[dim] = None
                output_size = None
            else:
                output_size *= -1
                output_shape[dim] = input_size // output_size
                output_size *= output_shape[dim]
        # Sanity check
        if (input_size is not None) and (output_size is not None) \
           and (input_size != output_size):
            raise ValueError("%s cannot be reshaped to specification %s. "
                             "The total size mismatches." %
                             (input_shape, self.shape))
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        # Replace all `[i]` with the corresponding input dimension
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                output_shape[dim] = input.shape[o[0]]
        # Everything else is handled by Theano
        return input.reshape(tuple(output_shape))

class Lipshitz_Layer(Layer):
    def __init__(self, incoming, n_out,n_max=2,W=None, b=None,init=0,nonlinearity=None,rescale=False,**kwargs):
        super(Lipshitz_Layer,self).__init__(incoming,**kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
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
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0
        self.max_gradient*=T.max(self.pre_gradient_norms)
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        if rescale:
            self.rescale_W = self.W / self.gradient_norms.dimshuffle(0,'x',1)
            self.rescale.append((self.W,self.rescale_W))
        else:
            self.norm=(self.gradient_norms).mean()

    def get_output_for(self,input,**kwargs):
        return self.nonlinearity((T.dot(input,self.W) + self.b).max(axis=1))
    def get_output_shape_for(self,input_shape):
        return (input_shape[0],self.n_out)

class LipConvLayer(Layer):
    def __init__(self,incoming,n_out,filter_size,n_max=2,W=None,b=None,init=0,nonlinearity=None,rescale=False,**kwargs):
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
               self.input_shape[1],n_max,n_out]
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
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0
        self.max_gradient*=T.max(self.pre_gradient_norms)
        if rescale:
            self.rescale_W = self.W / self.gradient_norms.dimshuffle(0,1,'x','x','x')
            self.rescale.append((self.W,self.rescale_W))
        else:
            self.norm=(self.gradient_norms).mean()

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

class Subpixel_Layer(Layer):
    def __init__(self,incoming,n_out,filter_size,multiplier,n_max=2,W=None,b=None,init=1,nonlinearity=None,rescale=False,**kwargs):
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
               self.input_shape[1],n_max,n_out]
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
        try:
            self.rescale=self.input_layer.rescale
        except AttributeError:
            self.rescale=[]
        try:
            self.max_gradient=self.input_layer.max_gradient
        except AttributeError:
            self.max_gradient=1.0
        self.max_gradient*=T.max(self.pre_gradient_norms)
        if rescale:
            self.rescale.append((self.W,self.rescale_W))
            self.rescale_W = self.W / self.gradient_norms.dimshuffle(0,1,'x','x','x')
        else:
            self.norm=(self.gradient_norms).mean()

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
