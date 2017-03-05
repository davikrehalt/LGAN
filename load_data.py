from __future__ import print_function,division
import six.moves.cPickle as pickle
import gzip
import os
import sys
import theano
import numpy as np
import theano.tensor as T

__docformat__ = 'restructedtext en'
def load_mnist():
    print('Loading MNIST')
    dataset='mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        dataset = new_path

    if not os.path.isfile(dataset):
        dataset='mnist.pkl.gz'
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)
    # Load the dataset
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    def one_hot(data_xy, borrow=True):
        data_x, data_y = data_xy
        data_x=np.asarray(data_x, dtype=theano.config.floatX)
        one_hot_y = np.zeros((data_y.shape[0],10))
        one_hot_y[range(data_y.shape[0]),data_y]=1.0
        one_hot_y=np.asarray(one_hot_y, dtype=theano.config.floatX)
        return data_x, one_hot_y 

    test_set_x, test_set_y = one_hot(test_set)
    valid_set_x, valid_set_y = one_hot(valid_set)
    train_set_x, train_set_y = one_hot(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

