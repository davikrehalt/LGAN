from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import lasagne
from load_data import load_mnist
import time

def iterate_minibatches(inputs, targets, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_standard_cnn(input_var):
    from lasagne.layers import InputLayer,ReshapeLayer, Conv2DLayer,DenseLayer,FlattenLayer
    network = InputLayer(shape=(None, 784),input_var=input_var)
    network = ReshapeLayer(network, (-1, 1, 28, 28))
    network = Conv2DLayer(network, 32, 5)
    network = Conv2DLayer(network, 32, 5)
    network = Conv2DLayer(network, 32, 5)
    network = FlattenLayer(network)
    network = DenseLayer(network, 256)
    network = DenseLayer(network, 10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return network

def build_maxout_cnn(input_var):
    from lasagne.layers import InputLayer
    from layers import Lipshitz_Layer,LipConvLayer,ReshapeLayer,FlattenLayer
    network = InputLayer(shape=(None, 784),input_var=input_var)
    network = ReshapeLayer(network, (-1, 1, 28, 28))
    network = LipConvLayer(network,32, (5, 5), init=1)
    network = LipConvLayer(network,32, (5, 5), init=1)
    network = LipConvLayer(network,32, (5, 5), init=1)
    network = FlattenLayer(network)
    network = Lipshitz_Layer(network, 256, init=1)
    network = Lipshitz_Layer(network, 10, init=1,
                             nonlinearity=lasagne.nonlinearities.softmax)
    return network

def main(model='standard',n_epochs=100):
    print('Loading data')
    datasets = load_mnist()

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    print("Building model")
    if model == 'standard':
        network=build_standard_cnn(input_var)
    elif model == 'maxout':
        network=build_maxout_cnn(input_var)
        max_gradient=theano.function([],network.max_gradient)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(
        prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params,learning_rate=0.005)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(
        test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var,axis=1)),
                      dtype=theano.config.floatX)

    print("Compiling Functions")
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print('Training')
    for epoch in range(n_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, 500):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(valid_x, valid_y, 500):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if model == "maxout":
            print("  maximum gradient:\t\t{:.2f}".format(1.0*max_gradient()))
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test_x, test_y, 500):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == "__main__":
    main('maxout')
