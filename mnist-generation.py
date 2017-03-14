from __future__ import print_function,division
import numpy as np
import theano
import theano.tensor as T
import lasagne
from load_data import load_mnist
import time

def iterate_minibatches(inputs, targets, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_standard_generator(input_var=None):
    from lasagne.layers import (InputLayer, ReshapeLayer,
                                DenseLayer, batch_norm)
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, pad=2)
    print ("Generator output:", layer.output_shape)
    return layer

def build_standard_discriminator(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm)
    from lasagne.nonlinearities import Rectify, sigmoid
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2,
                                   pad=2, nonlinearity=Rectify))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2,
                                   pad=2, nonlinearity=Rectify))
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=Rectify))
    layer = DenseLayer(layer, 1,nonlinearity=None)
    print ("Discriminator output:", layer.output_shape)
    return layer


def main(num_epochs=200, initial_eta=2e-4):
    print('Loading data')
    datasets = load_mnist()

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    print("Building model and compiling functions...")
    if model == 'standard':
        generator = build_standard_generator(noise_var)
        discriminator = build_standard_discriminator(input_var)

    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator,
            lasagne.layers.get_output(generator))
    
    generator_loss = fake_out.mean()
    discriminator_loss = real_out.mean()-fake_out.mean()
    
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=eta, beta1=0.5)
    updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5))

    train_fn = theano.function([noise_var, input_var],
                               discriminator_loss,
                               updates=updates)

    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            train_err += np.array(train_fn(noise, inputs))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            plt.imsave('mnist_samples.png',
                       (samples.reshape(6, 7, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(6*28, 7*28)),
                       cmap='gray')

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    np.savez('mnist_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
