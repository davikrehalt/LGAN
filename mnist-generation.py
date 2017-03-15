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


def build_generator(input_var=None,use_batch_norm=True):
    from lasagne.layers import (InputLayer, ReshapeLayer,
                                batch_norm)
    from layers import Lipshitz_Layer,LipConvLayer,Subpixel_Layer
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    if use_batch_norm:
        raise NotImplementedError
    else:
        layer = Lipshitz_Layer(layer, 128*6*6,init=1)
        layer = ReshapeLayer(layer, (-1, 128, 6, 6))
        layer = Subpixel_Layer(layer, 64, (3,3), 2)
        layer = Subpixel_Layer(layer, 32, (3,3), 2)
        layer = Subpixel_Layer(layer, 16, (3,3), 2)
        layer = Subpixel_Layer(layer, 8, (3,3), 2)
        layer = LipConvLayer(layer,1,(9,9),init=1)
        layer = ReshapeLayer(layer, (-1, 784))
    print ("Generator output:", layer.output_shape)
    return layer

def build_discriminator(input_var=None,use_batch_norm=True):
    from lasagne.layers import (InputLayer, ReshapeLayer,
                                FlattenLayer,batch_norm)
    from layers import Lipshitz_Layer,LipConvLayer
    layer = InputLayer(shape=(None, 784),input_var=input_var)
    layer = ReshapeLayer(layer, (-1, 1, 28, 28))
    if use_batch_norm:
        raise NotImplementedError
    else:
        layer = LipConvLayer(layer,32, (5, 5), init=1)
        layer = LipConvLayer(layer,32, (5, 5), init=1)
        layer = LipConvLayer(layer,32, (5, 5), init=1)
        layer = FlattenLayer(layer)
        layer = Lipshitz_Layer(layer, 1024)
        layer = Lipshitz_Layer(layer,1,nonlinearity=None)

    print ("Discriminator output:", layer.output_shape)
    return layer


def main(num_epochs=200,batch_norm=True):
    print('Loading data')
    datasets = load_mnist()

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    input_var = T.matrix('inputs')
    random_var = T.matrix('random')

    print("Building model")
    generator = build_generator(random_var,batch_norm)
    discriminator = build_discriminator(input_var,batch_norm)

    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator,
            lasagne.layers.get_output(generator))
    
    generator_loss = fake_out.mean()
    discriminator_loss = real_out.mean()-fake_out.mean()
    
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    generator_updates = lasagne.updates.rmsprop(generator_loss, generator_params)
    discriminator_updates = lasagne.updates.rmsprop(discriminator_loss, discriminator_params)
    print("Compiling functions")
    generator_train_fn = theano.function([random_var],
                               generator_loss,
                               updates=generator_updates)
    discriminator_train_fn = theano.function([random_var, input_var],
                               discriminator_loss,
                               updates=discriminator_updates)

    gen_fn = theano.function([random_var], 
        lasagne.layers.get_output(generator, deterministic=True))

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        generator_err = 0
        discriminator_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, 128):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            generator_err += np.array(generator_train_fn(noise))
            discriminator_err += np.array(discriminator_train_fn(noise, inputs))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  generator loss:\t\t{}".format(generator_err / train_batches))
        print("  discriminator loss:\t\t{}".format(discriminator_err / train_batches))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))
        import matplotlib.pyplot as plt
        plt.imsave('mnist_samples.png',
                   (samples.reshape(6, 7, 28, 28)
                           .transpose(0, 2, 1, 3)
                           .reshape(6*28, 7*28)),
                   cmap='gray')

    # Optionally, you could now dump the network weights to a file like this:
    #np.savez('mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    #np.savez('mnist_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main(batch_norm=False)
