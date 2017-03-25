from __future__ import print_function,division
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from ops import tmax,tmin,tbox
from load_data import load_mnist

def iterate_minibatches(inputs, targets, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_generator(input_var=None,use_batch_norm=True):
    from lasagne.layers import InputLayer,batch_norm
    from layers import (Lipshitz_Layer,LipConvLayer,Subpixel_Layer
        ,ReshapeLayer, FlattenLayer)

    layer = InputLayer(shape=(None, 10), input_var=input_var)
    if use_batch_norm:
        raise NotImplementedError
    else:
        layer = Lipshitz_Layer(layer, 128*7*7,init=1)
        layer = ReshapeLayer(layer, (-1, 128, 7, 7))
        layer = Subpixel_Layer(layer, 64, (3,3), 2)
        layer = Subpixel_Layer(layer, 32, (3,3), 2)
        layer = Subpixel_Layer(layer, 16, (3,3), 2,
            nonlinearity=lasagne.nonlinearities.sigmoid)
        layer = ReshapeLayer(layer, (-1, 784))
    print("Generator output:", layer.output_shape)
    print("Number of parameters:", lasagne.layers.count_params(layer)) 
    return layer

def build_discriminator(input_var=None,use_batch_norm=True):
    from lasagne.layers import InputLayer,batch_norm
    from layers import (Lipshitz_Layer,LipConvLayer,Subpixel_Layer
        ,ReshapeLayer, FlattenLayer)

    layer = InputLayer(shape=(None, 784),input_var=input_var)
    if use_batch_norm:
        raise NotImplementedError
    else:
        layer = ReshapeLayer(layer, (-1, 1, 28, 28))
        layer = LipConvLayer(layer,16, (5, 5))
        layer = LipConvLayer(layer,32, (5, 5))
        layer = LipConvLayer(layer,64, (5, 5))
        layer = LipConvLayer(layer,128, (5, 5))
        layer = FlattenLayer(layer)
        layer = Lipshitz_Layer(layer,512)
        layer = Lipshitz_Layer(layer,1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    print ("Discriminator output:", layer.output_shape)
    print("Number of parameters:", lasagne.layers.count_params(layer)) 
    return layer


def main(num_epochs=200,batch_norm=True):
    import matplotlib.pyplot as plt
    from PIL import Image
    batch_size=128
    noise_size=10
    discrim_step=1
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
    max_gradient=theano.function([],discriminator.max_gradient)

    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator,
            lasagne.layers.get_output(generator))
    
    generator_loss = fake_out.mean()
    discriminator_loss = ((real_out)**2).mean()+((1.0-fake_out)**2).mean()+0.1*discriminator.norm
    
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    generator_updates = lasagne.updates.rmsprop(generator_loss, generator_params,learning_rate=0.05)
    discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params)

    print("Compiling functions")
    generator_train_fn = theano.function([random_var],
                               updates=generator_updates)
    discriminator_train_fn = theano.function([random_var, input_var],
                               updates=discriminator_updates)
    get_real_score = theano.function([input_var],
                               real_out.mean())
    get_fake_score = theano.function([random_var],
                               fake_out.mean())
    rescale_discriminator = theano.function([],updates=discriminator.rescale)
    get_max_gradient = theano.function([],discriminator.max_gradient)
    get_norm = theano.function([],discriminator.norm)
    gen_fn = theano.function([random_var], 
        lasagne.layers.get_output(generator, deterministic=True))

    print("Pre-training Discriminator")
    start_time = time.time()
    for batch in iterate_minibatches(train_x, train_y, batch_size):
        inputs, targets = batch
        noise = lasagne.utils.floatX(np.random.rand(batch_size, noise_size))
        discriminator_train_fn(noise, inputs)
    print("Pretraining took {:.3f}s".format(
        time.time() - start_time))

    print("Training")
    for epoch in range(num_epochs):
        real_sum = 0
        fake_sum = 0
        valid_batches = 0
        for batch in iterate_minibatches(valid_x, valid_y, batch_size):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(batch_size, noise_size))
            real_sum += get_real_score(inputs)
            fake_sum += get_fake_score(noise)
            valid_batches += 1
        print("max gradient: %f" % get_max_gradient())
        print("norm: %f" % get_norm())
        print("real score: %f" % (real_sum/valid_batches)) 
        print("fake score: %f" % (fake_sum/valid_batches)) 
        print("Starting Epoch %d" % epoch)
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, batch_size):
            inputs, targets = batch
            for _ in range(discrim_step):
                noise = lasagne.utils.floatX(np.random.rand(batch_size, noise_size))
                discriminator_train_fn(noise, inputs)
                rescale_discriminator()
            noise = lasagne.utils.floatX(np.random.rand(batch_size, noise_size))
            generator_train_fn(noise)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch, num_epochs, time.time() - start_time))

        # And finally, we plot some generated data
        samples = 255*gen_fn(lasagne.utils.floatX(np.random.rand(20, noise_size)))
        for i in range(20):
            array=np.array(samples[i])
            array=array.reshape((28,28))
            im=Image.fromarray(array).convert('L')
            im.save('mnist_'+str(i)+'.png')
        print('Images saved')

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
