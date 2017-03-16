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
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    if use_batch_norm:
        raise NotImplementedError
    else:
        layer = Lipshitz_Layer(layer, 256*12*12,init=1)
        layer = ReshapeLayer(layer, (-1, 256, 12, 12))
        layer = Subpixel_Layer(layer, 128, (3,3), 2)
        layer = Subpixel_Layer(layer, 64, (3,3), 2)
        layer = LipConvLayer(layer,1,(9,9),init=1,
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
        layer = FlattenLayer(layer)
        layer = Lipshitz_Layer(layer,1024)
        layer = Lipshitz_Layer(layer,1)

    print ("Discriminator output:", layer.output_shape)
    print("Number of parameters:", lasagne.layers.count_params(layer)) 
    return layer


def main(num_epochs=200,batch_norm=True):

    import matplotlib.pyplot as plt
    from PIL import Image
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
    discriminator_loss = real_out.mean()-fake_out.mean()
    
    '''
    real_out = lasagne.layers.get_output(discriminator)[:,0]
    fake_out = lasagne.layers.get_output(discriminator,
            lasagne.layers.get_output(generator))
    fake_out_score=fake_out[:,0]
    mse_loss=T.mean((fake_out[:,1:]-random_var)**2)
    generator_loss = fake_out_score.mean()+0.0*mse_loss
    discriminator_loss = real_out.mean()-fake_out_score.mean()+(
        0.0*mse_loss+1.0*discriminator.max_gradient)
    '''
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    generator_updates = lasagne.updates.sgd(generator_loss, generator_params,learning_rate=0.01)
    discriminator_updates = lasagne.updates.sgd(discriminator_loss, discriminator_params,learning_rate=0.1)

    print("Compiling functions")
    generator_train_fn = theano.function([random_var],
                               fake_out.mean(),
                               updates=generator_updates)
    discriminator_train_fn = theano.function([random_var, input_var],
                               updates=discriminator_updates)
    get_real_score = theano.function([input_var],
                               real_out.mean())
    get_fake_score = theano.function([random_var],
                               fake_out.mean())
    rescale_discriminator = theano.function([],updates=discriminator.rescale)

    gen_fn = theano.function([random_var], 
        lasagne.layers.get_output(generator, deterministic=True))

    for epoch in range(num_epochs):
        print("Pre-training Discriminator")
        for batch in iterate_minibatches(train_x, train_y, 128):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            discriminator_train_fn(noise, inputs)
            rescale_discriminator()
            #print(get_real_score(inputs)-get_fake_score(noise))
        inputs=valid_x
        noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
        print("score: %f" % (get_real_score(inputs)-get_fake_score(noise)))
        print("Starting Epoch %d" % epoch)
        generator_err = 0
        discriminator_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, 128):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            discriminator_train_fn(noise, inputs)
            rescale_discriminator()
            discriminator_err += get_real_score(inputs)-get_fake_score(noise)
            current_gen_err=np.array(generator_train_fn(noise))
            generator_err += current_gen_err
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  generator loss:\t\t{}".format(generator_err / train_batches))
        print("  discriminator loss:\t\t{}".format(discriminator_err / train_batches))

        # And finally, we plot some generated data
        samples = 255*gen_fn(lasagne.utils.floatX(np.random.rand(20, 100)))
        print(samples.shape)
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
