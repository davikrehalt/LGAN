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
        layer = Lipshitz_Layer(layer, 512*7*7,init=1)
        layer = ReshapeLayer(layer, (-1, 512, 7, 7))
        layer = Subpixel_Layer(layer, 256, (3,3), 2)
        layer = Subpixel_Layer(layer, 128, (3,3), 2)
        layer = Subpixel_Layer(layer, 64, (3,3), 2)
        layer = LipConvLayer(layer, 1,(1,1),init=1,
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
        layer = LipConvLayer(layer,16, (5, 5),init=1)
        layer = LipConvLayer(layer,32, (5, 5),init=1)
        layer = LipConvLayer(layer,64, (5, 5),init=1)
        layer = LipConvLayer(layer,128, (5, 5),init=1)
        layer = FlattenLayer(layer)
        layer = Lipshitz_Layer(layer,512,init=1)
        layer = Lipshitz_Layer(layer,1+10,init=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    print ("Discriminator output:", layer.output_shape)
    print("Number of parameters:", lasagne.layers.count_params(layer)) 
    return layer


def main(num_epochs=200,batch_norm=True):
    import matplotlib.pyplot as plt
    from PIL import Image
    batch_size=128
    noise_size=10

    print('Loading data')
    datasets = load_mnist()

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    image_var = T.matrix('image')
    representation_var = T.matrix('representation')

    print("Building model")
    generator = build_generator(representation_var,batch_norm)
    discriminator = build_discriminator(image_var,batch_norm)
    max_gradient=theano.function([],discriminator.max_gradient)

    real_out=lasagne.layers.get_output(discriminator)[:,0]
    fake_out=lasagne.layers.get_output(discriminator,
        lasagne.layers.get_output(generator))[:,0]

    return_representation = lasagne.layers.get_output(discriminator,
        lasagne.layers.get_output(generator))[:,1:]
    return_image = lasagne.layers.get_output(generator,
        lasagne.layers.get_output(discriminator)[:,1:])

    reconstruction_loss1 = ((return_image-image_var)**2).mean()
    reconstruction_loss2 = ((return_representation-representation_var)**2).mean()
    reconstruction_loss = reconstruction_loss1+reconstruction_loss2
    
    generator_loss = (fake_out**2).mean()
    discriminator_loss =(real_out**2).mean()+((1.0-fake_out)**2).mean()
    
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    total_params=generator_params+discriminator_params

    autoencoder_updates=lasagne.updates.adam(reconstruction_loss1, total_params)

    generator_updates1 = lasagne.updates.sgd(reconstruction_loss, generator_params,learning_rate=0.05)
    discriminator_updates1 = lasagne.updates.sgd(reconstruction_loss, discriminator_params,learning_rate=0.05)

    generator_updates2 = lasagne.updates.sgd(generator_loss, generator_params,learning_rate=0.05)
    discriminator_updates2 = lasagne.updates.sgd(discriminator_loss, discriminator_params,learning_rate=0.05)

    print("Compiling functions")
    autoencoder_train= theano.function([image_var],reconstruction_loss1,
                               updates=autoencoder_updates)
    generator_train_fn1 = theano.function([representation_var,image_var],reconstruction_loss,
                               updates=generator_updates1)
    discriminator_train_fn1 = theano.function([representation_var, image_var],reconstruction_loss,
                               updates=discriminator_updates1)
    generator_train_fn2 = theano.function([representation_var],
                               updates=generator_updates2)
    discriminator_train_fn2 = theano.function([representation_var, image_var],
                               updates=discriminator_updates2)
    get_max_gradient = theano.function([],discriminator.max_gradient)
    gen_fn = theano.function([representation_var], 
        lasagne.layers.get_output(generator, deterministic=True))
    get_real_score = theano.function([image_var],real_out.mean())
    get_fake_score = theano.function([representation_var],fake_out.mean())

    print("Making autoencoder")
    for epoch in range(10):
        print("Starting Epoch %d" % epoch)
        start_time = time.time()
        auto_cost=[]
        for batch in iterate_minibatches(train_x, train_y, batch_size):
            inputs, targets = batch
            auto_cost.append(autoencoder_train(inputs))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch, num_epochs, time.time() - start_time))
        print("reconstruction_loss: ",np.mean(auto_cost))

        # And finally, we plot some generated data
        samples = 255*gen_fn(lasagne.utils.floatX(np.random.rand(20, noise_size)))
        for i in range(20):
            array=np.array(samples[i])
            array=array.reshape((28,28))
            im=Image.fromarray(array).convert('L')
            im.save('mnist_'+str(i)+'.png')
        print('Images saved')
    print("Training")
    for epoch in range(num_epochs):
        '''
        real_sum = 0
        fake_sum = 0
        valid_batches = 0
        for batch in iterate_minibatches(valid_x, valid_y, batch_size):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(batch_size, noise_size))
            real_sum += get_real_score(inputs)
            fake_sum += get_fake_score(noise)
            valid_batches += 1
        real_score=real_sum/valid_batches
        fake_score=fake_sum/valid_batches
        balance=0.5*(real_score**2+(1.0-fake_score)**2)
        print("real score: %f" % real_score) 
        print("fake score: %f" % fake_score) 
        print("balance: %f" % balance) 
        '''
        print("max gradient: %f" % get_max_gradient())
        print("Starting Epoch %d" % epoch)
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, batch_size):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(batch_size, noise_size))
            if np.random.random()<0.5:
                discriminator_train_fn(noise, inputs)
            else: 
                generator_train_fn(noise,inputs)

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
