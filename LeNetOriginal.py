"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logisticOriginal import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=1,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=5): # epoch no was 200
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        [cost,layer3.p_y_given_x,layer3.y_pred,layer0.W,layer1.W,layer2.W,layer3.W],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 50 # was 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    counter4 = 0
    filterHolder = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            # compute zero-one loss on validation set
            #print(cost_ij)
            #print(numpy.array(cost_ij[3])[0,0,...])
            #print(numpy.array(cost_ij[3])[1,0,...])
            #print('shape')
            #print(numpy.array(cost_ij[3])[0,0,...])
            filter0 = numpy.array(cost_ij[3])[0,0,...]
            filter0 = filter0/(filter0.max()/255.0)
            filter01 = numpy.array(cost_ij[3])[3,0,...]
            filter01 = filter01/(filter01.max()/255.0)
            filter02 = numpy.array(cost_ij[3])[6,0,...]
            filter02 = filter02/(filter02.max()/255.0)
            filter03 = numpy.array(cost_ij[3])[12,0,...]
            filter03 = filter03/(filter03.max()/255.0)
            filter04 = numpy.array(cost_ij[3])[16,0,...]
            filter04 = filter04/(filter04.max()/255.0)
            filter05 = numpy.array(cost_ij[3])[19,0,...]
            filter05 = filter05/(filter05.max()/255.0)
            
            filter1 = numpy.array(cost_ij[4])[0,0,...]
            filter1 = filter1/(filter1.max()/255.0)
            #filter2 = numpy.array(cost_ij[5])[0,0,...]
            #filter2 = filter2/(filter2.max()/255.0)
            hiddenW = numpy.array(cost_ij[5])[:,0]
            hiddenW = hiddenW/(hiddenW.max()/255.0)
            logRWT = numpy.array(cost_ij[6])[:,0]
            logRWT = logRWT/(logRWT.max()/255.0)
            logRWF = numpy.array(cost_ij[6])[:,1]
            logRWF = logRWF/(logRWF.max()/255.0)
            hiddenW = numpy.reshape(hiddenW[0:25],[5,5])
            logRWT = numpy.reshape(logRWT[0:25],[5,5])
            logRWF = numpy.reshape(logRWF[0:25],[5,5])
            #print('shapes')
            #print(hiddenW.shape)
            #print(logRWT.shape)
            #filter1.resize([filter0.shape[0],filter0.shape[1]])
            #print(filter1)
            #print()
            filter1 = numpy.vstack([filter1,numpy.zeros([filter0.shape[0]-filter1.shape[0],filter1.shape[1]])])
            filter1 = numpy.hstack([filter1,numpy.zeros([filter1.shape[0],filter0.shape[1]-filter1.shape[1]])])
            #filter2 = numpy.vstack([filter2,numpy.zeros([filter0.shape[0]-filter2.shape[0],filter2.shape[1]])])
            #filter2 = numpy.hstack([filter2,numpy.zeros([filter2.shape[0],filter0.shape[1]-filter2.shape[1]])])
            hiddenW = numpy.vstack([hiddenW,numpy.zeros([filter0.shape[0]-hiddenW.shape[0],hiddenW.shape[1]])])
            hiddenW = numpy.hstack([hiddenW,numpy.zeros([hiddenW.shape[0],filter0.shape[1]-hiddenW.shape[1]])])
            logRWT = numpy.vstack([logRWT,numpy.zeros([filter0.shape[0]-logRWT.shape[0],logRWT.shape[1]])])
            logRWT = numpy.hstack([logRWT,numpy.zeros([logRWT.shape[0],filter0.shape[1]-logRWT.shape[1]])])
            logRWF = numpy.vstack([logRWF,numpy.zeros([filter0.shape[0]-logRWF.shape[0],logRWF.shape[1]])])
            logRWF = numpy.hstack([logRWF,numpy.zeros([logRWF.shape[0],filter0.shape[1]-logRWF.shape[1]])])
            totFilter = numpy.hstack([filter0,filter1,hiddenW,logRWT,logRWF])
            totFilter = numpy.vstack([totFilter,numpy.hstack([filter01,filter02,filter03,filter04,filter05])])
            #print(filter0)
            #plt.imshow(filter0, cmap = cm.Greys_r,interpolation="nearest")
            #plt.show()
            filterHolder.append(totFilter)#=filter0
            
            counter4 +=1
            if (iter + 1) % validation_frequency == 0:

                
                validation_losses = [validate_model(i) for i
                                     in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(int(n_test_batches))
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    fig = plt.figure() # make figure
    im = plt.imshow(filterHolder[0], cmap = cm.Greys_r, interpolation="nearest") 
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(filterHolder[j])
        print(j)
        #print(filterHolder[j])
        # return the artists set
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames=len(filterHolder), 
                              interval=10, blit=True, repeat=True)
                              
    #plt.imshow(filterHolder[0], cmap = cm.Greys_r, interpolation="nearest")
    print(filterHolder)
    plt.show()

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)