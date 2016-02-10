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
import copy

import numpy
import pickle
import theano
import theano.tensor as T
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from PIL import Image


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
        self.preOutput = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))+(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))*0.001
        #self.output = T.switch((pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) > 0, (pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')), 0 * (pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')))
        # store parameters of this layer
        self.params = [self.W, self.b]
        self.L1 = (
            abs(self.W).sum()
            
        )
        self.L2 = (
            (self.W**2).sum()
            
        )
        # keep track of model input
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        
def heatMap(testImageNo = 5,filterFactor=2,dataset='dataset3.pkl'):#filtersize must be odd
    #create dataset with shifted block
    #predictedValues = predict(length,newDataset)
    #create numpy array that is the size of the image - edge reduction
    #loop through predictedValues placing the result from 0 to 1 into the new array
    #show the final image ontop of original
    basePath = r'C:\Users\Matt\Desktop\DogProj\data'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x.set_value(test_set_x.get_value(borrow=True)-numpy.mean(test_set_x.get_value(borrow=True)))
    test_set_x = test_set_x.get_value()
    test_image = test_set_x[testImageNo,...]#[testnumber,imgsize*imgsize]
    print(test_image.shape)
    print(numpy.sqrt(test_image.shape[0]))
    test_image = test_image.reshape(numpy.sqrt(test_image.shape[0]),numpy.sqrt(test_image.shape[0]))
    filterSize = test_image.shape[0]/filterFactor
    coarseness = 15
    if filterSize % 2 == 0:
     filterSize -= 1
    blockData = numpy.zeros([int((test_image.shape[0])/coarseness)**2,test_image.shape[0]**2])
    print(test_image.shape[0])
    
    for i in range(int((test_image.shape[0])/coarseness)):
        print(i)
        for k in range(int((test_image.shape[0])/coarseness)):
            print(k)
            test_image_block = copy.deepcopy(test_image)
            test_image_block[i*coarseness:i*coarseness+((filterSize)),k*coarseness:k*coarseness+((filterSize))]=0
            # #plt.imshow(test_image_block, cmap = cm.Greys_r, interpolation='nearest')
            # #plt.show()
            blockData[i*int((test_image.shape[1])/coarseness)+k,...]=test_image_block.reshape(test_image.shape[0]**2)
    dataset = [(0,0),(0,0),(blockData,numpy.ones(blockData.shape[0]))]#ones or zeros depends on class of image
    globalPath = r'C:\Users\Matt\Desktop\DogProj\scripts\DogProjScripts'
    pickle.dump( dataset, open( os.path.join(basePath,"blockData.pkl"), "wb" ) ) # needed
    imgTest_image = Image.fromarray((test_image-numpy.min(test_image))*255)
    print(numpy.min(test_image))
    print(numpy.max(test_image))
    
    #b = (imgTest_image.resize([(predictedValues.shape[0]),(predictedValues.shape[0])]))
    #plt.imshow(b, cmap = cm.Greys_r, interpolation='nearest')
    #plt.show()
    predictedValues,validHolder = predict(blockData.shape[0],dataset='blockData.pkl')
    print('pred')
    print(predictedValues.shape)
    predictedValues=predictedValues.reshape(numpy.sqrt(predictedValues.shape[0]),numpy.sqrt(predictedValues.shape[0]))
    print(numpy.max(predictedValues))
    print(numpy.min(predictedValues))
    holdMin=numpy.min(predictedValues)
    holdMax= numpy.max(predictedValues)
    blockMin = blockData[numpy.argmin(predictedValues),...].reshape([imgTest_image.size[0],imgTest_image.size[0]])
    blockMax = blockData[numpy.argmax(predictedValues),...].reshape([imgTest_image.size[0],imgTest_image.size[0]])
    #plt.imshow(predictedValues, cmap = cm.Greys_r, interpolation='nearest')
    print((predictedValues.shape))
    b = (imgTest_image.resize([predictedValues.shape[0],predictedValues.shape[0]]))
    predictedValues *= 1/numpy.max(predictedValues)
    imgPredictedValues = Image.fromarray(numpy.uint8(plt.cm.hot(predictedValues)*255)) 
    imgPredictedValues = imgPredictedValues.resize([imgTest_image.size[0],imgTest_image.size[0]])
    
    #plt.imshow((test_image), cmap=plt.cm.gray, interpolation='nearest')
    #plt.imshow((imgTest_image), cmap=plt.cm.hot, interpolation='nearest', alpha=.6)
    #plt.show()
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    axes = axes.ravel()
    ax0, ax1 , ax2, ax3 = axes
    
    ax0.imshow((imgTest_image), cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title("Original")
    #b = numpy.resize(test_image,[predictedValues.shape[0],predictedValues.shape[0]])
    ax1.imshow((imgTest_image), cmap=plt.cm.gray, interpolation='nearest')
    #labels = labels*np.random.rand(labels.shape[0],labels.shape[1])
    #labels = random.shuffle(labels)
    #ax1.imshow(numpy.resize(predictedValues,[test_image.shape[0],test_image.shape[1]]), cmap=plt.cm.hot, interpolation='nearest', alpha=.9)
    ax1.imshow((imgPredictedValues), cmap=plt.cm.hot, interpolation='nearest', alpha=.7)
    ax1.set_title("Segmented")
    #test_image_block
    ax2.imshow((blockMin), cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title("Blocked min: "+str(holdMin))
    ax3.imshow((blockMax), cmap=plt.cm.gray, interpolation='nearest')
    ax3.set_title("Blocked max: "+str(holdMax))
    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    plt.show()
    
def predict(testNumber,dataset='dataset3.pkl'):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    rng = numpy.random.RandomState(23455)
    finalSize = 200
    index = T.lscalar()
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')
    # load the saved model
    #layer4 = pickle.load(open('best_model.pkl'))
    basePath = r'C:\Users\Matt\Desktop\DogProj\data'
    f = open(os.path.join(basePath,'best_model.pkl'), 'rb')
    layer4,layer3,layer2,layer1,layer0,validHolder,trainHolder = pickle.load(f) # 
    print('blah')
    print(numpy.array(layer0.W.get_value())[3,0,...])
    f.close()
    # compile a predictor function
    
    # We can test it on some examples from test test
    ##dataset='dataset3.pkl'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    #test_set_x = test_set_x.get_value()
    test_set_x.set_value(test_set_x.get_value(borrow=True)-numpy.mean(test_set_x.get_value(borrow=True)))
    test_set_x = test_set_x.get_value()
    layer0_input = x.reshape((testNumber, 1, 200, 200))
    layer0new = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=layer0.image_shape,
        filter_shape=layer0.filter_shape, #5,5 before
        poolsize=(2, 2)
    )
    layer0new.W.set_value(layer0.W.get_value())
    layer0new.b.set_value(layer0.b.get_value())
    layer1new = LeNetConvPoolLayer(
        rng,
        input=layer0new.output,
        image_shape=layer1.image_shape,
        filter_shape=layer1.filter_shape,
        poolsize=(2, 2)
    )
    layer1new.W.set_value(layer1.W.get_value())
    layer1new.b.set_value(layer1.b.get_value())
    layer2new = LeNetConvPoolLayer(
        rng,
        input=layer1new.output,
        image_shape=layer2.image_shape,
        filter_shape=layer2.filter_shape,
        poolsize=(2, 2)
    )
    layer2new.W.set_value(layer2.W.get_value())
    layer2new.b.set_value(layer2.b.get_value())
    layer3_input = layer2new.output.flatten(2)
    layer3new = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=layer3.n_in,
        n_out=layer3.n_out,#was 50, isn't this batch_size? nope no. hidden units
        activation=T.tanh
    )
    layer3new.W.set_value(layer3.W.get_value())
    layer3new.b.set_value(layer3.b.get_value())
    layer4new = LogisticRegression(rng,input=layer3new.output, n_in=layer3.n_out, n_out=2)
    layer4new.W.set_value(layer4.W.get_value())
    layer4new.b.set_value(layer4.b.get_value())
    test_model = theano.function(
        [index],
        [layer4new.y_pred,y,layer4new.p_y_given_x,x,layer0.W,layer1.W,layer2.W],
        givens={
            x: test_set_x[0:testNumber,...],
            y: test_set_y[0:testNumber]
        },
        on_unused_input='warn'
    )
    predicted_values = test_model(1)
    ''',y,p_y_given_x,x'''
    filter0_ = numpy.array(predicted_values[4])[0,0,...]
    filter0 = filter0_/(abs(filter0_).max()/255.0)
    filter01_ = numpy.array(predicted_values[4])[1,0,...]
    filter01 = filter01_/(abs(filter01_).max()/255.0)
    filter02_ = numpy.array(predicted_values[4])[2,0,...]
    filter02 = filter02_/(abs(filter02_).max()/255.0)
    filter03_ = numpy.array(predicted_values[4])[3,0,...]
    filter03 = filter03_/(abs(filter03_).max()/255.0)
    filter04_ = numpy.array(predicted_values[4])[4,0,...]
    filter04 = filter04_/(abs(filter04_).max()/255.0)
    filter05_ = numpy.array(predicted_values[4])[5,0,...]
    filter05 = filter05_/(abs(filter05_).max()/255.0)
    filter06_ = numpy.array(predicted_values[4])[6,0,...]
    filter06 = filter06_/(abs(filter06_).max()/255.0)
    filter07_ = numpy.array(predicted_values[4])[7,0,...]
    filter07 = filter07_/(abs(filter07_).max()/255.0)
    filter08_ = numpy.array(predicted_values[4])[8,0,...]
    filter08 = filter08_/(abs(filter08_).max()/255.0)
    filter09_ = numpy.array(predicted_values[4])[9,0,...]
    filter09 = filter09_/(abs(filter09_).max()/255.0)
    
    filter1_ = numpy.array(predicted_values[5])[0,0,...]
    filter1 = filter0_/(abs(filter0_).max()/255.0)
    filter11_ = numpy.array(predicted_values[5])[1,0,...]
    filter11 = filter11_/(abs(filter11_).max()/255.0)
    filter12_ = numpy.array(predicted_values[5])[2,0,...]
    filter12 = filter12_/(abs(filter12_).max()/255.0)
    filter13_ = numpy.array(predicted_values[5])[3,0,...]
    filter13 = filter13_/(abs(filter13_).max()/255.0)
    filter14_ = numpy.array(predicted_values[5])[4,0,...]
    filter14 = filter14_/(abs(filter14_).max()/255.0)
    filter15_ = numpy.array(predicted_values[5])[5,0,...]
    filter15 = filter15_/(abs(filter15_).max()/255.0)
    filter16_ = numpy.array(predicted_values[5])[6,0,...]
    filter16 = filter16_/(abs(filter16_).max()/255.0)
    filter17_ = numpy.array(predicted_values[5])[7,0,...]
    filter17 = filter17_/(abs(filter17_).max()/255.0)
    filter18_ = numpy.array(predicted_values[5])[8,0,...]
    filter18 = filter18_/(abs(filter18_).max()/255.0)
    filter19_ = numpy.array(predicted_values[5])[9,0,...]
    filter19 = filter19_/(abs(filter19_).max()/255.0)
    
    filter2_ = numpy.array(predicted_values[6])[0,0,...]
    filter2 = filter2_/(abs(filter2_).max()/255.0)
    filter21_ = numpy.array(predicted_values[6])[1,0,...]
    filter21 = filter21_/(abs(filter21_).max()/255.0)
    filter22_ = numpy.array(predicted_values[6])[2,0,...]
    filter22 = filter22_/(abs(filter22_).max()/255.0)
    filter23_ = numpy.array(predicted_values[6])[3,0,...]
    filter23 = filter23_/(abs(filter23_).max()/255.0)
    filter24_ = numpy.array(predicted_values[6])[4,0,...]
    filter24 = filter24_/(abs(filter24_).max()/255.0)
    filter25_ = numpy.array(predicted_values[6])[5,0,...]
    filter25 = filter25_/(abs(filter25_).max()/255.0)
    filter26_ = numpy.array(predicted_values[6])[6,0,...]
    filter26 = filter26_/(abs(filter26_).max()/255.0)
    filter27_ = numpy.array(predicted_values[6])[7,0,...]
    filter27 = filter27_/(abs(filter27_).max()/255.0)
    filter28_ = numpy.array(predicted_values[6])[8,0,...]
    filter28 = filter28_/(abs(filter28_).max()/255.0)
    filter29_ = numpy.array(predicted_values[6])[9,0,...]
    filter29 = filter29_/(abs(filter29_).max()/255.0)
    
    totFilter0 = numpy.hstack([filter0,filter01,filter02,filter03,filter04,filter05,filter06,filter07,filter08,filter09])
    totFilter1 = numpy.hstack([filter1,filter11,filter12,filter13,filter14,filter15,filter16,filter17,filter18,filter19])
    totFilter2 = numpy.hstack([filter2,filter21,filter22,filter23,filter24,filter25,filter26,filter27,filter28,filter29])
    totFilter = numpy.vstack([totFilter0,totFilter1,totFilter2])
    plt.imshow(totFilter, cmap = cm.Greys_r, interpolation='nearest')
    plt.show()
    
    #print(layer3.output)
    
    print(test_set_x.shape)
    print ("Predicted values for the first 2 examples in test set:")
    y = predicted_values[1]
    print(numpy.array(range(predicted_values[2].shape[0])))
    #print([0:(predicted_values[2].shape[0])])
    print(numpy.transpose(numpy.array(range(predicted_values[2].shape[0]))).shape)
    CountTrans = numpy.transpose(numpy.array(range(predicted_values[2].shape[0])))
    print(CountTrans.shape[0])
    CountTrans = CountTrans.reshape(CountTrans.shape[0],1)
    #CountTrans.dimshuffle('x', 0)
    print(CountTrans.shape)
    predPrint = numpy.hstack([predicted_values[2],CountTrans])
    print(predPrint)
    print(predicted_values[0])
    print('test error = '+str(sum(predicted_values[0]!=y)/y.shape[0]))
    print('Actual values:')
    print(y)
    print('here')
    print(predicted_values[2][:,y[0]].shape)
    print(validHolder)
    plt.plot(validHolder)
    plt.plot(trainHolder)
    plt.show()
    return(predicted_values[2][:,y[0]],validHolder)
    
    
def inspect_inputs(i, node, fn):
    print( i, node, "input(s) value(s):", [input[0] for input in fn.inputs],)

def inspect_outputs(i, node, fn):
    print( "output(s) value(s):", [output[0] for output in fn.outputs])    
def evaluate_lenet5(learning_rate=0.01, n_epochs=12,
                    dataset='dataset3.pkl',
                    nkerns=[20, 50,50], batch_size=10,
                    L1Value=0.00005,L2Value=0.0003):#nkerns should be 20,50,50, was 2,2,2 then 5,5,5 (slower cz more weights)
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
    basePath = r'C:\Users\Matt\Desktop\DogProj\data'
    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print('train set x')
    print(numpy.max(train_set_x.get_value(borrow=True)))
    print(numpy.min(train_set_x.get_value(borrow=True)))
    print((train_set_x.get_value(borrow=True)).sum()/((train_set_x.get_value(borrow=True).shape[1])*(train_set_x.get_value(borrow=True).shape[0])))
    #for L in range(train_set_x.get_value(borrow=True).shape[1]):
    #    train_set_x.set_value(train_set_x.get_value(borrow=True)[L,...]-numpy.mean(train_set_x.get_value(borrow=True)[L,...]))
    #train_set_x.set_value(train_set_x.get_value(borrow=True)-numpy.mean(train_set_x.get_value(borrow=True)))
    a = (train_set_x.get_value(borrow=True) > 0)#.astype(float)
    b = (train_set_x.get_value(borrow=True) < 0)#.astype(float)
    #train_set_x.set_value(a)
    valid_set_x.set_value(valid_set_x.get_value(borrow=True)-numpy.mean(valid_set_x.get_value(borrow=True)))
    a = (valid_set_x.get_value(borrow=True) > 0)#.astype(float)
    b = (valid_set_x.get_value(borrow=True) < 0)#.astype(float)
    #valid_set_x.set_value(a)
    test_set_x.set_value(test_set_x.get_value(borrow=True)-numpy.mean(test_set_x.get_value(borrow=True)))
    a = (test_set_x.get_value(borrow=True) > 0)#.astype(float)
    b = (test_set_x.get_value(borrow=True) < 0)#.astype(float)
    #test_set_x.set_value(a)
    print(numpy.max(train_set_x.get_value(borrow=True)))
    print(numpy.min(train_set_x.get_value(borrow=True)))
    print((train_set_x.get_value(borrow=True)).sum()/((train_set_x.get_value(borrow=True).shape[1])*(train_set_x.get_value(borrow=True).shape[0])))
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
    print( '... building the model')
    finalSize = 200
   
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, finalSize, finalSize))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, finalSize, finalSize),
        filter_shape=(nkerns[0], 1, 9, 9), #5,5 before
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 96, 96),
        filter_shape=(nkerns[1], nkerns[0], 9, 9),
        poolsize=(2, 2)
    )
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 44, 44),
        filter_shape=(nkerns[2], nkerns[1], 9, 9),
        poolsize=(2, 2)
    )
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 18 * 18,
        n_out=81,#was 50, isn't this batch_size? nope no. hidden units
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(rng,input=layer3.output, n_in=81, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = (layer4.negative_log_likelihood(y)+L2Value*(layer0.L2+layer0.L2+layer3.L2+layer4.L2))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params+ layer3.params + layer2.params + layer1.params + layer0.params

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
        [cost,layer4.p_y_given_x,layer4.y_pred,layer0.W,layer1.W,
        layer2.W,layer3.W,layer4.W,layer0.output,layer4.b,layer4.p_y_given_x,
        y,layer4.errors(y),layer0.preOutput,layer1.preOutput,layer2.preOutput,
        layer0.output,layer2.output,layer3.preOutput,layer4.preOutput,
        layer4.W,layer4.b,layer4.input],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ''',
        mode=theano.compile.MonitorMode(
                        pre_func=inspect_inputs,
                        post_func=inspect_outputs)'''
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print( '... training')
    # early-stopping parameters
    patience = 500#10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9995#0.995  # a relative improvement of this much is
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
    counter3 = 0
    counter4 = 0
    filterHolder = []
    validHolder = []
    trainHolder = []
    costHolder = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        costHolder = []
        for minibatch_index in range(int(n_train_batches)):
            
            iter = (epoch - 1) * n_train_batches + minibatch_index
            print(iter)
            print(epoch)
            if iter % 100 == 0:
                print( 'training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            print()
            #print(cost_ij)
            #print(numpy.array(cost_ij[3])[0,0,...])
            #print(numpy.array(cost_ij[3])[1,0,...])
            #print('shape')
            #print(numpy.array(cost_ij[3])[0,0,...])
            filter0_ = numpy.array(cost_ij[3])[0,0,...]
            filter0 = filter0_/(filter0_.max()/255.0)
            #filter0 = filter0_
            filter01_ = numpy.array(cost_ij[3])[1,0,...]
            filter01 = filter01_/(filter01_.max()/255.0)
            #filter0 = filter0_
            #print('filter0.shape')
            #print(filter0.shape)
            filter1_ = numpy.array(cost_ij[4])[0,0,...]
            filter1 = filter1_/(filter1_.max()/255.0)
            #filter1 =filter1_
            filter11_ = numpy.array(cost_ij[4])[1,0,...]
            filter11 = filter11_/(filter11_.max()/255.0)
            #filter1 =filter1_
            filter2_ = numpy.array(cost_ij[5])[0,0,...]
            filter2 = filter2_/(filter2_.max()/255.0)
            filter21_ = numpy.array(cost_ij[5])[1,0,...]
            filter21 = filter21_/(filter21_.max()/255.0)
            #filter2 =filter2_
            hiddenW_ = numpy.array(cost_ij[6])[:,0]
            hiddenW = hiddenW_/(hiddenW_.max()/255.0)
            #hiddenW=hiddenW_
            logRWT_ = numpy.array(cost_ij[7])[:,0]
            logRWT = logRWT_/(logRWT_.max()/255.0)
            #logRWT=logRWT_
            logRWF_ = numpy.array(cost_ij[7])[:,1]
            logRWF = logRWF_/(logRWF_.max()/255.0)
            #logRWF=logRWF_
            hiddenW = numpy.reshape(hiddenW[0:81],[9,9])
            logRWT = numpy.reshape(logRWT[0:81],[9,9])#was [5,5]
            logRWF = numpy.reshape(logRWF[0:81],[9,9])
            #gradientP = numpy.array(cost_ij[7])
            
            #print('shapes')
            #print(hiddenW.shape)
            #print(logRWT.shape)
            #filter1.resize([filter0.shape[0],filter0.shape[1]])
            #print(filter1)
            #print()
            filter1 = numpy.vstack([filter1,numpy.zeros([filter0.shape[0]-filter1.shape[0],filter1.shape[1]])])
            filter1 = numpy.hstack([filter1,numpy.zeros([filter1.shape[0],filter0.shape[1]-filter1.shape[1]])])
            filter2 = numpy.vstack([filter2,numpy.zeros([filter0.shape[0]-filter2.shape[0],filter2.shape[1]])])
            filter2 = numpy.hstack([filter2,numpy.zeros([filter2.shape[0],filter0.shape[1]-filter2.shape[1]])])
            hiddenW = numpy.vstack([hiddenW,numpy.zeros([filter0.shape[0]-hiddenW.shape[0],hiddenW.shape[1]])])
            hiddenW = numpy.hstack([hiddenW,numpy.zeros([hiddenW.shape[0],filter0.shape[1]-hiddenW.shape[1]])])
            logRWT = numpy.vstack([logRWT,numpy.zeros([filter0.shape[0]-logRWT.shape[0],logRWT.shape[1]])])
            logRWT = numpy.hstack([logRWT,numpy.zeros([logRWT.shape[0],filter0.shape[1]-logRWT.shape[1]])])
            logRWF = numpy.vstack([logRWF,numpy.zeros([filter0.shape[0]-logRWF.shape[0],logRWF.shape[1]])])
            logRWF = numpy.hstack([logRWF,numpy.zeros([logRWF.shape[0],filter0.shape[1]-logRWF.shape[1]])])
            totFilter = numpy.hstack([filter0,filter1,filter2,hiddenW,logRWT,logRWF])
            totlayer2 = numpy.hstack([filter01,filter11,filter21,numpy.zeros([9,3*9])])
            totFilter = numpy.vstack([totFilter,totlayer2])
            '''
            print('preOutput1')
            print(numpy.mean(abs(numpy.array(cost_ij[13]))))
            print(numpy.mean(abs(numpy.array(cost_ij[14]))))
            print(numpy.mean(abs(numpy.array(cost_ij[15]))))
            print('preOutput3')
            print(numpy.mean(abs(numpy.array(cost_ij[18]))))
            
            print('postlayer0')
            print(numpy.mean(abs(numpy.array(cost_ij[16]))))
            print('postlayer2')
            print(numpy.mean(abs(numpy.array(cost_ij[17]))))
            print('preOutput4')
            print((numpy.array(cost_ij[19])))
            print('layer4W,layer4B,layer4input')
            #print((numpy.array(cost_ij[20])))
            #print((numpy.array(cost_ij[21])))
            print((numpy.array(cost_ij[22])))
            '''
            #totFilter = numpy.array(cost_ij[8][0,0,...])
            #print(filter0)
            #plt.imshow(filter0, cmap = cm.Greys_r,interpolation="nearest")
            #plt.show()
            filterHolder.append(totFilter)#=filter0
            costHolder.append(numpy.mean(cost_ij[12]))
            if iter>1:
                a=1
                '''
                #print(filterHolder[int(iter-1)][0])
                print('abs values')
                ##print(filter0_)
                ##print(filter1_)
                ##print(numpy.reshape(hiddenW_[0:81],[9,9]))
                print(numpy.reshape(logRWT_[0:81],[9,9]))
                print(numpy.array(cost_ij[9]).shape)
                print((numpy.array(cost_ij[9])[0]))
                print('end abs values')
                print('p_y_given_x')
                print(numpy.array(cost_ij[10]))
                print(numpy.array(cost_ij[11]))
                
                print(iter)
                #print(len(filterHolder))
                ##print(filterHolder[int(iter)][0:9,0:9]-filterHolder[int(iter)-1][0:9,0:9])
                ##print(filterHolder[int(iter)][0:9,9:18]-filterHolder[int(iter)-1][0:9,9:18])
                #print(filterHolder[int(iter-1)][0:9,0:9].shape)
                #print(filterHolder[int(iter-1)][2].shape)
                #print(filterHolder[int(iter-1)][3].shape)
                ##print(filterHolder[int(iter)][0:9,18:27]-filterHolder[int(iter)-1][0:9,18:27])
                print(filterHolder[int(iter)][0:9,36:45]-filterHolder[int(iter)-1][0:9,36:45])
                '''
                print('cost')
                print(numpy.array(cost_ij[0]))
                print('p_y_given_x')
                print(numpy.array(cost_ij[10]))
                print((numpy.array(cost_ij[11])))
                
            counter4 +=1
                
            '''
            print('layer4.Wb')
            print(layer4.W.get_value())
            print(layer4.b.get_value())
            
            
            print('layer3.Wb')
            print(layer3.W.get_value())
            print(layer3.b.get_value())
            print('layer2.Wb')
            print(layer2.W.get_value())
            print(layer2.b.get_value())
            print('layer1.Wb')
            print(layer1.W.get_value())
            print(layer1.b.get_value())
            print('layer0.Wb')
            print(layer0.W.get_value())
            print(layer0.b.get_value())
            '''
            #print(layer4.input.eval())
            #print(layer4.p_y_given_x.eval({'input':layer4.input,'SelfW':layer4.W,'SelfB':layer4.b}))
            
            #x_printed = theano.printing.Print('this is a very important value')(x)

            #f = theano.function([x], x * 5)
            #f_with_print = theano.function([x], x_printed * 5)
            #assert numpy.all( f_with_print([1, 2, 3]) == [5, 10, 15])
            
            if (iter + 1) % validation_frequency == 0:
                
                if (counter3) % 5 == 0:
                    a=1
                    '''
                    filter01_ = numpy.array(cost_ij[3])[2,0,...]
                    filter01 = filter01_/(filter01_.max())
                    filterHolder.append(filter01)
                    totFilter = numpy.array(cost_ij[8][0,2,...])
                    filterHolder.append(totFilter)
                    totFilter = numpy.array(cost_ij[8][1,2,...])
                    filterHolder.append(totFilter)
                    totFilter = numpy.array(cost_ij[8][2,2,...])
                    filterHolder.append(totFilter)
                    totFilter = numpy.array(cost_ij[8][3,2,...])
                    filterHolder.append(totFilter)
                    totFilter = numpy.array(cost_ij[8][4,2,...])
                    filterHolder.append(totFilter)
                    '''
                    #totFilter = numpy.array(cost_ij[8][5,0,...])
                    #filterHolder.append(totFilter)
                counter3 +=1    
                    
                    
                    
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                validHolder.append(this_validation_loss)
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
                    best_params = params
                    
                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(int(n_test_batches))
                    ]
                    test_score = numpy.mean((test_losses))
                    print(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    with open(os.path.join(basePath,'best_model.pkl'), 'wb') as f:
                        pickle.dump([layer4,layer3,layer2,layer1,layer0,validHolder,trainHolder], f)
              
            if iter >150000:
                break
            if patience <= iter:
                a=1
                #done_looping = True
                #break
        trainHolder.append(sum(costHolder)/len(costHolder)) 
        print('TrainHolder : ')
        print(trainHolder)
        with open(os.path.join(basePath,'final_model.pkl'), 'wb') as f:
            pickle.dump([layer4,layer3,layer2,layer1,layer0,validHolder,trainHolder], f)
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print((params))
    print('Valid Holder')
    print(validHolder)
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
    #print(filterHolder)
    plt.show()
    
if __name__ == '__main__':
    evaluate_lenet5()
    #pred,validHolder = predict(100)
    #heatMap()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)