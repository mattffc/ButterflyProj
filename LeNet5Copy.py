@@ -110,20 +110,25 @@ class LeNetConvPoolLayer(object):
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

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
        
def heatMap(testImageNo = 32,filterFactor=6,dataset='dataset3.pkl'):#filtersize must be odd
def heatMap(testImageNo = 78,filterFactor=2,dataset='dataset3.pkl'):#filtersize must be odd
    #create dataset with shifted block
    #predictedValues = predict(length,newDataset)
    #create numpy array that is the size of the image - edge reduction
@@ -138,20 +143,21 @@ def heatMap(testImageNo = 32,filterFactor=6,dataset='dataset3.pkl'):#filtersize 
    print(numpy.sqrt(test_image.shape[0]))
    test_image = test_image.reshape(numpy.sqrt(test_image.shape[0]),numpy.sqrt(test_image.shape[0]))
    filterSize = test_image.shape[0]/filterFactor
    coarseness = 15
    if filterSize % 2 == 0:
     filterSize -= 1
    blockData = numpy.zeros([int((test_image.shape[0])/filterSize)**2,test_image.shape[0]**2])
    blockData = numpy.zeros([int((test_image.shape[0])/coarseness)**2,test_image.shape[0]**2])
    print(test_image.shape[0])
    
    for i in range(int((test_image.shape[0])/filterSize)):
    for i in range(int((test_image.shape[0])/coarseness)):
        print(i)
        for k in range(int((test_image.shape[0])/filterSize)):
        for k in range(int((test_image.shape[0])/coarseness)):
            print(k)
            test_image_block = copy.deepcopy(test_image)
            test_image_block[i*filterSize:i*filterSize+((filterSize)),k*filterSize:k*filterSize+((filterSize))]=0
            test_image_block[i*coarseness:i*coarseness+((filterSize)),k*coarseness:k*coarseness+((filterSize))]=0
            # #plt.imshow(test_image_block, cmap = cm.Greys_r, interpolation='nearest')
            # #plt.show()
            blockData[i*int((test_image.shape[1])/filterSize)+k,...]=test_image_block.reshape(test_image.shape[0]**2)
            blockData[i*int((test_image.shape[1])/coarseness)+k,...]=test_image_block.reshape(test_image.shape[0]**2)
    dataset = [(0,0),(0,0),(blockData,numpy.ones(blockData.shape[0]))]#ones or zeros depends on class of image
    globalPath = r'C:\Users\Matt\Desktop\DogProj\scripts\DogProjScripts'
    pickle.dump( dataset, open( os.path.join(globalPath,"blockData.pkl"), "wb" ) ) # needed
@@ -162,7 +168,7 @@ def heatMap(testImageNo = 32,filterFactor=6,dataset='dataset3.pkl'):#filtersize 
    #b = (imgTest_image.resize([(predictedValues.shape[0]),(predictedValues.shape[0])]))
    #plt.imshow(b, cmap = cm.Greys_r, interpolation='nearest')
    #plt.show()
    predictedValues = predict(blockData.shape[0],dataset='blockData.pkl')
    predictedValues,validHolder = predict(blockData.shape[0],dataset='blockData.pkl')
    print('pred')
    print(predictedValues.shape)
    predictedValues=predictedValues.reshape(numpy.sqrt(predictedValues.shape[0]),numpy.sqrt(predictedValues.shape[0]))
@@ -219,7 +225,7 @@ def predict(testNumber,dataset='dataset3.pkl'):
    y = T.ivector('y')
    # load the saved model
    #layer4 = pickle.load(open('best_model.pkl'))
    f = open('final_model.pkl', 'rb')
    f = open('final_model_26test_27valid.pkl', 'rb')
    layer4,layer3,layer2,layer1,layer0,validHolder = pickle.load(f) # 
    print('blah')
    print(numpy.array(layer0.W.get_value())[3,0,...])
@@ -352,8 +358,8 @@ def predict(testNumber,dataset='dataset3.pkl'):
    totFilter1 = numpy.hstack([filter1,filter11,filter12,filter13,filter14,filter15,filter16,filter17,filter18,filter19])
    totFilter2 = numpy.hstack([filter2,filter21,filter22,filter23,filter24,filter25,filter26,filter27,filter28,filter29])
    totFilter = numpy.vstack([totFilter0,totFilter1,totFilter2])
    #plt.imshow(totFilter, cmap = cm.Greys_r, interpolation='nearest')
    #plt.show()
    plt.imshow(totFilter, cmap = cm.Greys_r, interpolation='nearest')
    plt.show()
    
    #print(layer3.output)
    
@@ -376,17 +382,21 @@ def predict(testNumber,dataset='dataset3.pkl'):
    print(y)
    print('here')
    print(predicted_values[2][:,y[0]].shape)
    return(predicted_values[2][:,y[0]])
    print(validHolder)
    #plt.plot(validHolder)
    #plt.show()
    return(predicted_values[2][:,y[0]],validHolder)
    
    
def inspect_inputs(i, node, fn):
    print( i, node, "input(s) value(s):", [input[0] for input in fn.inputs],)

def inspect_outputs(i, node, fn):
    print( "output(s) value(s):", [output[0] for output in fn.outputs])    
def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
def evaluate_lenet5(learning_rate=0.01, n_epochs=20,
                    dataset='dataset3.pkl',
                    nkerns=[10, 20,20], batch_size=10,
                    L1Value=0.00001):#nkerns should be 20,50,50, was 2,2,2 then 5,5,5 (slower cz more weights)
                    nkerns=[20, 50,50], batch_size=10,
                    L1Value=0.00005,L2Value=0.0003):#nkerns should be 20,50,50, was 2,2,2 then 5,5,5 (slower cz more weights)
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
@@ -414,7 +424,9 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
    print(numpy.max(train_set_x.get_value(borrow=True)))
    print(numpy.min(train_set_x.get_value(borrow=True)))
    print((train_set_x.get_value(borrow=True)).sum()/((train_set_x.get_value(borrow=True).shape[1])*(train_set_x.get_value(borrow=True).shape[0])))
    train_set_x.set_value(train_set_x.get_value(borrow=True)-numpy.mean(train_set_x.get_value(borrow=True)))
    #for L in range(train_set_x.get_value(borrow=True).shape[1]):
    #    train_set_x.set_value(train_set_x.get_value(borrow=True)[L,...]-numpy.mean(train_set_x.get_value(borrow=True)[L,...]))
    #train_set_x.set_value(train_set_x.get_value(borrow=True)-numpy.mean(train_set_x.get_value(borrow=True)))
    a = (train_set_x.get_value(borrow=True) > 0)#.astype(float)
    b = (train_set_x.get_value(borrow=True) < 0)#.astype(float)
    #train_set_x.set_value(a)
@@ -507,7 +519,7 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
    layer4 = LogisticRegression(rng,input=layer3.output, n_in=81, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = (layer4.negative_log_likelihood(y)+(layer0.L1+layer0.L1+layer3.L1+layer4.L1)*L1Value)
    cost = (layer4.negative_log_likelihood(y)+L2Value*(layer0.L2+layer0.L2+layer3.L2+layer4.L2))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
@@ -546,7 +558,11 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,

    train_model = theano.function(
        [index],
        [cost,layer4.p_y_given_x,layer4.y_pred,layer0.W,layer1.W,layer2.W,layer3.W,layer4.W,layer0.output,layer4.b,layer4.p_y_given_x,y,layer4.errors(y)],
        [cost,layer4.p_y_given_x,layer4.y_pred,layer0.W,layer1.W,
        layer2.W,layer3.W,layer4.W,layer0.output,layer4.b,layer4.p_y_given_x,
        y,layer4.errors(y),layer0.preOutput,layer1.preOutput,layer2.preOutput,
        layer0.output,layer2.output,layer3.preOutput,layer4.preOutput,
        layer4.W,layer4.b,layer4.input],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
@@ -658,11 +674,30 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
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
            #filterHolder.append(totFilter)#=filter0
            filterHolder.append(totFilter)#=filter0
            costHolder.append(numpy.mean(cost_ij[12]))
            if iter>1:
                a=1
@@ -730,7 +765,7 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
                
                if (counter3) % 5 == 0:
                    a=1
                    
                    '''
                    filter01_ = numpy.array(cost_ij[3])[2,0,...]
                    filter01 = filter01_/(filter01_.max())
                    filterHolder.append(filter01)
@@ -744,7 +779,7 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
                    filterHolder.append(totFilter)
                    totFilter = numpy.array(cost_ij[8][4,2,...])
                    filterHolder.append(totFilter)
                    
                    '''
                    #totFilter = numpy.array(cost_ij[8][5,0,...])
                    #filterHolder.append(totFilter)
                counter3 +=1    
@@ -796,8 +831,8 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
        trainHolder.append(sum(costHolder)/len(costHolder)) 
        print('TrainHolder : ')
        print(trainHolder)
    with open('final_model.pkl', 'wb') as f:
        pickle.dump([layer4,layer3,layer2,layer1,layer0,validHolder], f)
        with open('final_model.pkl', 'wb') as f:
            pickle.dump([layer4,layer3,layer2,layer1,layer0,validHolder], f)
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print((params))
@@ -818,17 +853,17 @@ def evaluate_lenet5(learning_rate=0.1, n_epochs=50,
        #print(filterHolder[j])
        # return the artists set
        return im,
    #ani = animation.FuncAnimation(fig, updatefig, frames=len(filterHolder), 
    #                          interval=2000, blit=True, repeat=True)
    ani = animation.FuncAnimation(fig, updatefig, frames=len(filterHolder), 
                              interval=10, blit=True, repeat=True)
                              
    #plt.imshow(filterHolder[0], cmap = cm.Greys_r, interpolation="nearest")
    #print(filterHolder)
    #plt.show()
    plt.show()
    
if __name__ == '__main__':
    #evaluate_lenet5()
    #predict(130)
    heatMap()
    evaluate_lenet5()
    #pred,validHolder = predict(100)
    #heatMap()


def experiment(state, channel):