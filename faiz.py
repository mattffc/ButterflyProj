import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


total_num_data_points = 10000
input_feature_size = 4
hidden_layer_1_neurons = 400000

x_data = range(total_num_data_points)
#y_data = np.array(x_data)**2/(total_num_data_points**2)-.5+ 0.1*np.random.rand(total_num_data_points)#np.sin(x_data)*0.9 #+ 0.1*np.random.rand(total_num_data_points)
y_data = np.sin(x_data)*0.9-0.5
training_set = y_data[0:7000]
test_set = y_data[7000:total_num_data_points]

# Matrix of inputs first 4 columns here and output is final column
input_output_matrix = [[training_set[x],training_set[x+1],training_set[x+2],training_set[x+3], training_set[x+4]] for x in xrange(len(training_set) - 4 )]
    
# Shuffle rows of this matrix
input_output_matrix_form = np.array(input_output_matrix)
np.random.shuffle(input_output_matrix_form)
###
input_output_matrix_form = input_output_matrix_form.astype('float32')
###
# Random Number Generator
rng = np.random


N =  input_output_matrix_form.shape[0] # number of samples


epochs = 1
training_steps = N*epochs


# declare Theano symbolic variables
#x = T.vector('x')
#y = T.scalar('y')
batch_size = 10
x = T.matrix('x')
y = T.vector('y')
w_1 = theano.shared(rng.randn(batch_size,input_feature_size,hidden_layer_1_neurons), name='w1')
w_1.set_value(w_1.get_value()/1000) 
b_1 = theano.shared(np.zeros((hidden_layer_1_neurons,)), name='b1')
w_2 = theano.shared(rng.randn(batch_size,hidden_layer_1_neurons), name='w2')
b_2 = theano.shared(0., name='b2')
w_2.set_value(w_2.get_value()/1000) 
print 'Initial model:'
print w_1.get_value(), b_1.get_value()
print w_2.get_value(), b_2.get_value()


# Construct Theano expression graph
p_1 = T.tanh(-T.dot(T.tanh(-T.dot(x, w_1)-b_1), w_2)-b_2)

# probability that target = 1
# prediction = p_1 > 0.5 # the prediction threshold

#cost = xent.mean() + 0.01 * (w**2).sum() # the cost to minimize


cost = abs(T.mean((p_1 - y)))

# computing partial derivatives wrt weights + biases
gw_1, gb_1, gw_2, gb_2 = T.grad(cost, [w_1, b_1, w_2, b_2])

batch_size = 10
index = T.lscalar()
# Compile
input_output_matrix_form=theano.shared((input_output_matrix_form),
                                 borrow=True)
#shared_x = theano.shared(numpy.asarray(data_x,
#                                               dtype=theano.config.floatX),
#                                 borrow=borrow)
train = theano.function(
                    inputs = [index],
                    outputs = [p_1, abs(y-p_1),w_1,w_2],
                    updates = {w_1 : w_1-0.01*gw_1, b_1 : b_1-0.01*gb_1,
                               w_2 : w_2-0.01*gw_2, b_2 : b_2-0.01*gb_2},
                    #givens = {
                    #x: input_output_matrix_form[(index ): ((index + 1) )][0:4],
                    #y: input_output_matrix_form[index * batch_size: (index + 1) * batch_size][4]
                    #}
                    givens={
                    x: input_output_matrix_form[index * batch_size: (index + 1) * batch_size][0:4],
                    y: input_output_matrix_form[index * batch_size: (index + 1) * batch_size][4]
                    }
                    
                    )
                               
                               
predict = theano.function(
                    inputs = [x],
                    outputs = p_1)


# Train
e=0
i=0
predStore = []
while i < 600:
    i +=1
    pred, err,ww,w3 = train(i*10)
    
    predStore.append(err)
    if i % 10 == 0:
        #print np.mean(input_output_matrix_form[i][0:4])
        a=9
        print('err')
        print(np.mean(predStore))
        predStore = []
        #print('pred')
        #print(pred)
        #print('np.mean(ww)')
        #print(np.mean(ww))
        #print(np.mean(w3))
    if i == 5998 and e<1:
        i = 0
        e +=1
        print('err')
        print(err)
#print 'Final model:'
#print w_1.get_value(), b_1.get_value()
#print w_2.get_value(), b_2.get_value()





