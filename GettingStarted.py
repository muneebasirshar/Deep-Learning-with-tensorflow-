
import tensorflow as tf     #importing  tensor flow librarry
hello = tf.constant('hello,Muneeba')  # storing a strong as constant
sess = tf.Session() # creating a tensorflow session
print(sess.run(hello)) # sess.run is responsible for running the session

#PLACEHOLDERS
#Placeholders are the terminals/data point through which we will be feeding data into the network we will build. Its like gate point for our input and output data.

inputs=tf.placeholder('float',[None,2],name='Input') #the data type that will feed to the placeholder is float , 2nd parameter is the shape of input.
targets=tf.placeholder('float',name='Target')

#Now lets say we don’t know how many input set we are going to feed at the same time. it can be 1 it can be 100 sets at a time. So we specify that with None.
#if we set the input shape of the placeholder as [None,3] then it will be able to take any sets of data in a single shot. that is what we did in out earlier code.
#In the second case I didn’t select any shape, in this case it will accept any shape. but there is a chance of getting error in runtime if the data the network is
## expecting has a different shape than the data we provided in the placeholder.
#the third parameter name will help us to identify nodes on tenoorboard

#VARIABLES
#Variables in tensorflow are different from regular variables, unlike placeholders we will never set any specific values in these, nor use it for storing any data.

weight1=tf.Variable(tf.random_normal(shape=[2,3],stddev=0.02),name="Weight1")
biases1=tf.Variable(tf.random_normal(shape=[3],stddev=0.02),name="Biases1") #it has only output connections no input. that is it is the inpyt itself

#in this case we initialized the first one as 2×3 matrix with random values with variation of +/- 0.02. for the shape the first one is the number of input connection
## that the layer will receive. and the 2nd one is the number of output connection that the layer will produce for the next layer.

hLayer=tf.matmul(inputs,weight1)
hLayer=hLayer+biases1
#So here is two matrix operations first one is matrix multiplications of inputs and the weights1 matrix which will produce the output without bias. and in the next line
#we did matrix addition which basically performed an element by element additions.

hLayer=tf.sigmoid(hLayer, name='hActivation') #writing an activation function
#creating the output layer
weight2=tf.Variable(tf.random_normal(shape=[3,1],stddev=0.02),name="Weight2")
biases2=tf.Variable(tf.random_normal(shape=[1],stddev=0.02),name="Biases2")
#As we can see the output layer has only 1 neuron and the previous hidden layer had 3 neurons.
#So the weight matrix has 3 input and 1 output thus the shape is [3,1]. and the bias has only one neuron to connect to thus bias shape in [1]

output=tf.matmul(hLayer,weight2)
output=output+biases2
output=tf.sigmoid(output, name='outActivation')
#cost=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=targets)
# update  tf.nn.softmax_cross_entropy_with_logits is for classification problems only
# we will be using tf.squared_difference()
cost=tf.squared_difference(targets, output)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer().minimize(cost)

#tf.squared_difference  takes two input fitsr is the target value and second is the predicted value.

#SESSION
#generating inputs
import numpy as np

inp=[[0,0],[0,1],[1,0],[1,1]]
out=[[0],[1],[1],[0]]

inp=np.array(inp)
out=np.array(out)

epochs=4000 # number of time we want to repeat

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        inp=[[a,b]]
        inp=np.array(inp)
        prediction=sess.run([output],feed_dict={inputs: inp})
        print(prediction)

aver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
    saver.save(sess, "model.ckpt") # saving the session with file name "model.ckpt"


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        inp=[[a,b]]
        inp=np.array(inp)
        prediction=sess.run([output],feed_dict={inputs: inp})
        print(prediction)