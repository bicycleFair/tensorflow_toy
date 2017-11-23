import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#CNN:
#Define weight,bias,conv,pool
#Build CNN (convolutional layers and network layers)


mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
	
#Compute accuracy
def compute_accuracy(images,labels):
	global prediction
	pred_y = sess.run(prediction, feed_dict={input_x:images,keep_prob:1})
	correct_ones = tf.equal(tf.argmax(pred_y,1),tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_ones, tf.float32))
	result = sess.run(accuracy,feed_dict={input_x:images,input_y:labels,keep_prob:1})
	return result
	
	
#Define something for CNN
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
	
def conv2d(x, W):
	# stride [1, x_movement, y_movement, 1]
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding="SAME")
	
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
	
#Define placeholder for inputs to network
input_x = tf.placeholder(tf.float32,[None,784]) #28x28
input_y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#Reshape input_x data ==> [n_samples,28,28,1] ==> 2d to 4d
x_images = tf.reshape(input_x, [-1,28,28,1])

# Conv1 layer. Q: How to increase deep?
W_conv1 = weight_variable([5,5,1,32]) # patch: 5x5 input_deep: 1 output_deep: 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_images,W_conv1) + b_conv1) # output size: 28x28
h_pool1 = max_pool_2x2(h_conv1)							# output size: 14x14

# Conv2 layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output size: 14x14
h_pool2 = max_pool_2x2(h_conv2)							 #output size: 7x7

# Func1 layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#[n_samples,7,7,64] ==> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# Func2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

#Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(input_y*tf.log(prediction), reduction_indices = [1]))

training = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#Run 1000 times using SGD
for i in range(1000):
	batch_x,batch_y = mnist.train.next_batch(100)
	sess.run(training, feed_dict={input_x:batch_x,input_y:batch_y, keep_prob:1})
	#Test the performance after every 50 times
	if i%50 == 0:
		print(compute_accuracy(mnist.test.images,mnist.test.labels))
	
	
	
	
	
	
	
	
