import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#get data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#define layer
def add_layer(inputs,in_size,out_size,activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	G = tf.matmul(inputs,weights) + biases
	if activation_function is None:
		outputs = G
	else:
		outputs = activation_function(G)
	return outputs
	
#define compute_accuracy
def compute_accuracy(images,labels):
	global prediction
	y_pre = sess.run(prediction,feed_dict={x_input:images})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy, feed_dict={x_input:images,y_input:labels})
	return result
	
#define placeholder for inputs to network [None,748]: None is unknown amount of sample, 784 pixels for one pic
x_input = tf.placeholder(tf.float32,[None,784])
y_input = tf.placeholder(tf.float32,[None,10])

#add output layer
prediction = add_layer(x_input, 784,10,activation_function=tf.nn.softmax)

#loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_input * tf.log(prediction),
								reduction_indices=[1]))
			
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)					
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
	#train part of the data sample each time -> reduce training time
	batch_x,batch_y = mnist.train.next_batch(100)
	sess.run(train,feed_dict={x_input:batch_x,y_input:batch_y})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))

