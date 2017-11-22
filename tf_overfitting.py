# Toy with overfitting, see how to deal with overfitting using tensorflow dropout
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

#define layer
def add_layer(inputs, in_size,out_size,keep_prob,activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	G = tf.matmul(inputs,weights) + biases
	#dropout
	G = tf.nn.dropout(G,keep_prob)
	
	if activation_function == None:
		outputs = G
	else:
		outputs = activation_function(G)
		tf.summary.histogram('outputs',outputs)
	return outputs
	
#get input
x_input = tf.placeholder(tf.float32,[None, 64]) #8x8
y_input = tf.placeholder(tf.float32,[None, 10])

#Dropout: How many percents of data you want to keep after each training layers
keep_prob = tf.placeholder(tf.float32)

#build nn
l1 = add_layer(x_input,64,50,keep_prob,tf.nn.tanh)
prediction = add_layer(l1,50,10,keep_prob,tf.nn.softmax)

#loss func
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_input*tf.log(prediction),
				reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)

#optimizer
training = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#merge summary
merged = tf.summary.merge_all()

#prediction
sess = tf.Session()

## summary writer
train_writer = tf.summary.FileWriter('logs/train',sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())	
for i in range(1000):
	sess.run(training, feed_dict={x_input:X_train,y_input:y_train,keep_prob:0.6})
	if i % 50 == 0:
		#Advantage of placeholder: feed whatever data you want, end the code will do all computation 
		#base on your new feed_dict
		train_result = sess.run(merged, feed_dict={x_input:X_train,y_input:y_train,keep_prob:1})
		test_result = sess.run(merged, feed_dict={x_input:X_test,y_input:y_test,keep_prob:1})
		train_writer.add_summary(train_result,i)
		test_writer.add_summary(test_result,i)
		
	

