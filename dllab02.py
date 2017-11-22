import tensorflow as tf
import numpy as np
import os
import gzip
import pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

#Load MNIST (reused from assignment 1)
def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

	#separating data into trainining, validation and test dataset
    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    return rval

# load
Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
X_valid, y_valid = Dval

n_train_samples = X_train.shape[0]
train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
X_train = X_train[train_idxs]
y_train = y_train[train_idxs]

#defining network following tutorial 
#https://www.tensorflow.org/tutorials/layers
"""
SKELETON
def model_fn(features, labels, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
"""
def cnn_model_fn(features, labels, mode, params):
	"""Model function for CNN."""
	num_filters, learning_rate = params["num_filters"], params["learning_rate"]
	kernel = [3, 3]
	#network architecture
	
    # Input Layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
	conv1 = tf.layers.conv2d(inputs=input_layer, filters=num_filters, kernel_size= kernel, padding="same", activation=tf.nn.relu)

    # Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

    # Convolutional Layer #2 
	conv2 = tf.layers.conv2d(inputs=pool1, filters=num_filters, kernel_size=kernel, padding="same", activation=tf.nn.relu)
    
	# Pooling Layer #2
	
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)

    # Dense Layer (fully connected)
	pool2_flat = tf.reshape(pool2, [-1, 26 * 26 * num_filters])
	dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    # Logits Layer
	output = tf.layers.dense(inputs=dense, units=10)

  

    #2. calculate losses cross entropy with sgd for training and evaluation
	onehot_y = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
	cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_y, logits=output)

    # #3. define the training operation/optimizer
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss=cross_entropy_loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy_loss, train_op=train_op)

	#4 generate predictions
	predictions = {"classes": tf.argmax(input=output, axis=1), "probabilities": tf.nn.softmax(output, name="softmax_tensor")}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	#5 Add evaluation metrics 
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	specs = tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy_loss, eval_metric_ops=eval_metric_ops)
	return specs

##########################	
# CHANGING LEARNING RATE #
##########################
rates = [0.1, 0.01, 0.001, 0.0001]
max_epochs = 100
epochs = np.arange(max_epochs)
default_filter = 16

#plotting stuff
fig01 = plt.figure()
plt.title("Validation performance for different learning rates (GPU ver.)", fontsize=18)
plt.xlabel("Epoch #", fontsize = 16)
plt.ylabel("Validation Loss", fontsize = 16)
plt.grid()
color = ['r', 'g', 'b', 'magenta']
l1, l2, l3, l4 = mpatches.Patch(color='r', label='rate = 0.1'), mpatches.Patch(color='g', label='rate = 0.01'), mpatches.Patch(color='b', label='rate = 0.001'), mpatches.Patch(color='magenta', label='rate = 0.0001')
plt.legend(handles=[l1, l2, l3, l4])

for i in range(len(rates)):
    # initializing estimator
    mnist_cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_rates"+str(i), params ={"num_filters": default_filter, "learning_rate": rates[i]})
    losses = np.zeros(max_epochs)

    for ep in range(max_epochs):
        # train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train}, y=y_train, batch_size=100, num_epochs=1, shuffle=True)
        mnist_cnn.train(input_fn=train_input_fn, steps=1)
		
		#evaluate
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_valid}, y=y_valid, num_epochs=1, shuffle=False)
        losses[ep] = mnist_cnn.evaluate(input_fn=eval_input_fn)["loss"]
        print("Epoch {}, validation loss {}" .format(ep, losses[ep]))

    print("Learning rate: {}." .format(rates[i]))
    plt.plot(epochs, losses, c=color[i])
#plt.savefig('rates.pdf')
plt.show()


###################
#       RUNTIME   #
###################
'''

with tf.device("/device:GPU:0"):
    gpu_filters = [8, 16, 32, 64, 128, 256]
    times = []

    for i in range(len(gpu_filters)):
        # Create the Estimator
        mnist_cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_time_gpu" + str(i), params ={"num_filters": gpu_filters[i], "learning_rate": 0.1})
        start_time= time.time()
        max_epochs = 20

        for ep in range(max_epochs):
            # Train the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train}, y=y_train, batch_size=100, num_epochs=1, shuffle=True)
            mnist_cnn.train(input_fn=train_input_fn, steps=1)

        end_time = time.time()
        times.append(end_time - start_time)
        print("Filters: {}, Runtime: {} ms" .format(gpu_filters[i], end_time - start_time))

    #plotting
	plt.figure()
	plt.title("GPU training time", fontsize = 18)
	plt.xlabel("Filter Depth", fontsize = 16)
	plt.ylabel("Training time", fontsize = 16)
	l1 = mpatches.Patch(color='g', label='GPU times')
	plt.legend(handles=[l1])
	plt.scatter(gpu_filters, times, s = 80, c='g')
	plt.grid()
	plt.savefig("gpuruntime.pdf")
	plt.show()

#CPU version
"""
session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)

with tf.Session(config=session_conf) as sess:
"""
with tf.device('/device:CPU:0'):
    cpu_filters = [8, 16, 32, 64]
    times = []

    for i in range(len(cpu_filters)):
        # Create the Estimator
        mnist_cnn = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_time_cpu" + str(i), params ={"num_filters": cpu_filters[i], "learning_rate": 0.1})
        start_time= time.time()
        max_epochs = 20

        for ep in range(max_epochs):
            train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train}, y=y_train, batch_size=100, num_epochs=1, shuffle=True)
            mnist_cnn.train(input_fn=train_input_fn, steps=1)

        end_time = time.time()
        times.append(end_time - start_time)
        print("Filters: {}, Runtime: {} ms" .format(cpu_filters[i], end_time - start_time))
       
    #plotting
	plt.figure()
	plt.title("CPU training time", fontsize = 18)
	plt.xlabel("Filter Depth", fontsize = 16)
	plt.ylabel("Training time", fontsize = 16)
	l1 = mpatches.Patch(color='b', label='CPU times')
	plt.legend(handles=[l1])
	plt.scatter(gpu_filters, times, s = 80, c='b')
	plt.grid()
	plt.savefig("cpuruntime.pdf")
	plt.show()
'''

"""
#merging data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cpu_nums = [8, 16, 32, 64]
cpu_times = [38.54977607727051, 39.10606122016907, 41.242592573165894, 47.831615924835205]
gpu_nums = [8, 16, 32, 64, 128, 256]
gpu_times = [43.43289303779602, 41.314709186553955, 42.302164793014526, 44.25623297691345, 47.875519037246704, 54.78205466270447]
plt.figure()
plt.title("Training time", fontsize = 18)
plt.xlabel("Filter Depth", fontsize = 16)
plt.ylabel("Training time, ms", fontsize = 16)
l1 = mpatches.Patch(color='b', label='CPU times')
l2 = mpatches.Patch(color='g', label='GPU times')
plt.legend(handles=[l1, l2])
plt.scatter(cpu_nums, cpu_times, s = 80, c='b')
plt.scatter(gpu_nums, gpu_times, s = 80, c='g')
plt.grid()
plt.show()
"""