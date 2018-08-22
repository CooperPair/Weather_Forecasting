import numpy as np
import pandas as pd
from pandas import Series
from pandas import read_csv
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import warnings


warnings.filterwarnings("ignore")
data = pd.read_csv("testset.csv")
#drop datetime variable
#data = data.drop(['datetime'], 1)

temperature = data[' _tempm']
temperature.replace(0, np.NaN)

temperature.fillna(temperature.mean(), inplace = True)

#print(temperature.shape)


#Making data a numpy array
temperature = temperature.values
temperature = temperature.reshape((100990,1))
n = temperature.shape[0]
#visualising dataset finding if missing value is there or not
'''
plt.plot(temperature)
plt.show()
'''
#Preparing training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1 
test_end = n
temp_train = temperature[train_start:train_end]
y_test = temperature[train_end+1:]
temp_test = temperature[train_end+1:]

# temp_train = temperature[np.arange(train_start, train_end)]
# temp_test = temperature[np.arange(test_start, test_end)]

#Placeholder
X = tf.placeholder(dtype=tf.float32 , shape=[100990,1])
Y = tf.placeholder(dtype=tf.float32 , shape = [None] )#output actual


#Mode architecture parameters:
#Three layer model
n_temp = 1
n_neurons_1 = 200
n_neurons_2 = 200
n_neurons_3 = 200
n_target = 1


# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Layer 1: Variables for hidden weights and biases
W1 = tf.Variable(weight_initializer([n_temp, n_neurons_1]))#1*200
b1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
b2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3: Variables for hidden weights and biases
W3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
b3 = tf.Variable(bias_initializer([n_neurons_3]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_3, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W2), b2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W3), b3))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

#cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))
#Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
net = tf.Session()


# Run initializer
net.run(tf.global_variables_initializer())


# Fitting the neural network
# Number of epochs and batch size
epochs = 1
batch_size = 100990

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(temp_train)))
    print(shuffle_indices)#shuffle the indices
    temp_train = temp_train[shuffle_indices]
    print(temp_train)#corrosponding indices value get also shuffled
    # Minibatch training
    for i in range(0, len(temp_train)//batch_size):
        print("I am smart")
        start = i * batch_size#
        batch_x = temp_train[start:start + batch_size]
        # Run optimizer with batch
        p = net.run(opt, feed_dict={X: batch_x})
        print(p)

        ''' # Show progress
        if np.mod(i, 4) == 0:'''
        # Prediction
        pred = net.run(out, feed_dict={X: temp_test})
        print(pred)
        line2.set_ydata(pred)
        plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
        file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'       #making image file
        plt.savefig(file_name)
        plt.pause(0.01)
        plt.show()
'''
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: temp_test, Y: y_test})
print(mse_final)
'''