import numpy as np
import pandas as pd
from pandas import Series
from pandas import read_csv
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import warnings


warnings.filterwarnings("ignore")
data = pd.read_csv("delhi-weather-data/testset.csv")
#drop datetime variable
data = data.drop(['datetime'], 1)

temperature = data[' _tempm']
temperature.replace(0, np.NaN)

temperature.fillna(temperature.mean(), inplace = True)

#dimension of datasets
n = temperature.shape[0]
#m = temperature.shape[1]
print(n)
#print(m)

#Making data a numpy array
temperature = temperature.values
#visualising dataset finding if missing value is there or not
'''plt.plot(temperature)
plt.show()
'''
#Preparing training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
temp_train = temperature[np.arange(train_start, train_end)]
temp_test = temperature[np.arange(test_start, test_end)]


print(temp_train)
temp_train1 = np.reshape(temp_train, (1, -1))
print(temp_train)




'''size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
train = pd.Series(train)
test = pd.Series(test)
history = [x for x in train]
'''

"""#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(temp_train)
temp_train = scaler.transform(temp_train)
temp_test = scaler.transform(temp_test)
"""
#Building X and  y
X_train = temp_train[:]
y_train = temp_train[:]
X_test = temp_test[:]
y_test = temp_test[:]


#Placeholder
X = tf.placeholder(dtype=tf.float32 , shape=[None , 1])
Y = tf.placeholder(dtype=tf.float32 , shape = [None] )

n_classes = 10
batch_size = 100

#Mode architecture parameters:
#Three layer model
n_temp = 1
n_neuron_1 = 3
n_neuron_2 = 3
n_neuron_3 = 2
n_target = 1

tf.set_random_seed(1)                   # so that your "random" numbers match ours
    
### START CODE HERE ### (approx. 6 lines of code)
W1 = tf.get_variable("W1", [3,1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [3,1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [3,3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b2 = tf.get_variable("b2", [3,1], initializer = tf.zeros_initializer())
W3 = tf.get_variable("W3", [2,3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b3 = tf.get_variable("b3", [2,1], initializer = tf.zeros_initializer())

W_out = tf.get_variable("W_out", [1,3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b_out = tf.Variable("b_out", [1,1])
### END CODE HERE ###

    

'''
# Layer 1: Variables for hidden weights and biases
W1 = tf.Variable(weight_initializer([n_temp, n_neurons_1]))
b1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
b2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
b3 = tf.Variable(bias_initializer([n_neurons_3]))
'''

