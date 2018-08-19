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
#dimension of datasets
n = data.shape[0]#number of rows
print(n)
p = data.shape[1]##number of columns
print(p)
temperature = data[' _tempm']
temperature.replace(0, np.NaN)

temperature.fillna(temperature.mean(), inplace = True)
plt.plot(temperature)
plt.show()
X = temperature.values
print(X)
size = int(len(X)*0.8)

train , test = X[0:size] , X[size:len(X)]
train = pd.Series(train)
test = pd.Series(test)

history = [x for x in train]

prediction = list()
hold = list()

