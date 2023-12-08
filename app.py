# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from model.Network import Network
from model.FCLayer import FCLayer
from model.ActivationLayer import ActivationLayer
from helpers.ActivationFunctions import tanh, tanh_prime
from helpers.LossFunctions import mse, mse_prime
from helpers.TextProcessing import clean_text


tweets_df = pd.read_csv("tweet_emotions.csv") 
tweets_df.drop('tweet_id', axis=1, inplace=True)
tweets_df["text"] = tweets_df["content"].apply(lambda x: clean_text(x))
tweets_df["text"] = tweets_df["text"].apply(lambda x: ' '.join(x))

x = np.array(tweets_df['text'])
y = np.array(tweets_df['sentiment'])

le = LabelEncoder()
Y = le.fit_transform(y)
vec = TfidfVectorizer()
X = vec.fit_transform(x)

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=42)

# Network
net = Network()
net.add(FCLayer(1, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(X_train[0:1000], Y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])