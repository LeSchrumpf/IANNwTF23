import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

from MNIST_MLP import data_prep, training, visualization

#DECREASED LEARNING RATE

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train','test'], as_supervised = True , with_info=True)

batchsize = 32
train_ds = train_ds.apply(data_prep)
test_ds = test_ds.apply(data_prep)

#Hyper Parameters
class NN(tf.keras.Model):
    def __init__(self):
        super(NN, self).__init__()
        #After a bit of research, I have found that the relu activation
        #generally provides the best accuracies for this task.
        #I also want three hidden layers, although this isn't based on any particular reason
        self.hidden1 = tf.keras.layers.Dense(256, tf.nn.relu)
        self.hidden2 = tf.keras.layers.Dense(128, tf.nn.relu)
        #And of course the softmax to turn the output into easily read probabilities
        self.out = tf.keras.layers.Dense(10, tf.nn.softmax)
    def call(self, inputs):
        output = self.hidden1(inputs)
        output = self.hidden2(output)
        output = self.out(output)
        return output

#Going through every single input to the function one by one
how_many_epochs = 10

used_model = NN()

#training set is train_ds/minibatches of train_ds
#test set is test_ds/minibatches of test_ds

cce = tf.keras.losses.CategoricalCrossentropy()
learning_rate = 0.1
sgd = tf.keras.optimizers.SGD(learning_rate, 0)

tr_loss = []
tr_acc = []
te_loss = []
te_acc = []

tr_loss, tr_acc, te_loss, te_acc = training(how_many_epochs, used_model, train_ds, test_ds, cce, sgd, tr_loss, tr_acc, te_loss, te_acc)

print("Finished training")

visualization(tr_loss, tr_acc, te_loss, te_acc)