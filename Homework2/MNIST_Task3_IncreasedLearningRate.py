import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

from MNIST_MLP import data_prep, training, visualization, NN

#INCREASED LEARNING RATE AND MOMENTUM

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train','test'], as_supervised = True , with_info=True)

batchsize = 32
train_ds = train_ds.apply(data_prep)
test_ds = test_ds.apply(data_prep)

#Hyper Parameters

#Going through every single input to the function one by one
how_many_epochs = 10

used_model = NN()

#training set is train_ds/minibatches of train_ds
#test set is test_ds/minibatches of test_ds

cce = tf.keras.losses.CategoricalCrossentropy()
learning_rate = 0.5
sgd = tf.keras.optimizers.SGD(learning_rate, 0.3)

tr_loss = []
tr_acc = []
te_loss = []
te_acc = []

tr_loss, tr_acc, te_loss, te_acc = training(how_many_epochs, used_model, train_ds, test_ds, cce, sgd, tr_loss, tr_acc, te_loss, te_acc)

print("finished training")

visualization(tr_loss, tr_acc, te_loss, te_acc)