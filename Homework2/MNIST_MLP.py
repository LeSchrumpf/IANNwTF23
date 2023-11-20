import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

#Task 2.1 Loading the MNIST dataset


(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train','test'], as_supervised = True , with_info=True)

#To answer questions for task 2.1:

#There are 10000 testing images
#There are 60000 training images

#The image shape is (28, 28, 1)

#The pixel values are in range 0 to 255


#Task 2.2 Setting up the data pipeline

#declaring the batchsize here. It seems to get difficult when placing this as a positional argument
batchsize = 32

def data_prep(mnist):
    #changing data type
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    #flattening
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #normalize
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
    #one hot encoding
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))
    
    #Task 3 Additions
    #Caching helps not blowing up my computer's memory
    mnist = mnist.cache()
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(batchsize)
    mnist = mnist.prefetch(20)

    return mnist

train_ds = train_ds.apply(data_prep)
test_ds = test_ds.apply(data_prep)

#Task 2.3 Building a deep neural network with TensorFlow


class NN(tf.keras.Model):
    def __init__(self):
        super(NN, self).__init__()
        #After a bit of research, I have found that the relu activation
        #generally provides the best accuracies for this task.
        #I also want three hidden layers, although this isn't based on any particular reason
        self.hidden1 = tf.keras.layers.Dense(256, tf.nn.relu)
        self.hidden2 = tf.keras.layers.Dense(256, tf.nn.relu)
        self.hidden3 = tf.keras.layers.Dense(256, tf.nn.relu)
        #And of course the softmax to turn the output into easily read probabilities
        self.out = tf.keras.layers.Dense(10, tf.nn.softmax)
    def call(self, inputs):
        output = self.hidden1(inputs)
        output = self.hidden2(output)
        output = self.hidden3(output)
        output = self.out(output)
        return output

#Task 2.4 Training the network

#This isn't exactly a pure "training function" if I understand correctly. 
#However, due to the desired parameters of this function and the usual way in which neural networks function
#There is a testing step at the end after all.
def training(epochs, model, training_set, test_set, loss_func, optimizer, train_losses, train_accuracies, test_losses, test_accuracies):
    for x in range(0, epochs):
        mini_loss = []
        mini_accs = []
        for input,target in training_set:
            with tf.GradientTape() as tape:
                #create a prediction by feeding the input to the model
                prediction = model(input)
                #figure out how poorly/well the model did by comparing model's output with target
                loss = loss_func(target, prediction)
                #if the highest value in the set of predictions corresponds to the target, add a 1. If not, a 0
                accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
                #the mean gives us a nice singular value to gauge the accuracy from
                accuracy = np.mean(accuracy)

            #applying the optimizer function to all trainable variables
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #loss and accuracy for each individual target/prediction pair
            mini_loss.append(loss)
            mini_accs.append(accuracy)
        #mean of the losses and accuracies in this epoch is appended to the lists
        train_losses.append(tf.reduce_mean(mini_loss))
        train_accuracies.append(tf.reduce_mean(mini_accs))

        mini_test_loss = []
        mini_test_accuracy = []
        for input,target in test_set:
            prediction = model(input)
            #calculate loss using the specified function
            current_loss = loss_func(target, prediction)

            #Same thing as we did above with the training accuracies
            current_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            current_accuracy = np.mean(current_accuracy)

            #add loss and accuracy for this particular value to the temporary list
            mini_test_loss.append(current_loss)
            mini_test_accuracy.append(current_accuracy)
    
        #finally, add these two calculated values to the overall list
        test_losses.append(tf.reduce_mean(mini_test_loss))
        test_accuracies.append(tf.reduce_mean(mini_test_accuracy))

    return train_losses, train_accuracies, test_losses, test_accuracies


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

#Task 2.5 Visualization


def visualization(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure()
    line1, = plt.plot(train_losses, "b-")
    line2, = plt.plot(test_losses, "r-")
    line3, = plt.plot(train_accuracies, "b:")
    line4, = plt.plot(test_accuracies, "r:")
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3, line4,), ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.show()

visualization(tr_loss, tr_acc, te_loss, te_acc)



