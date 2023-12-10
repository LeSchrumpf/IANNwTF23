import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

(train_ds, test_ds), ds_info = tfds.load('cifar10', split=['train','test'], as_supervised = True , with_info=True)

#(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train','test'], as_supervised = True , with_info=True)


#The CIFAR10 consists of:
#ID = A unique number for every single image/label set
#Image = The image data consisting of 32x32 pixel images, all of which have 3 color channels ranging from 0 to 255
#Label = What the image represents. There exist 10 different classes of image, represented by a number associated with a noun like "ship" or "dog", etc.

batchsize = 32

def data_prep(data):
    #changing data type
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    #flattening
    #data = data.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #normalize
    data = data.map(lambda img, target: ((img/128.)-1., target))
    #one hot encoding. Since this is still a classification task
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))
    
    #Task 3 Additions
    #Caching helps not blowing up my computer's memory
    data = data.cache()
    data = data.batch(batchsize)
    data = data.prefetch(20)

    return data

train_ds = train_ds.apply(data_prep)
test_ds = test_ds.apply(data_prep)

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        #First Block
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')
        self.pooling1 = tf.keras.layers.MaxPooling2D()

        #Second Block
        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')
        self.pooling2 = tf.keras.layers.MaxPooling2D()

        self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.conv6 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        #Final Pool
        self.globalpool = tf.keras.layers.GlobalAvgPool2D()

        #softmaxing it since it's still a classification task
        self.out = tf.keras.layers.Dense(10, activation='softmax')
    
    def __call__(self, input):
        #First Block
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.pooling1(input)


        #Second Block
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.pooling2(input)

        input = self.conv5(input)
        input = self.conv6(input)

        input = self.globalpool(input)

        #Final output
        input = self.out(input)
        return input
    

#This training loop is taken directly from my MNIST homework, since the basic idea is still the same.
#The model used may be different, but that only changes the definition of the variable "model" and what applying
#the call function of model does to the data in detail.
#For a commented version of this function, see my MNIST homework :)
def training(epochs, model, training_set, test_set, loss_func, optimizer, train_losses, train_accuracies, test_losses, test_accuracies):
    for x in range(0, epochs):


        mini_loss = []
        mini_accs = []
        for input, target in training_set:
            with tf.GradientTape() as tape:
                prediction = model(input)
                loss = loss_func(target, prediction)
                accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
                accuracy = np.mean(accuracy)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            mini_loss.append(loss)
            mini_accs.append(accuracy)
        train_losses.append(tf.reduce_mean(mini_loss))
        train_accuracies.append(tf.reduce_mean(mini_accs))


        mini_test_loss = []
        mini_test_accuracy = []
        for input,target in test_set:
            prediction = model(input)
            current_loss = loss_func(target, prediction)

            current_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            current_accuracy = np.mean(current_accuracy)

            mini_test_loss.append(current_loss)
            mini_test_accuracy.append(current_accuracy)
        test_losses.append(tf.reduce_mean(mini_test_loss))
        test_accuracies.append(tf.reduce_mean(mini_test_accuracy))

    return train_losses, train_accuracies, test_losses, test_accuracies

#Visualization

def visualization(train_losses, train_accuracies, test_losses, test_accuracies, optimizer, learning_rate):
    plt.figure()
    line1, = plt.plot(train_losses, "b-")
    line2, = plt.plot(test_losses, "r-")
    line3, = plt.plot(train_accuracies, "b:")
    line4, = plt.plot(test_accuracies, "r:")
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3, line4,), ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.title("CNN using optimizer: " + str(optimizer._name) + " with learning rate: " + str(learning_rate))
    plt.show()

#HYPERPARAMETERS!

used_model = CNN()
how_many_epochs = 15

#training set is train_ds/minibatches of train_ds
#test set is test_ds/minibatches of test_ds

#using the categorical cross entropy because it is still a classification task, so that loss function works best
cce = tf.keras.losses.CategoricalCrossentropy()

#Optimizer parameters
rate = 0.05
optim = tf.keras.optimizers.SGD(learning_rate=rate)

tr_loss = []
tr_acc = []
te_loss = []
te_acc = []

tr_loss, tr_acc, te_loss, te_acc = training(how_many_epochs, used_model, train_ds, test_ds, cce, optim, tr_loss, tr_acc, te_loss, te_acc)

visualization(tr_loss, tr_acc, te_loss, te_acc, optim, rate)
