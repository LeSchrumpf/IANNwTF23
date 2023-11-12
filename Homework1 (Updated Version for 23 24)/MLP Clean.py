import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#Task 2.1 DATA PREPROCESSING
dataset = load_digits()

target = dataset.target
input = np.reshape(dataset.data, (1797, 64))


#rescale
input = (input - np.min(input)) / (np.max(input) - np.min(input))

#onehot encoding
def onehot_encoder(tar):
    onehot_target = np.zeros((1797, 10), float)
    for x in range (0, np.shape(tar)[0]):
        for y in range (0, 10):
            if tar[x] == y:
                onehot_target[x, y] = 1
    return onehot_target

target = onehot_encoder(target)

def shuffler(input, target, batchsize):
    #shuffling the indices
    new_index = np.random.permutation(np.shape(input)[0])
    input = input[new_index]
    target = target[new_index]

    #minibatching
    mini_input = []
    mini_target = []
    for x in range(0, np.shape(input)[0]):
        mini_input.append(input[x])
        mini_target.append(target[x])
        if len(mini_input) >= batchsize:
            yield mini_input, mini_target
            mini_input = []
            mini_target = []
    if mini_input:
        #done in a tuple, which makes getting the individual outputs a bit awkward, but it works
        yield mini_input, mini_target


#2.2 SIGMOID ACTIVATION FUNCTION
class Sigmoid:
    def __init__(self):
        print("Sigmoid here")
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    def __call__(self, input):
        activation = []
        batchsize = np.shape(input)[0]
        vector_length = np.shape(input)[1]
        for x in range (0, batchsize):
            mini_activation = []
            for y in range (0, vector_length):
                mini_activation.append(self.sigmoid(1 + np.exp(-input[x][y])))
            activation.append(mini_activation)
        return activation
    
    #I don't understand why the error_signal is applied here
    #We are calculating all partial derivatives for the various functionalities we have
    #(loss / activation functions  / weights / input to the weights)
    #but couldn't we just calculate each step and multiply them at the end?
    #On a simpler level, I do not understand in which way to use the "preactivation" as an input here
    #Isn't the effect of the preactivations what we want to calculate? For that, I did not find any
    #formulas that needed a preactivation as any sort of input
    def backwards(self, activation, error_signal):
        preactivation = []
        #for every element in the batch
        for x in range(0, np.shape(activation)[0]):
            mini_preact = []
            #take the sigmoid derivative
            for y in range(0, np.shape(activation)[1]):
                mini_preact.append(self.sigmoid_der(activation[x][y]) * error_signal[x][0])
            preactivation.append(mini_preact)
        return preactivation






#sigmoid test   
#activation = Sigmoid()
#minibatch_1 = activation(minibatch_1)
#print(minibatch_1)

#2.3 SOFTMAX ACTIVATION FUNCTION
class Softmax:
    def __init__(self):
        print("Softmax here")
    def __call__(self, input):
        activation = []
        batchsize = np.shape(input)[0]
        vector_length = np.shape(input)[1]
        for x in range (0, batchsize):
            e_x = np.exp(x - np.max(x))
            mini_activation = []
            for y in range(0, vector_length):
                #calculation is supposedly more numerically stable
                mini_activation.append(np.exp(input[x][y] - np.max(input[x])) / np.exp((input)[x] - np.max(input[x])).sum())
            activation.append(mini_activation)
        return activation
    
    #I'm not sure if this is needed. I have implemented it but do not use it in the backprop
    def backwards(self, prediction):
        #quick and slow version of the jacobian
        return(np.diag(prediction) - np.outer(prediction, prediction))
            #jacobian = np.diag(prediction)
            #for x in range(len(jacobian)):
             #   for y in range(len(jacobian)):
              #      if x == y:
               #         jacobian[x][y] = prediction[x] * (1 - prediction[x])
                #    else:
                 #       jacobian[x][y] = -prediction[x] * prediction[y]
            #return jacobian
        

#test
#activation = Softmax()
#minibatch_1 = activation(minibatch_1)
#print(minibatch_1)
#print(sum(minibatch_1[0])) # Should equate to 1. Equates to 1 decently enough but not perfectly

#2.4 MLP WEIGHTS

class MLP_Layer:
    #necessary attributes
    weight_matrix = []
    bias = []

    def __init__(self, input_size, percept_size):
        #setting all of the bias to 0
        self.bias = [0] * percept_size
        #adding random weights
        self.weight_matrix = np.zeros((input_size, percept_size))
        for x in range(0, input_size):
            for y in range(0, percept_size):
                self.weight_matrix[x, y] = np.random.normal(0, 0.2)

        self.softmax = Softmax()
        self.sigmoid = Sigmoid()
    
    #calculation forwards
    def forward(self, input, activation_func):
        #running the input values through the weights and biases
        preactivation = np.matmul(input, self.weight_matrix)
        preactivation = np.add(preactivation, self.bias)

        #lastly apply the desired activation function
        if activation_func == 'softmax':
            activation = self.softmax
        elif activation_func == 'sigmoid':
            activation = self.sigmoid
        else:
            #In case I mispell 
            print("Specify activation function in lower case, please")
        output = activation(preactivation)
        return output
    #calculation backwards
    #def backward(self, activation_from_prev, error_signal):
        #np.dot(activation_from_prev.T, self.weight_matrix)

#2.5 PUTTING TOGETHER THE MLP

class MLP:
    #the list containing all hidden layers
    hidden_layer_list = []
    #list containing losses
    loss_list = []


    def __init__(self, input_size, hidden_sizes, output_size):
        self.sizes = hidden_sizes
        self.layer_list = []
        #for every hidden layer...
        for x in range(0, len(hidden_sizes)):
            #take the size of the previous layer as the input_size of our MLP_Layer
            if x == 0:
                #(except for the input layer, which essentially creates a layer of size input_size with inputs of size input_size)
                previous_size = input_size
            else:
                previous_size = hidden_sizes[x - 1]
            #and create a new layer using the current item in the list (desired size of the layer) as the percept_size
            self.layer_list.append(MLP_Layer(previous_size, hidden_sizes[x]))
        #lastly, we want to define the output layer to access it more easily during the forward function later
        self.output_layer = MLP_Layer(hidden_sizes[-1], output_size)

    def forward(self, input):
        #as long as there still are hidden layers...
        for x in range(0, len(self.sizes)):
            #calculate the layer's output by running the input through it with sigmoid activation functions
            input = self.layer_list[x].forward(input, 'sigmoid')
        #at the end the output layer is waiting. The final output is calculated using the softmax to turn it into probabilities
        input = self.output_layer.forward(input, 'softmax')
        return input


#2.6 CCE LOSS FUNCTION
class CCE:
    def __init__(self):
        print("Loss function here")
    #formula
    def cce_loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat))
    #this is actually the combination of the CCE and Softmax derivative, but it leads us to our goal either way (as far as I understood)
    def derivative(self, y, y_hat):
        return y_hat - y
    def forward(self, prediction, target):
        loss_list = []
        #for every batch
        for x in range(0, np.shape(prediction)[0]):
            loss = []
            #for every node in the output layer
            for y in range(0, np.shape(prediction)[1]):
                #add the loss using the formula
                loss.append(self.cce_loss(target[x][y], prediction[x][y]))
            #lastly, we only need the sum of that loss, since there would be a lot of zeros otherwise
            loss_list.append(np.sum(loss))
        return loss_list
    
    #essentially, this subtracts 1 from the value at the index of the correct class
    def backward(self, prediction, target):
        loss_der_list = []
        for x in range(0, np.shape(prediction)[0]):
            loss_der = []
            for y in range(0, np.shape(prediction)[1]):
                #without this "if" it wouldn't be the right shape 
                # but I also do not understand why it has to be of shape (minibatch, 1)
                #and not shape (minibatch, 10)
                if(target[x][y] == 1):
                    loss_der.append(self.derivative(target[x][y], prediction[x][y]))
            loss_der_list.append(loss_der)
        return loss_der_list


#creating a minibatch for testing
minibatch_gen = shuffler(input, target, 3)
minibatch_1 = next(minibatch_gen)
minitargets_1 = minibatch_1[1]
minibatch_1 = minibatch_1[0]

#testing forward MLP
MLP_whole = MLP(64, [32], 10)
final_output = MLP_whole.forward(minibatch_1)
loss_func = CCE()
print(loss_func.forward(final_output, minitargets_1))

#backward MLP
loss = loss_func.backward(final_output, minitargets_1)
sg = Sigmoid()
preacts_of_output_layer = sg.backwards(final_output, loss)
print(np.shape(preacts_of_output_layer))



