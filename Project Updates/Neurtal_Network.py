########### Using Tensor Flow to Develop a Neural Network for Image Classification of Numbers in the MNIST Dataset ###############
%matplotlib inline
import tensorflow 
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflor.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.models import Sequential

# The shape of the Training data is (60,000 , 28, 28)
# The shape of the Test Data is (10,000 , 28, 28)
(Training_A,Training_label),(Testing_A,Testing_label) = mnist.load_data()

# One-Hot Encoding: Encoding the classes to create an array so that the class that the number belongs to is set to 1, and rest to 0
# Converting both training and testing labels to one-hot encoding
# The shape of the training labels is: (60,000, 10)  
# The shape of the testing labels is: (10000, 10)
Encoded_Train_Label = to_categorical(Training_label)
Encoded_Test_Label = to_categorical(Testing_label)

# Use numpy to reshape the training and testing data set. Each pixel was a 28 *28 data set
# In this step we convert it into a 1D array of 784 pixels
# So the new training data set shape will be: (60000, 784)
# The new testing data set shape will be : (10000, 784)
Reshaped_A_Train = np.reshape(Training_A,(60000,784))
Reshaped_A_test = np.reshape(Testing_A,(10000,784))

#Standardisation steps for the array
#Standardisation is performed to make the mean=0 and the variance =1	

2nd_AStandr = np.std(Reshaped_A_test)
A_standr = np.std(Reshaped_A_Train)

Data_Mean = np.mean(Reshaped_A_Train)
Second_mean = np.mean(Reshaped_A_test)

A_test_standr = (Reshaped_A_test-Second_mean)/2nd_AStandr
A_train_standr = (Reshaped_A_Train-Data_Mean)/A_standr

						#########Creating a neural network using Package provided functions###############
# Considering each of the pixels in the 784 1D array to be it's features x in the neural network so we will have x1...x784 for qa simple neural network
# Our target variable in this neural network (say, y) will be the weighted sum of these 784 features through an activation function and hidden layers
# An activation function helps the neural network by figuring out non-linear patterns in the data
# In the code below     output layers are used which are weighted sum of the input features passed through the activation function
# From reading up about the correct architecture of a neural network, it was found that although there are some rules of thumb, 
# they don;t always work , and so the only way to find the best fitting neural network for the dataset is by testing many different parameters
# and measuring performance. I carried this experimentation out seperately and the descriptive code will be added to another document on my Git
# Based on my analysis I found the following parameters to be best suited for this neural network

# Using Dense layers which are the usual deeply connected neural network layers. It implements "output = activation(dot(input, kernel) + bias)""
# Using a sequential model
# Using Relu as the activation function which will give the same output as the input if function is positive and 0 otherwise
model = Sequential([Dense(500, activation = 'relu', input_shape = (784,)), Dense(500, activation = 'relu'), Dense(10, activation = 'relu')])
#Compiling the neural network using the optimiser as Stochastic Gradient Descent and the loss factor as categorical crossentropy, this loss
#function is used because there are more than one category in which the images can be miscalssified into
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Finding a model fit with the standardised training data set and the encoded target variable
# I ran 15 epochs, epochs is the number of times the model runs through the data set in order to learn the weights
model.fit( A_train_standr, Encoded_Train_Label, epochs = 15)

Uno_Loss, Test_Accuracy = model.evaluate(A_test_standr, Encoded_Test_Label)
Dos_Loss, Training_Accuracy = model.evaluate(A_train_standr, Encoded_Train_Label)

print('Accuracy of Test data set: ', Test_Accuracy)
print('Accuracy of training data set: ', Training_Accuracy)

