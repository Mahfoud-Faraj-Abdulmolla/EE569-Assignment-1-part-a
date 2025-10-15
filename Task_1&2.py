# Import libraries ( numpy , matplotlib , scipy   )

import numpy as np
import matplotlib.pyplot as plt
import scipy





#  ***************************************************************************************************

# Task 1: Implementing a Linear Computation Node

class Linear:
    def __init__(self,w,b):

        self.w = w
        self.b = b
        self.Input =None
    def forword(self,Input):
        self.Input = Input
        return Input @ self.w.T + self.b
    def Backpopagation(self,output_gradient):


        d_input =  output_gradient @ self.w
        d_w =     output_gradient.T @ self.Input
        d_b=      np.sum(output_gradient,axis=0)

        return d_input,d_w,d_b

#   *******************************************************************************************************

#   Task 2: Integrating the Linear Node into the Logistic Regression Code

# ******** Dataset ******* Note: In the second task, no specific dataset was requested, so random data was used for testing model.

Input = np.random.randn(3,2)
y_correct = np.array([[1],[0],[1]])




# sigmoid function ***
def sigmoid(z):
    return 1/(1+np.exp(-z))


# Binary cross entroopy loss ***

def loss(y_prediction,y_correct):
    epsilon = 1e-8
    return -np.mean(y_correct * np.log(y_prediction + epsilon) + (1 - y_correct) * np.log(1 - y_prediction + epsilon))


#*** Initialize w and b ***

input_size = Input.shape[1]
output_size = 1
w =np.random.randn(output_size,input_size)*0.02
b= np.zeros(output_size)


# Node for ML

Linear_node = Linear(w,b)

# learning rate ____

learning_rate = 0.1

# Number of epochs ----

epoch = int(input("Number of epochs"))
# ******* Training *******

for i in range(epoch):
    # Forward Pass
    z= Linear_node.forword(Input)
    y_prediction = sigmoid(z)
    # calcolate losses
    los = loss(y_prediction,y_correct)
    print("loss is" ,los)
    # Backpobagation
    dz = y_prediction-y_correct
    dx,dw,db = Linear_node.Backpopagation(dz)
    # Ubdate w and b
    Linear_node.w -= learning_rate * dw
    Linear_node.b -= learning_rate * db


# **********************************************************************************************************









































