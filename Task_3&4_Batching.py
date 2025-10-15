# Import libraries ( numpy , matplotlib , scipy   )

import numpy as np
import matplotlib.pyplot as plt
import scipy





#   ***************************************************************************************************

# Task 3: : Introducing Batching

class Linear:
    def __init__(self,w,b):

        self.w = w
        self.b = b
        self.Input =None
    def forword(self,Input):
        self.Input = Input
        return Input @ self.w.T + self.b
    def Backpopagation(self,output_gradient):


        batch_size =self.Input.shape[0]


        d_input =  output_gradient @ self.w
        d_w =     output_gradient.T @ self.Input / batch_size
        d_b=      np.mean(output_gradient,axis=0)

        return d_input,d_w,d_b

#   *******************************************************************************************************



# ******** Dataset ******* Note: In the second task, no specific dataset was requested, so random data was used for testing model.

Input = np.random.randn(500,2)
y_correct = np.random.randint(0,2,(500,1))




# sigmoid function ***
def sigmoid(z):
    return 1/(1+np.exp(-z))


# Binary cross entroopy loss ***

def loss(y_prediction,y_correct):
    epsilon = 1e-8
    return -np.mean(y_correct * np.log(y_prediction + epsilon) + (1 - y_correct) * np.log(1 - y_prediction + epsilon))


# Initialize w and b ***

input_size = Input.shape[1]
output_size = 1
w =np.random.randn(output_size,input_size)*0.02
b= np.zeros(output_size)


# Node for ML

Linear_node = Linear(w,b)

# learning rate & batch size ____

learning_rate = 0.01
batch_size = 1

# ******* Training *******

def training_model(batch_size,epoch=100):

    input_size = Input.shape[1]
    output_size =1

    w = np.random.randn(output_size, input_size) * 0.02
    b = np.zeros(output_size)
    Linear_node= Linear(w,b)
    losses= []
    for i in range(epoch):

         epock_loss =0
         for j in range(0, len(Input), batch_size):
            Input_batch = Input[j:j + batch_size]
            y_batch = y_correct[j:j + batch_size]


            # Forward Pass
            z= Linear_node.forword(Input_batch)
            y_prediction = sigmoid(z)
            # calcolate losses
            los = loss(y_prediction,y_batch)
            epock_loss+= los

            # Backpobagation
            dz = y_prediction-y_batch
            dx,dw,db = Linear_node.Backpopagation(dz)
            # Ubdate w and b
            Linear_node.w -= learning_rate * dw
            Linear_node.b -= learning_rate * db
         losses.append(epock_loss/(len(Input)/batch_size))

    return losses
# **************Task 4*****************


# ********* compare different Batch size *********

batch_size= [1,2,10,50,250,500]

# Number of epochs ----

epoch = 500

plt.figure(figsize=(10,6))

for batch in batch_size :
    losses = training_model(batch,epoch)
    plt.plot(range(1, len(losses) + 1), losses,label=f"Batch size = {batch}")

# ********Plot********
plt.xlabel("Epochs")
plt.ylabel("Training Losses")
plt.title("Effect of Batch Size on Training Losses")
plt.legend()
plt.grid(True)
plt.show()













