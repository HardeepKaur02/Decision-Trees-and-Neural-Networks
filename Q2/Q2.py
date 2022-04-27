import csv
import numpy as np
import sys
import math
import time
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
#np.random.seed(3141592)


def relu(z):
    return np.maximum(z, 0.0)

def d_relu(z):
    return np.where(z > 0, 1.0, 0.0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1-x)


def forward_prop(batch_x,theta,num_layers,g,M):
    
    outputs = [None for i in range(num_layers)]
    outputs[0] = batch_x
    for layer in range(1,num_layers-1):
        outputs[layer] = np.concatenate((np.ones((M, 1)),g(np.dot(outputs[layer-1],theta[layer]))),axis = 1)  
    layer = num_layers-1       
    outputs[layer] = np.concatenate((np.ones((M, 1)),sigmoid(np.dot(outputs[layer-1],theta[layer]))),axis = 1)  
    
    return outputs 

def backward_prop(batch_y,outputs, theta,num_layers,d_g,M):
    
    final_output = outputs[-1][:, 1:]
    deltas = [None for i in range(num_layers)]
    deltas[-1] = (batch_y - final_output).T * d_sigmoid(final_output.T) / M 
    for layer in range(num_layers-2,0,-1):
        deltas[layer] = np.dot(theta[layer+1][1:,:],deltas[layer+1]) * d_g(outputs[layer][:, 1:].T)
    return deltas 
        

class NeuralNetwork:

    def __init__(self,n,h,r,g=0):
        
        #### @params ####
        # n: Number of features #
        # h: hidden layer sizes  #
        # r: Number of target classes #
        # g: activation function #
        
        self.num_features = n
        self.num_hidden_layers = len(h)
        self.target_classes = r
        self.layer_sizes = [n] + h + [r]
        
        # Assuming same activation function for each perceptron
        if g == 0:
            self.activation_function = sigmoid
            self.d_activation_function = d_sigmoid
        else:
            self.activation_function = relu 
            self.d_activation_function = d_relu

        # Parameter initialisation (Using He initialisation method) #
        
        self.theta = [[None]] # for input layer
        for layer in range(1,self.num_hidden_layers+2): # for hidden layers and output layer
            # random init
            prev = self.layer_sizes[layer-1]
            curr = self.layer_sizes[layer]
            theta_layer = np.random.uniform(-1,1,(prev+1,curr))
            #theta_layer = np.random.normal(0,1/prev,size=(prev+1,curr))
            theta_layer[0] = np.zeros(curr)  #initialise biases with 0
            # multiply by sqrt(2/prev)
            theta_layer = theta_layer * math.sqrt(2/prev)
            self.theta.append(theta_layer)

    def train(self, x_train, y_train, M, alpha_0=0.1, epsilon=0.00001, adaptive=False,max_epochs=110):

        # adding intercept term
        # [x_train] = m x (num_features+1)
        # [y_train] = m x r
        J_history = []
        epochs = []
        m = x_train.shape[0]
        x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)
        y_train = np.concatenate((np.ones((m, 1)), y_train), axis=1)

        # using Mini Batch SGD #

        prev_err = math.inf
        epoch_num = 1

        outputs = [None for i in range(len(self.layer_sizes))]
        deltas = [None for i in range(len(self.layer_sizes))]        

        while True:

            # new epoch begins #
            alpha = alpha_0
            if adaptive:
                alpha /= math.sqrt(epoch_num)
            
            # random shuffling before SGD
            p = np.random.permutation(m)
            x_train, y_train = x_train[p], y_train[p]

            avg_err = 0
            num_batches = m//M

            for i in range(num_batches):
                batch_x = x_train[i*M : (i+1)*M] # M x (num_features+1)
                batch_y = y_train[i*M : (i+1)*M][:,1:] # M x r

                # #forward propagation
                outputs = forward_prop(batch_x,self.theta,self.num_hidden_layers+2,self.activation_function,M)
                # dimensions of [outputs] = M x (curr+1) for each layer
                final_output = outputs[-1][:, 1:]
                avg_err += np.sum((batch_y - final_output)**2)/(2*M)

                # #backward propagation
                deltas = backward_prop(batch_y,outputs, self.theta,self.num_hidden_layers+2,self.d_activation_function,M)
                # dimensions of [deltas] = curr x M for each layer

                # update thetas
                for layer in range(1,len(self.layer_sizes)):
                    self.theta[layer] += alpha * (deltas[layer] @ outputs[layer - 1]).T
            avg_err /= num_batches
            epochs.append(epoch_num)
            J_history.append(avg_err)
            #check convergence
            if abs(avg_err-prev_err) < epsilon:
                plt.plot(epochs,J_history)
                plt.title("Cost function vs number of epochs")
                plt.xlabel("No. of epochs")
                plt.ylabel("Cost Function")
                plt.savefig("J_vs_epochs.png")
                plt.show()
                return epoch_num,avg_err
            prev_err = avg_err
            epoch_num+=1
        return epoch_num,prev_err
    
    def predict(self,x_test):

        m_test = x_test.shape[0]
        layer_output = np.concatenate((np.ones((m_test, 1)), x_test), axis=1) 
        # feedforwarding        
        predictions = forward_prop(layer_output,self.theta,self.num_hidden_layers+2,self.activation_function,m_test)
        final_output = predictions[-1][:, 1:]      
        # returning predictions as class labels (not one-hot encoding)
        return np.argmax(final_output,axis = 1)

def hotEncode(c_attr,x,r):   # 1,10,3,..
    encoded = [0 for i in range(r)]
    if x:
        for j in range(len(c_attr)):
            num = j//2
            rank = j%2
            attr = c_attr[j]  # 10
            pos = num*17 + rank*4 + attr-1
            encoded[pos] = 1
    else:
        encoded[c_attr] = 1
    return encoded

def HotEncode(file_name):
    in_file = open(file_name)
    csvReader = csv.reader(in_file)
    x,y,temp = [], [], []
    for row in csvReader:
        row = list(map(int,row))
        new_row = hotEncode(row[:10],1,85)
        x.append(new_row)
        new_y = hotEncode(row[-1],0,10)
        y.append(new_y)
        temp.append(row[-1])
    return x,y,temp

def Confusion_Matrix(y_pred,y,r=10):
    cm = [[0 for i in range(10)] for i in range(10)]
    for i in range(len(y)):
        cm[y[i]][y_pred[i]] += 1
    return cm


def main():
    part = sys.argv[1]
    if part == "a" or part == "b":
        input_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        x_train, y_train, temp = HotEncode(input_file_name)
        m = len(x_train)
        x_train = np.array(x_train).reshape(m,-1)
        y_train = np.array(y_train).reshape(m,10)
        # print(temp.count(0))
        # print(temp.count(1))
        # print(len(temp))

        x_test,y_test,temp = HotEncode(test_file_name)
        m_test = len(x_test)
        x_test = np.array(x_test).reshape(m_test,-1)
        y_test = np.array(y_test).reshape(m_test,10)
        start = time.time()
        nn = NeuralNetwork(85,[100,100],10,g=0)
        epoch, avg_err= nn.train(x_train,y_train,100)
        #print(nn.theta)
        print("Time taken: " + str(time.time() - start))
        print("No. of epochs: " + str(epoch))
        # print(avg_err)
        y_pred_train = nn.predict(x_train)
        y_pred_test = nn.predict(x_test)

        y_train = np.argmax(y_train,axis = 1)
        y_test = np.argmax(y_test,axis = 1)
        acc_train, acc_test = 0,0
        for i in range(len(y_train)):
            if y_train[i] == y_pred_train[i]:
                acc_train+=1
        for i in range(len(y_test)):
            if y_test[i] == y_pred_test[i]:
                acc_test+=1
        acc_train /= len(y_train)
        acc_test /= len(y_test)
        print("Accuracy on Training Set: " + str(100*acc_train))
        print("Accuracy on Test Data Set: " + str(100*acc_test))
    
    if part == "c":

        # extracting training data
        input_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        x_train, y_train, temp = HotEncode(input_file_name)
        m = len(x_train)
        x_train = np.array(x_train).reshape(m,-1)
        y_train = np.array(y_train).reshape(m,10)
        # extracting test data
        x_test,y_test,temp = HotEncode(test_file_name)
        m_test = len(x_test)
        x_test = np.array(x_test).reshape(m_test,-1)
        y_test = np.array(y_test).reshape(m_test,10)

        test_accuracies = []
        train_accuracies = []
        times = []

        units = [5,10,15,20,25]
        # iterating over all hidden layer unit values
        for hidden_layer_units in units:
            layers = []
            layers.append(hidden_layer_units)
            nn = NeuralNetwork(85,layers,10,g=0)
            t = time.time()
            epoch, average_error = nn.train(x_train,y_train,100)
            y_pred_test = nn.predict(x_test)
            y_pred_train = nn.predict(x_train)
            times.append(time.time() - t)        
            y_train0 = np.argmax(y_train,axis = 1)
            y_test0 = np.argmax(y_test,axis = 1)
            conf_mat = Confusion_Matrix(y_pred_test,y_test0)
            print(conf_mat)

            acc_train, acc_test = 0,0
            for i in range(len(y_train0)):
                if y_train0[i] == y_pred_train[i]:
                    acc_train+=1
            for i in range(len(y_test0)):
                if y_test0[i] == y_pred_test[i]:
                    acc_test+=1
            acc_train /= len(y_train0)
            acc_test /= len(y_test0)

            test_accuracies.append(acc_test*100)
            train_accuracies.append(acc_train*100)

            print('hidden layer units:', hidden_layer_units)
            print('test accuracy:', acc_test, '%')
            print('train accuracy:', acc_train, '%')
            print('time taken:', times[-1])
            print('number of epochs:', epoch)
            print('average error:', average_error)

        plt.title('Accuracy plot')
        plt.xlabel('Hidden layer units')
        plt.ylabel('Accuracy (in %)')
        plt.ylim(45,55)
        plt.xticks(experimental_values)
        plt.plot(units, test_accuracies, label='Test accuracies')
        plt.plot(units, train_accuracies, label='Train accuracies')
        plt.legend()
        plt.savefig('NN_accuracy_plot.png')
        plt.show()
        plt.close()
        plt.title('Time taken')
        plt.xlabel('Hidden layer units')
        plt.ylabel('Time taken (in s)')
        plt.xticks(experimental_values)    
        plt.plot(units, times,label='Time')
        plt.legend()
        plt.savefig('NN_time_plot.png')
        plt.show()
        plt.close()

    if part == "d":
            
        input_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        x_train, y_train, temp = HotEncode(input_file_name)
        m = len(x_train)
        x_train = np.array(x_train).reshape(m,-1)
        y_train = np.array(y_train).reshape(m,10)
        # extracting test data
        x_test,y_test,temp = HotEncode(test_file_name)
        m_test = len(x_test)
        x_test = np.array(x_test).reshape(m_test,-1)
        y_test = np.array(y_test).reshape(m_test,10)

        test_accuracies = []
        train_accuracies = []
        times = []


        units = [5,10,15,20,25]
        # iterating over all hidden layer unit values
        for hidden_layer_units in units:
            layers = []
            layers.append(hidden_layer_units)
            nn = NeuralNetwork(85,layers,10,g=0)
            t = time.time()
            epoch, average_error = nn.train(x_train,y_train,100,adaptive = True)
            y_pred_test = nn.predict(x_test)
            y_pred_train = nn.predict(x_train)
            times.append(time.time() - t)        
            y_train0 = np.argmax(y_train,axis = 1)
            y_test0 = np.argmax(y_test,axis = 1)
            conf_mat = Confusion_Matrix(y_pred_test,y_test0)
            print(conf_mat)

            acc_train, acc_test = 0,0
            for i in range(len(y_train0)):
                if y_train0[i] == y_pred_train[i]:
                    acc_train+=1
            for i in range(len(y_test0)):
                if y_test0[i] == y_pred_test[i]:
                    acc_test+=1
            acc_train /= len(y_train0)
            acc_test /= len(y_test0)

            test_accuracies.append(acc_test*100)
            train_accuracies.append(acc_train*100)

            print('hidden layer units:', hidden_layer_units)
            print('test accuracy:', acc_test, '%')
            print('train accuracy:', acc_train, '%')
            print('time taken:', times[-1])
            print('number of epochs:', epoch)
            print('average error:', average_error)

        plt.title('Accuracy plot')
        plt.xlabel('Hidden layer units')
        plt.ylabel('Accuracy (in %)')
        plt.ylim(45,55)
        plt.plot(units, test_accuracies, label='Test accuracies')
        plt.plot(units, train_accuracies, label='Train accuracies')
        plt.savefig('nn_accuracy_plot_adaptive.png')
        plt.show()
        plt.close()
        plt.title('Time taken')
        plt.xlabel('Hidden layer units')
        plt.ylabel('Time taken (in s)')
        plt.plot(units, times)
        plt.savefig('nn_time_plot_adaptive.png')
        plt.show()
        plt.close()
    
    if part == "e":
        input_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        x_train, y_train, temp = HotEncode(input_file_name)
        m = len(x_train)
        x_train = np.array(x_train).reshape(m,-1)
        y_train = np.array(y_train).reshape(m,10)
        # print(temp.count(0))
        # print(temp.count(1))
        # print(len(temp))

        x_test,y_test,temp = HotEncode(test_file_name)
        m_test = len(x_test)
        x_test = np.array(x_test).reshape(m_test,-1)
        y_test = np.array(y_test).reshape(m_test,10)

        start = time.time()
        nn = NeuralNetwork(85,[100,100],10,g=0) # using sigmoid function
        epoch, avg_err= nn.train(x_train,y_train,100,adaptive = True)
        #print(nn.theta)
        print("Time taken using sigmoid: " + str(time.time() - start))
        print("No. of epochs using sigmoid: " + str(epoch))
        # print(avg_err)
        y_pred_train = nn.predict(x_train)
        y_pred_test = nn.predict(x_test)

        y_train0 = np.argmax(y_train,axis = 1)
        y_test0 = np.argmax(y_test,axis = 1)

        conf_mat = Confusion_Matrix(y_pred_test,y_test0)
        print(conf_mat)

        acc_train, acc_test = 0,0
        for i in range(len(y_train0)):
            if y_train0[i] == y_pred_train[i]:
                acc_train+=1
        for i in range(len(y_test0)):
            if y_test0[i] == y_pred_test[i]:
                acc_test+=1
        acc_train /= len(y_train0)
        acc_test /= len(y_test0)
        print("Accuracy on Training Set using sigmoid: " + str(100*acc_train))
        print("Accuracy on Test Data Set using sigmoid: " + str(100*acc_test))

        #### Model using ReLU ####
        start = time.time()
        nn_2 = NeuralNetwork(85,[100,100],10,g=1) # using ReLU function
        epoch, avg_err= nn_2.train(x_train,y_train,100,adaptive = True)
        
        print("Time taken using ReLU: " + str(time.time() - start))
        print("No. of epochs using ReLU: " + str(epoch))
        
        y_pred_train = nn_2.predict(x_train)
        y_pred_test = nn_2.predict(x_test)
        conf_mat = Confusion_Matrix(y_pred_test,y_test0)
        print(conf_mat)

        acc_train, acc_test = 0,0
        for i in range(len(y_train0)):
            if y_train0[i] == y_pred_train[i]:
                acc_train+=1
        for i in range(len(y_test0)):
            if y_test0[i] == y_pred_test[i]:
                acc_test+=1
        acc_train /= len(y_train0)
        acc_test /= len(y_test0)
        print("Accuracy on Training Set using ReLU: " + str(100*acc_train))
        print("Accuracy on Test Data Set using ReLU: " + str(100*acc_test))

    if part == "f":


        input_file_name = sys.argv[1]
        test_file_name = sys.argv[2]
        x_train, y_train, temp = HotEncode(input_file_name)
        m = len(x_train)
        x_train = np.array(x_train).reshape(m,-1)
        y_train = np.array(y_train).reshape(m,10)
        # extracting test data
        x_test,y_test,temp = HotEncode(test_file_name)
        m_test = len(x_test)
        x_test = np.array(x_test).reshape(m_test,-1)
        y_test = np.array(y_test).reshape(m_test,10)

        nn = MLPClassifier(activation='logistic',hidden_layer_sizes=(100,100),solver='sgd',batch_size=100,learning_rate='adaptive',learning_rate_init=0.1,max_iter=1000,)
        start = time.time()
        nn.fit(x_train, y_train)

        # prediction on training and test data
        y_pred_test = nn.predict(x_test)
        y_pred_train = nn.predict(x_train)
        y_pred_test = np.argmax(y_pred_test,axis = 1)
        y_pred_train = np.argmax(y_pred_train,axis = 1)

        y_train0 = np.argmax(y_train,axis = 1)
        y_test0 = np.argmax(y_test,axis = 1)

        acc_train, acc_test = 0,0
        for i in range(len(y_train0)):
            if y_train0[i] == y_pred_train[i]:
                acc_train+=1
        for i in range(len(y_test0)):
            if y_test0[i] == y_pred_test[i]:
                acc_test+=1
        acc_train /= len(y_train0)
        acc_test /= len(y_test0)

        print("Accuracy on Training Set using sigmoid: " + str(100*acc_train))
        print("Accuracy on Test Data Set using sigmoid: " + str(100*acc_test))
        print('Time taken:', time.time() - start)

        nn = MLPClassifier(activation='relu',hidden_layer_sizes=(100,100),solver='sgd',batch_size=100,learning_rate = 'adaptive', learning_rate_init=0.1,max_iter=1000,)
        start = time.time()
        nn.fit(x_train, y_train)

        # prediction on training and test data
        y_pred_test = nn.predict(x_test)
        y_pred_train = nn.predict(x_train)
        y_pred_test = np.argmax(y_pred_test,axis = 1)
        y_pred_train = np.argmax(y_pred_train,axis = 1)

        acc_train, acc_test = 0,0
        for i in range(len(y_train0)):
            if y_train0[i] == y_pred_train[i]:
                acc_train+=1
        for i in range(len(y_test0)):
            if y_test0[i] == y_pred_test[i]:
                acc_test+=1
        acc_train /= len(y_train0)
        acc_test /= len(y_test0)

        print("Accuracy on Training Set using ReLU: " + str(100*acc_train))
        print("Accuracy on Test Data Set using ReLU: " + str(100*acc_test))
        print('Time taken:', time.time() - start)


if __name__ == '__main__' :
    main()