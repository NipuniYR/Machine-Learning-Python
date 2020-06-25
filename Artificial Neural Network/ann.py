import numpy as np
#Load data from a text file, with missing values handled as specified, seperated by , 
data = np.genfromtxt('breast-cancer-wisconsin.data',delimiter=',')
#This data set contains some missing values, they have been read as 'nan'

#********************************************Data Processing********************************************

#remove all the values with 'nan'
data = data[~np.isnan(data).any(axis=1)]
# ~ - NOT
#np.isnan(data) - true - if nan   false - if not nan
#any() - true - if atleast one is true
#axis = 1 - column 
#any(axis = 1) - This will pass (all columns of) a row at a time, if atleast one true is there in that row it will return true
#My guess since we used ~ here any() will return false if atleast one is false

#remove the id column
data = np.delete(data,0,axis=1)

#now dataset has no id or nan
#Dataset details
#Column 0 - Clump Thickness: 1 - 10
#Column 1 Uniformity of Cell Size: 1 - 10
#Column 2 Uniformity of Cell Shape: 1 - 10
#Column 3 Marginal Adhesion: 1 - 10
#Column 4 Single Epithelial Cell Size: 1 - 10
#Column 5 Bare Nuclei: 1 - 10
#Column 6 Bland Chromatin: 1 - 10
#Column 7 Normal Nucleoli: 1 - 10
#Column 8 Mitoses: 1 - 10
#Column 9 Class: (2 for benign, 4 for malignant)

#Column 9 has 2 classes (binary category) let's replace 2 with 0 and 4 with 1
data[:,9][data[:,9]==2] = 0
data[:,9][data[:,9]==4] = 1

#shuffle data (shuffle rows) and divide to attributes and labels
np.random.shuffle(data)
attributes, labels = data[:, :9], data[:, 9:]

#normalize the attributes using min max method
x_max, x_min = attributes.max(), attributes.min()
attributes = (attributes - x_min)/(x_max - x_min)

# divide dataset into 70% - training set and 30% - testing set
margin = len(data)//10*7 #// is floor division
training_x, testing_x = attributes[:margin, :], attributes[margin:, :]
training_y, testing_y = labels[:margin, :], labels[margin:, :]

#***************************************End of Data Processing*************************************** 

#Let's consider a Neural network of 3 layers. Input has 9 units (9 attributes), output has 1 unit (i label) and hidden layer has 5 units

class NeuralNetwork:
    def __init__(self):
        #__init__ - constrouctor in the context of OOP
        #self - By using the self keyword, can easily access all the instances defined within a class, including its methods and attributes
        #initialize weight s with random values
        self.weights1 = np.random.rand(5,9)
        self.weights2 = np.random.rand(1,5)
        
        #declare variables for predicted, inputs ansd labels
        self.output = None #Null value
        self.input = None
        self.y = None
        
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def sigmoid_derivative(self,Z):
        #here Z = sigmoid(output) - output is what we get after feedforwarding
        return Z*(1-Z)
    
    def feedforward(self):
        #feedforward to layer 1
        self.layer1 = self.sigmoid(np.matmul(self.weights1,np.transpose(self.input)))
        
        #feedforward to layer 2
        self.layer2 = self.sigmoid(np.matmul(self.weights2,self.layer1))
        
        return np.transpose(self.layer2)
    
    def backpropergation(self,learning_rate):
        #get partial derivative for layer 2 weights
        #Cost function C = (Y - Y_pred)^2 = (Y - a2)^2
        #w2 = w2 - learning_rate*delta2*activation_Level1
        #delta2 = C'(a2)*Sigmoid'(Z2)
        #C'(a2) = 2*(Y - a2)*(-1) = (-2)*(Y - a2)
        #Sigomoid'(Z2) = self.sigmoid_derivative(a2) according to our derivative function implementation we have to pass a2 as the parameter instead of z2 (see self.sigmoid_derivative())
        #delta2 = (-2)*(Y - a2)*sigmoid'(Z2) = (-2)*(self.y - self.output)*self.sigmoid_derivative(self.output)
        d_weights2 = np.transpose(np.matmul(self.layer1,((-2)*(self.y - self.output)*self.sigmoid_derivative(self.output))))
        
        #get partial derivative for layer 2 weights
        #w1 = w1 - learning_rate*delta2*w2*sigmoid'(z1)*input
        d_weights1 = np.transpose(np.matmul(np.transpose(self.input),np.matmul(((-2)*(self.y - self.output)*self.sigmoid_derivative(self.output)),self.weights2)*(np.transpose(self.sigmoid_derivative(self.layer1)))))
        
        #adjust weight
        self.weights1 -= learning_rate*d_weights1
        self.weights2 -= learning_rate*d_weights2
        
    def train(self,X,y,learning_rate):
        #set input and lables
        self.input = X
        self.y = y
        
        #feedforward and set output
        self.output = self.feedforward()
        
        #backpropergate
        self.backpropergation(learning_rate)
        
        #return training error 
        return np.mean(np.square(y-np.round(self.output)))
    
    def test(self,X,y):
        # set input and labels
        self.input = X
        self.y = y
        
        #feedforward and setoutput
        self.output = self.feedforward()
        
        # print test results
        print("\nTesting Results\nError : " + str(np.mean(np.square(y - np.round(self.output))))+"\n")
        
NN = NeuralNetwork()
learning_rate = 0.01

#train this data for 1000 iterations
for i in range(1,1001):
    error = NN.train(training_x,training_y,learning_rate)
    #print error after every 10 iterations
    if i%10==0 or i==1:
        print("Iteration: "+str(i)+" | Error: "+str(error))
        
NN.test(testing_x,testing_y)