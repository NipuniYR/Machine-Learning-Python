#import numpy and matplotlib
#for large scale multi dimentional arrays and matrices
import numpy as np 
#plotting library
import matplotlib.pyplot as plt 

#load the dataset
train_data = np.genfromtxt('dataset.csv',delimiter=',')
#remove the header row
train_data = np.delete(train_data,0,0)

#take SAT score as X vector
X = train_data[:,0]
#take GPA as Y vector
Y = train_data[:,1]

#visualize data using matplotlib
plt.scatter(X,Y)
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.show()

#initialize m & c
m = 0 
c = 0 

L = 0.0001 #The Learning rate
iterations = 1000 #The number of iterations to perform Gradient Descent

n = float(len(X)) #number of elements in X (typecast to float so that when taking the average we won't be having any errors)

#Performing Gradient Descent
for i in range(iterations):
    Y_pred = (m*X) + c #The current predicted value for Y
    D_m = (-2/n)*sum(X*(Y-Y_pred)) #Derivative with respect to m
    D_c = (-2/n)*sum(Y-Y_pred) #Derivative with respect to c
    m = m - L*D_m #Update m
    c = c - L*D_c #Update c
    
#output m and c
print("m = ",m)
print("c = ",c)

#Final prediction
Y_pred = (m*X) + c

#Draw the best fitting line
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(Y_pred),max(Y_pred)],color='red')
plt.show()