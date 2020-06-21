import pandas as pd
import numpy as np
import statistics as stat

#df - data frame
df = pd.read_csv('insurance.csv') #will read the dataset into a dataframe
#Here we have to deal with non numerical data
#We have to make all the non numerical data into numerical data

#overview of the dataset
#column 0 - age 
#column 1 - sex (categorical - binary)
#column 2 - bmi 
#column 3 - children 
#column 4 - smoker (categorical - binary)
#column 5 - region (categorical)
#column 6 - charges

#charges will be our Y value. Let's seperate it from the dataset and get it as an array
Y = df['charges'].to_numpy()
#Let's remove charges column from the dataset
df = df.drop(['charges'],axis=1) #axis=1 (column)  axis=0 (row)

#******************************Data Preprocessing**************************************
#we can change the binary data into 0 & 1
df['sex'] = df['sex'].map({'female':0, 'male':1})
df['smoker'] = df['smoker'].map({'no':0, 'yes':1})

#One hot encoding for categorical data (not binary)
dummies = pd.get_dummies(df['region'])

df = pd.concat([df,dummies],axis=1) #data frame concatenation

#now that we have inserted dummies to region, let's delete region column
df = df.drop(['region'],axis=1)
df = df.drop(['southwest'],axis=1)

#let's convert df into an array
data = df.to_numpy()
#now training dataset is ready (all in numerical form)
#************************End of Data Preprocessing**************************************

#Gradient Descent method

#Feature scaling using mean normalization method
#We need to scale age(column 0) and bmi(column 2) of data
X = df.to_numpy()
mean_age = stat.mean(X[:,0])
sd_age = stat.stdev(X[:,0])
mean_bmi = stat.mean(X[:,2])
sd_bmi = stat.stdev(X[:,2])

X[:,0] = (X[:,0]-mean_age)/sd_age
X[:,2] = (X[:,2]-mean_bmi)/sd_bmi

L = 0.0001 #Learning rate
iterations = 2000 #number of iterations to perform the Gradient Descent

n = float(np.shape(X)[0]) #number of training examples
f = np.shape(X)[1] #number of features


#In this scenario as we have many features, we have many ms and a c
#Let's initialize them
c = 0 
m = np.zeros(f)

D_c = 0
D_m = np.zeros(f)

for i in range(iterations):
    Y_Pred = np.matmul(X,m) + c
    D_c = (-2/n)*sum(Y-Y_Pred)
    for j in range(f):
        D_m[j] = (-2/n)*sum(np.multiply((Y-Y_Pred),X[:,j])) 
        
    c = c - L*D_c
    for j in range(f):
        m[j] = m[j] - L*D_m[j]
    
#output
print("Gradient Descent")
print("c = ",c)
print("m = ")
for j in range(f):
    print(m[j])
    
#let's predict predict for 132th row in the data set (to check)
input_data = X[132,:]
predictedY = np.matmul(np.transpose(m),input_data) + c
print("Predicted Y = ",predictedY) 
#with 4 categories in onehot encoding actual = 11163.568  predicted = 11429.524
#with 3 categories in onehot encoding actual = 11163.568  predicted = 10814.423

#normal equation method
#let's add x0 = 1 to the matrix
X_norm = np.concatenate((np.ones((int(n),1)),data),axis=1) #array concatenation
m_norm = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_norm),X_norm)),np.transpose(X_norm)),Y)

print("\nNormal Equation")
print("c = ",m_norm[0])
print("m = ")
print(m_norm[1:])
input_data_norm = np.append(1,data[132,:])
predictedY_norm = np.matmul(np.transpose(m_norm),input_data_norm)
print("Prdeicted Y normal equation = ",predictedY_norm) 
#with 4 categories in onehot encoding actual = 11163.568  predicted = 30537.914
#with 3 categories in onehot encoding actual = 11163.568  predicted = 13842.843