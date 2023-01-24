# Author: Konstantinos Kontogeorgopoulos

# We import the pandas and numpy libraries
import pandas as pd
import numpy as np

# read .csv data file and put it into concrete_data pandas dataframe object
concrete_data = pd.read_csv('https://cocl.us/concrete_data')
# lets see the first 5 rows of the csv
concrete_data.head()

# I check how many data points i have
concrete_data.shape
# Let's check if the data needs cleaning.
concrete_data.describe()
concrete_data.isnull().sum()

# Now that i know my data is clean i proceed to split the data
# into predictors and target according to the assignment's instructions

# Normalize the Data 
# I normalize the data by subtracting the mean from the individual predictors and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()
# I print the top 5
predictors_norm.head()

# Lets import tensorflow 
import tensorflow as tf 
#And create the model
# Regression model
def regression_model():
    #I create the model
    model = tf.keras.Sequential()
    # In Dense method i put in the hidden layer 10 neurons with the RelU activation function
    model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(8,)))
    model.add(tf.keras.layers.Dense(1))
    
    # and add the adam optimizer and the mean squared error as the loss function 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#import train_test_split from sklearn
from sklearn.model_selection import train_test_split
#and split the data into a training and test sets with 70-30 % training - test
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=7)

# I call the method to create the model and put it into model var
model = regression_model()
# and i use 50 epochs to train it
model.fit(X_train, y_train, epochs=50, verbose=1)

# I evaluate the model 
loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val

# import mean_squared_error from sklearn.metrics
from sklearn.metrics import mean_squared_error
# compute the mean squared error between the predicted concrete stregth y_pred and the actual concrete strength y_test 
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)

# Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors. 
# Report the mean and the standard deviation of the mean squared errors.
all_mean_squared_errors = 50
#creating empty array
mean_squared_errors = []
#for 50 times i do the same as i did before
for i in range(0, all_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " + str(all_mean_squared_errors) + " mean squared errors with normalized data. \n")
print("Mean: "+ str(mean))
print("Standard Deviation: "+ str(standard_deviation))