import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#importing the dataset   
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#feature-scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)


#fitting logiistic regression to our training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state =0)
classifier.fit(x_train,y_train)

#predicting the test results
y_pred = classifier.predict(x_test)

#making the confusion matrix
#here confusion_matrix is not a class,it is a function
#to check for the performance of the model,no of correct and incorrect predictions
from sklearn.metrics import confusion_matrix  
conf_mat = confusion_matrix(y_test,y_pred)


#visualising our final results
from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2 =  
