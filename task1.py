# TASK 1

# LINEAR REGRESSION

#Importing all libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

url="http://bit.ly/w-data"
data = pd.read_csv(url)
data.head(10)

# DATA VISUALISATION 

data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

x=data.iloc[:,:-1].values 
print(x) #displaying array of hours.

y=data.iloc[:,1].values
print(y) #displaying array of scores

## Splitting the data into training and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# Training the algorithm

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

### Plotting the Regression Line

line=lr.coef_*x+lr.intercept_

plt.scatter(x,y) #plotting for test data
plt.plot(x,line)
plt.show()

### Making predictions

print(x_test)
y_pred=lr.predict(x_test)

#Actual vs predicted values
df=pd.DataFrame({'actual':y_test,'predicted':y_pred})
df

#what will be the predicted score if no. of hours is 9.25?

lr.predict([[9.5]])


### Evaluating 

from sklearn import metrics
metrics.mean_absolute_error(y_test,y_pred)
