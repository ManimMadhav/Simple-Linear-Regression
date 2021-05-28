import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#split test set and traning set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x ,y ,test_size=0.2 ,random_state=0)

#training the model on SLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict test set results
y_pred = regressor.predict(x_test)


#training set results visualization
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#test set results visualization
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

exp=10;
#predict() returns predicted 'y' for an input value by the user
print(regressor.predict([[exp]]))

#coef and intercept return values of m and c respectively,
#where m->slope and c->intercept
print(regressor.coef_)
print(regressor.intercept_)
