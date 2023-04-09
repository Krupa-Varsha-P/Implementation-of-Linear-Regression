# Implementation of Linear Regression Using Gradient Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```txt
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.
```
## Program:
```txt
Program to implement the linear regression using gradient descent.
Developed by: Krupa Varsha P
RegisterNumber:  212220220022
```
```python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("ex1.txt")
print("Profit Prediction Graph:")
plt.scatter(data['a'],data['b'])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")
```
```python3
def computeCost(X,y,theta):
  m=len(y) 
  h=X.dot(theta) 
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err) 
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m, 1)),data_n[:,0].reshape(m, 1),axis=1)
y=data_n[:,1].reshape (m,1) 
theta=np.zeros((2,1))
computeCost(X,y,theta)
```
```python3
def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta -= descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
```
```python3
print("Cost function using Gradient Descent:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(/Theta)$")
plt.title("Cost function using Gradient Descent")
```
```python3
print("Profit Prediction:")
plt.scatter(data['a'],data['b'])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("profit ($10,000)")  
plt.title("Profit Prediction")
```
```python3
def predict(X,theta):
  predictions=np.dot(theta.transpose(),X)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the Population 35,000:")
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))
```
```python3
predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the Population 70,000:")
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))
```
## Output:
linear regression using gradient descent

# Profit Prediction Graph
![ded31073-7b20-49c4-b240-ab516ee2bc0e](https://user-images.githubusercontent.com/100466625/230781919-e06e5e21-8117-4edc-bc3a-7ee2c62e217c.jpg)

# Compute Cost Value 
![0414d9d9-758f-4f4f-8be2-2add6b881de6](https://user-images.githubusercontent.com/100466625/230781940-0e13b4ea-6a17-4e55-82c4-d3054df1abce.jpg)

# h(x) value
![c3f3dbe5-0fb9-4932-8834-e00450e9ccde](https://user-images.githubusercontent.com/100466625/230781953-cf3a4ea8-fe1f-4b0f-a785-bb82d6abb082.jpg)

# Cost Function Using Gradient Descent Graph
![f9b6b5da-6f47-4dc7-8610-5e96553e78bc](https://user-images.githubusercontent.com/100466625/230781965-dbbd15ab-1d69-4fe8-a5b3-008448c032dd.jpg)

# Profit Prediction Graph
![4402dba4-b32a-4ba9-bc3d-b99aca86477c](https://user-images.githubusercontent.com/100466625/230781980-b2a1f162-1831-47d6-a79f-716528f601de.jpg)

# Profit for the Population 35,000
![21b280c8-0947-4afd-9b71-dd7998d7b978](https://user-images.githubusercontent.com/100466625/230781999-76176c5f-45d1-40b1-9bf0-3b3056df1a21.jpg)

# Profit for the Population 70,000
![45b37580-7771-4ba1-9748-d914f534adbe](https://user-images.githubusercontent.com/100466625/230782011-d6a7851a-98f8-4861-8841-e88f89a259d4.jpg)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
