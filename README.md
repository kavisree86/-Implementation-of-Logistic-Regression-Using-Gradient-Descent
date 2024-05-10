# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
### Array Value of x

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/b6f62589-50a0-4166-b283-5b74c6eb6bfb)

### Array Value of y

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/497f84ec-507b-4322-b75e-0345dd7cf7de)

### Exam 1 - score graph

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/9533d39a-86cd-400e-aef5-9aa5e551bdeb)


#### Sigmoid function graph


![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/e600dc9e-f640-451c-ac3d-438c40ccd11f)


### X_train_grad value

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/b460ffcd-190f-4185-8cba-d563df2d20de)

### Y_train_grad value

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/8208237b-c128-4dce-913a-026359478e39)


### Print res.x

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/7c4d19d2-264d-4ef1-8f44-12cc335f3119)


### Decision boundary - graph for exam score

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/ed679dd4-02cf-4179-b66b-a58f6cec5c4e)

### Proability value

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/4673776b-a094-4362-b8fb-c791a1d9d07b)

### Prediction value of mean

![image](https://github.com/kavisree86/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145759687/437629ed-662b-406a-99d1-d13e8b5f1b06)





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
*/
```
