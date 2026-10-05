import numpy as np 
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5,6,7,8])
Y=np.array([45, 50, 54, 60, 65, 70, 74, 80])

X_matrix=np.column_stack((np.ones(len(X)),X))

theta=np.linalg.inv(X_matrix.T @ X_matrix)@X_matrix.T @Y

b0=theta[0]
b1=theta[1]

print("Intercept b0=",b0)
print("Slope b1=",b1)


hours=9
predicted_score= b0 + b1*hours
print("Predicted score for 9 hours:", predicted_score)

Y_pred=b0+b1*X
plt.scatter(X,Y,label="Actual Data")
plt.plot(X,Y_pred,label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()