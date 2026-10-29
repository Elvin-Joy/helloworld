#8
import numpy as np
import time
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

X_train=X_train[:10000]
X_test=X_test[:20000]
y_train=y_train[:10000]
y_test=y_test[:20000]

X_train=X_train.reshape(len(X_train),784).astype("float32")/255.0
X_test=X_test.reshape(len(X_test),784).astype("float32")/255.0

k_values=[1,3,5,7]

for k in k_values:
  model=KNeighborsClassifier(n_neighbors=k)

  start_time=time.time()
  model.fit(X_train,y_train)
  train_time=time.time()-start_time

  start_time=time.time()
  y_pred=model.predict(X_test)
  prediction_time=time.time()-start_time

  accuracy = accuracy_score(y_test, y_pred)

  print("K =", k)
  print("Accuracy:", round(accuracy, 4))
  print("Training time:", round(train_time, 2), "seconds")
  print("Prediction time:", round(prediction_time, 2), "seconds")
  print()

