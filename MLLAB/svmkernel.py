#12
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


(X_train_full,y_train_full),(X_test_full,y_test_full)=(fashion_mnist.load_data())

X_train_full=X_train_full.reshape(-1,784)/255.0
X_test_full=X_test_full.reshape(-1,784)/255.0

X_train,_,y_train,_=train_test_split(X_train_full,y_train_full,train_size=3000,random_state=42,stratify=y_train_full)
X_test,_,y_test,_=train_test_split(X_test_full,y_test_full,train_size=1000,random_state=42,stratify=y_test_full)

models={
    "Linear":SVC(kernel="linear"),
    "Polynomial":SVC(kernel="poly",degree=3),
    "RBF":SVC(kernel="rbf")
}

results=[]
for name, model in models.items():
  print("TRAINING",name,"SVM...")

  model.fit(X_train,y_train)
  predictions=model.predict(X_test)
  results.append({ 
    "Kernel":name,
    "Accuracy":accuracy_score(y_test, predictions),
    "Precision":precision_score(y_test, predictions, average="weighted",zero_division=0),
    "Recall":recall_score(y_test, predictions, average="weighted",zero_division=0),
    "F1":f1_score(y_test, predictions, average="weighted",zero_division=0)
    })

results_df=pd.DataFrame(results).set_index("Kernel")
print("\nSVM Kernel Comparison:")
print(results_df.round(4))
results_df["Accuracy"].plot(
    kind="bar",ylim=(0,1),
    title="SVM Kernal Accuracy Comparison",
    ylabel="Accuracy"
)


plt.show()

            
