import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
dataset = pd.read_excel('C:/Users/Asus/OneDrive/Documents/Python Scripts/LASTRIDE.xlsx')
dataset.head()
dataset.dtypes
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 10)
knn_classifier.fit(X_train, y_train)
dataset
y_pred = knn_classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
