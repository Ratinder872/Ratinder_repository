import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('C:/Users/Asus/OneDrive/Documents/Python Scripts/mustarddata.xlsx')

np.random.shuffle(dataset.values)

X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeRegressor  
classifier= DecisionTreeRegressor(random_state=0)  
classifier.fit(X_train, y_train)  

y_pred= classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2*100)


# sAVE MODEL

#import pickle
#filename = 'dt_crop_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))

#filename = 'dt_crop_model.sav'
#loaded_classifier = pickle.load (open(filename, 'rb'))
#result = loaded_classifier.score(X_test, y_test)
#print (result)

