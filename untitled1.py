import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_excel("C:/Users/Asus/OneDrive/Documents/Python Scripts/demo.xlsx")
dataset
np.random.shuffle(dataset.values)
#split the data set into independent (X) and dependent (Y) data sets
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#split the data qet into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#scale the data (feature scaling)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train = sc.fit_transform(X_test)

#Using Logistic Regression Algorithm to the Training Set

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)