import pandas as pd
import numpy as np
import math
import operator
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df = pd.read_excel("MentalHealth (2).xlsx")
df
df = df.drop("Timestamp", axis=1)
df
df_bkp=df

df=pd.get_dummies(df, dummy_na=True)
df.describe()
df.info()
df
np.random.seed(5)
msk = np.random.rand(len(df)) < 0.7  #An array containing True(with probability 0.7) and False
train = df[msk]  #Rows having array value true
test = df[~msk]  #Rows having array value False
print('Number of observations in the training data: ', len(train))
print('Number of observations in the test data: ', len(test))
train.head()
test.head()

