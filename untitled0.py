import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("mustarddata.xlsx")
df
df_bkp=df

df=pd.get_dummies(df, dummy_na=True)
df.describe()
df.info()
 



