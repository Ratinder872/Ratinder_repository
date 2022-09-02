# use of pickle library
import seaborn as sns
df=sns.load_dataset('iris')
df.head()



import pickle
filename='file.pkl'
#serialize process wb is used bcz if use rb we can write only and cannot write
pickle.dump(df,open(filename,'wb'))

#unserialized
#pickle.dump(df,open(filename,'rb'))






dict_example()={'first_name' : 'RP','last_name':'SINGH'}

pickle.dump(df,open(pickle,'wb'))






