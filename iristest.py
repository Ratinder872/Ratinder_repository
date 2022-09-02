import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\Asus\\OneDrive\\Documents\\Python Scripts\\RIS.csv")
dataset.shape
dataset.head(5)
dataset.describe()
dataset.groupby('Species').size()
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from  sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
from pandas.plotting import parallel_coordinates
 

parallel_coordinates(dataset.drop("Id", axis=1), "Species")


dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(15, 10))
 
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

 
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
 
cm = confusion_matrix(y_test, y_pred)
cm
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 mber of samples
        if self.n_neighbors > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        # finding and saving all possible class labels
        self.classes_ = np.unique(y)
        
        self.X = X
        self.y = y
        
    #def predict(self, X_test):
        
        # number of predictions to make and number of features inside single sample
     #   n_predictions, n_features = X_test.shape
        
        # allocationg space for array of predictions
      #  predictions = np.empty(n_predictions, dtype=int)
        
        # loop over all observations
       # for i in range(n_predictions):
            # calculation of single prediction
        #   predictions[i] = single_prediction(self.X, self.y, X_test[i, :], self.n_neighbors)

        #return(predictions)
    def single_prediction(X, y, x_train, k):
    # number of samples inside training set
         n_samples = X.shape[0]
    
    # create array for distances and targets
         distances = np.empty(n_samples, dtype=np.float64)

    # distance calculation
         for i in range(n_samples):
            distances[i] = (x_train - X[i]).dot(x_train - X[i])
    
    # combining arrays as columns
         distances = sp.c_[distances, y]
    # sorting array by value of first column
         sorted_distances = distances[distances[:,0].argsort()]
    # celecting labels associeted with k smallest distances
         targets = sorted_distances[0:k,1]

         unique, counts = np.unique(targets, return_counts=True)
         return(unique[np.argmax(counts)])
    def predict(self, X_test):
     
     # number of predictions to make and number of features inside single sample
      n_predictions, n_features = X_test.shape
     
     # allocationg space for array of predictions
      predictions = np.empty(n_predictions, dtype=int)
     
     # loop over all observations
      for i in range(n_predictions):
         # calculation of single prediction
        predictions[i] = single_prediction(self.X, self.y, X_test[i, :], self.n_neighbors)

      return(predictions)
# Instantiate learning model (k = 3)
my_classifier = MyKNeighborsClassifier(n_neighbors=3)

# Fitting the model
my_classifier.fit(X_train, y_train)

# Predicting the Test set results
my_y_pred = my_classifier.predict(X_test)
accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
print('Accuracy of our model is equal 96.67')

