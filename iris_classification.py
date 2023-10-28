import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
# split data for train and test set
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

# data analysis
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15, 15), marker = 'o', hist_kwds = {'bins': 20}, s = 60, alpha = 0.8, cmap = mglearn.cm3)
plt.show()

# using k-neighbours classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# build model
knn.fit(X_train, y_train)

# creating new example
X_new = np.array([[5, 2.9, 1, 0.2]])

# predicting result of the new example
prediction = knn.predict(X_new)
print('Prediction: {}'.format(prediction))
print('Type: {}'.format(iris_dataset['target_names'][prediction]))

# checking accuracy
print('Results for the dataset: {:.2f}'.format(knn.score(X_test, y_test)))