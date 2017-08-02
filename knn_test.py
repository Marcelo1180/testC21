import numpy as np
from sklearn import neighbors, datasets

n_neighbors = 15

iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(x, y)
#https://www.youtube.com/watch?v=f_oVM4JiCMs

print(y)
