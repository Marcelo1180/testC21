import numpy
from sklearn import tree

atributos = [[140, 1], [130, 1], [150, 0], [170, 0]]
etiqueta = [0, 0, 1, 1]

clasificador = tree.DecisionTreeClassifier()
clasificador = clasificador.fit(atributos, etiqueta)

print(clasificador.predict([[140, 0]]))
