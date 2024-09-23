import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
import six
from IPython.display import Image  
import pydotplus
# Cargar dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

# Eliminar filas con valores no numéricos en la columna 'pregnant'
pima = pima[pima['pregnant'].apply(lambda x: str(x).isnumeric())]

# Convertir la columna 'pregnant' a tipo numérico
pima['pregnant'] = pima['pregnant'].astype(float)

# Dividir el dataset en características y variable objetivo
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols] # Características
y = pima.label # Variable objetivo

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Crear objeto clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el clasificador de árbol de decisión
clf = clf.fit(X_train, y_train)

# Predecir la respuesta para el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))



dot_data = six.StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())