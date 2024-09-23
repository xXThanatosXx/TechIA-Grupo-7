import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
# Convertir el conjunto de datos en un dataframe de Pandas
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=0)
# Definir los valores de los hiperparámetros a probar
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
# Crear una instancia de SVC
svc = SVC()

# Crear una instancia de GridSearchCV para buscar los mejores hiperparámetros
grid = GridSearchCV(svc, param_grid, verbose=2)

# Entrenar el modelo
grid.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid.best_params_)

# Evaluate the model on the test set
accuracy = grid.score(X_test, y_test)
print("Test set accuracy: ", accuracy)

# Print the mean test score for each combination of hyperparameters
means = grid.cv_results_['mean_test_score']
for mean, params in zip(means, grid.cv_results_['params']):
    print("%0.3f for %r" % (mean, params))
