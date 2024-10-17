# Configuraciones de Support Vector Regression (SVR) en `scikit-learn`

Support Vector Regression (SVR) es un algoritmo de aprendizaje automático poderoso utilizado para predecir valores continuos. En `scikit-learn`, el modelo SVR puede ajustarse mediante la selección adecuada de varios hiperparámetros. Este archivo README describe los principales hiperparámetros disponibles para SVR en `scikit-learn` y sus efectos.

## Hiperparámetros

### 1. `kernel` (Tipo de Kernel)
- **Descripción**: Especifica el tipo de kernel que se usará en el algoritmo.
- **Opciones**:
  - `'linear'`: Utiliza un kernel lineal. Adecuado para problemas con una relación aproximadamente lineal.
  - `'poly'`: Utiliza un kernel polinómico, que puede capturar relaciones no lineales.
  - `'rbf'` (Radial Basis Function): El kernel gaussiano es el más utilizado y es efectivo para datos no lineales.
  - `'sigmoid'`: Similar a la función de activación de una red neuronal.
- **Valor por defecto**: `'rbf'`

### 2. `C` (Parámetro de Regularización)
- **Descripción**: Parámetro de regularización que controla el equilibrio entre alcanzar un error bajo en los datos de entrenamiento y minimizar la complejidad del modelo.
- **Efecto**:
  - Valores altos de `C`: Intentan ajustar el modelo lo mejor posible a los datos de entrenamiento, lo que puede llevar al sobreajuste.
  - Valores bajos de `C`: Aumentan la regularización, permitiendo un margen de tolerancia más amplio.
- **Valores típicos**: `[0.1, 1, 10, 100]`
- **Valor por defecto**: `1.0`

### 3. `gamma`
- **Descripción**: Define el alcance de la influencia de una única muestra de entrenamiento, donde los valores bajos significan "lejos" y los valores altos significan "cerca".
- **Opciones**:
  - `'scale'`: Usa `1 / (n_features * X.var())` como `gamma`.
  - `'auto'`: Usa `1 / n_features` como `gamma`.
  - También se puede especificar un valor flotante, por ejemplo, `0.01`, `0.1`, `1`.
- **Efecto**:
  - Valores altos de `gamma`: Hacen que el modelo se ajuste más a los datos de entrenamiento (puede conducir al sobreajuste).
  - Valores bajos de `gamma`: Hacen que el modelo sea más suave, generalizando mejor.
- **Valor por defecto**: `'scale'`

### 4. `epsilon` (Margen de Insensibilidad al Error)
- **Descripción**: Especifica el margen dentro del cual no se penalizan las predicciones en la función de pérdida si los puntos están a una distancia `epsilon` del valor real.
- **Efecto**:
  - Valores altos de `epsilon`: Permiten mayor tolerancia a errores en la predicción.
  - Valores bajos de `epsilon`: Hacen que el modelo sea más sensible a pequeños errores.
- **Valores típicos**: `[0.1, 0.2, 0.5]`
- **Valor por defecto**: `0.1`

### 5. `degree` (Grado del Polinomio, Solo para el Kernel `poly`)
- **Descripción**: Grado de la función kernel polinómica. Ignorado por otros tipos de kernel.
- **Efecto**: Un grado más alto permite al modelo ajustar patrones más complejos, pero puede llevar al sobreajuste.
- **Valores típicos**: `[2, 3, 4]`
- **Valor por defecto**: `3`

### 6. `coef0`
- **Descripción**: Término independiente en las funciones kernel (`'poly'` y `'sigmoid'`).
- **Efecto**: Ajusta la influencia de los términos de orden superior frente a los de orden inferior en el kernel polinómico.
- **Valor por defecto**: `0.0`

## Ejemplo de Ajuste de Hiperparámetros en SVR Usando `GridSearchCV`

A continuación se muestra un ejemplo de cómo utilizar `GridSearchCV` para buscar los mejores hiperparámetros:

```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.datasets as dts 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Cargar el dataset Iris
iris = dts.load_iris()
X = iris.data
y = iris.target

# Crear un pipeline para escalar los datos y entrenar el modelo SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'svr__kernel': ['linear', 'rbf', 'poly'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svr__epsilon': [0.1, 0.2, 0.5],
    'svr__degree': [2, 3, 4]  # Solo se usa cuando el kernel es 'poly'
}

# Configurar la búsqueda con GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Entrenar la búsqueda de hiperparámetros con los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros y el rendimiento obtenido
print("Mejores hiperparámetros encontrados:", grid_search.best_params_)
print("Mejor puntuación (MSE negativo):", grid_search.best_score_)

# Usar el mejor modelo encontrado para predicciones
best_model = grid_search.best_estimator_
```