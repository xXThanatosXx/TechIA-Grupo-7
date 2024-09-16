from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar el dataset de diabetes
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Dividir el dataset en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normalizar las características en un rango entre 0 y 1 usando MinMaxScaler
scaler = MinMaxScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Entrenar el Perceptrón Multicapa (MLPRegressor) con una red neuronal más compleja
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000, alpha=0.01, learning_rate_init=0.001, random_state=1, solver='adam', activation='relu')
mlp.fit(X_train_std, y_train)

# Realizar predicciones
y_pred = mlp.predict(X_test_std)

# Evaluar el rendimiento del modelo usando el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, marker='o')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()

# Graficar el error por época de entrenamiento
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.show()
