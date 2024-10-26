import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Cargar los datos
df = pd.read_csv('D:\Shadow\GitHub\TechIA-Grupo-7\MisionTres\GAN\consumo_agua_racionamiento.csv')

# 2. Preprocesamiento de los datos
# Separar las características (X) de la variable objetivo (y)
X = df.drop(columns=['Racionamiento_necesario', 'Fecha'])  # Eliminamos la columna objetivo y fecha
y = df['Racionamiento_necesario']

# Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de los datos (es importante para que los modelos de redes neuronales funcionen bien)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Añadir una dimensión extra para que funcione con Conv1D
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# 3. Definir el modelo CNN
model = Sequential()

# Capa de convolución 1D
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dropout(0.3))  # Evitar el sobreajuste

# Otra capa de convolución 1D
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(Dropout(0.3))

# Capa de aplanado (Flatten)
model.add(Flatten())

# Capa densa totalmente conectada
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Capa de salida (usamos softmax para clasificación binaria)
model.add(Dense(1, activation='sigmoid'))

# 4. Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 5. Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2)

# 6. Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Precisión en el conjunto de prueba: {test_acc:.2f}')

# 7. Hacer predicciones (opcional)
predicciones = model.predict(X_test_scaled)
predicciones_binarias = (predicciones > 0.5).astype(int)

# 8. Mostrar las predicciones y los valores reales en un diagrama de barras
indices = np.arange(len(y_test))

plt.figure(figsize=(14, 7))

# Diagrama de barras de las predicciones
plt.bar(indices, predicciones_binarias.flatten(), color='blue', width=0.4, label='Predicciones', alpha=0.6)

# Diagrama de barras de los valores reales
plt.bar(indices + 0.4, y_test.values, color='green', width=0.4, label='Valores Reales', alpha=0.6)

plt.xlabel('Índice de ejemplo')
plt.ylabel('Racionamiento (0 = No, 1 = Sí)')
plt.title('Predicciones vs Valores Reales de Racionamiento')
plt.legend()
plt.show()


