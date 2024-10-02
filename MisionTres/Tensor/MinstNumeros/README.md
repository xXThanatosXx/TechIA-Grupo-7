# Clasificación de Dígitos Escritos a Mano con Keras y TensorFlow

Este proyecto demuestra cómo construir y entrenar una red neuronal convolucional para clasificar imágenes de dígitos escritos a mano del conjunto de datos MNIST. Utilizaremos **Keras** y **TensorFlow** para implementar diferentes tipos de capas en la red neuronal.

## Tabla de Contenidos

- [Introducción](#introducción)
- [Descripción de las Capas Utilizadas](#descripción-de-las-capas-utilizadas)
  - [Capa Convolucional (`Conv2D`)](#capa-convolucional-conv2d)
  - [Capa de Pooling (`MaxPooling2D`)](#capa-de-pooling-maxpooling2d)
  - [Capa de Dropout (`Dropout`)](#capa-de-dropout-dropout)
  - [Capa de Aplanamiento (`Flatten`)](#capa-de-aplanamiento-flatten)
  - [Capa Densa (`Dense`)](#capa-densa-dense)
  - [Capa de Salida](#capa-de-salida)
- [Requisitos Previos](#requisitos-previos)
- [Cómo Ejecutar el Código](#cómo-ejecutar-el-código)
- [Explicación del Código](#explicación-del-código)
- [Resultados Esperados](#resultados-esperados)
- [Referencias](#referencias)

## Introducción

El objetivo es construir una red neuronal que pueda reconocer dígitos escritos a mano utilizando el conjunto de datos MNIST. Este conjunto de datos es un estándar en el campo del aprendizaje automático y contiene 70,000 imágenes en escala de grises de dígitos del 0 al 9.

## Descripción de las Capas Utilizadas

A continuación se describen las capas utilizadas en la red neuronal y su función:

### Capa Convolucional (`Conv2D`)

Las capas convolucionales aplican filtros (kernels) para extraer características locales de las imágenes, como bordes, texturas y formas, empleadas en el reconocimiento de patrones en imágenes.

- **Sintaxis**:

  ```python
  layers.Conv2D(filtros, tamaño_kernel, activación, input_shape)
  ```

- filtros: Número de mapas de características que la capa aprenderá.
- tamaño_kernel: Tamaño del filtro aplicado (e.g., (3, 3)).
- activación: Función de activación, comúnmente 'relu'.

### Capa de Pooling (MaxPooling2D)
Las capas de pooling reducen la dimensionalidad espacial de las características, manteniendo la información más importante. Esto ayuda a reducir el número de parámetros y el sobreajuste.

```python
layers.MaxPooling2D(tamaño_pool)
```

- tamaño_pool: Tamaño de la ventana de pooling (e.g., (2, 2)).

### Capa de Dropout (Dropout)
Las capas de dropout desactivan aleatoriamente una fracción de neuronas durante el entrenamiento, lo que ayuda a prevenir el sobreajuste y mejora la generalización del modelo.

```python
layers.Dropout(tasa)
```

### Capa de Aplanamiento (Flatten)

Esta capa convierte las características 2D en un vector 1D, permitiendo que las características sean procesadas por capas densas.

```python
layers.Flatten()
```

### Capa Densa (Dense)
Las capas densas son capas totalmente conectadas donde cada neurona recibe entradas de todas las neuronas de la capa anterior.

```python
layers.Dense(unidades, activación)
```
- unidades: Número de neuronas en la capa.
- activación: Función de activación (e.g., 'relu').

### Capa de Salida
La capa final utiliza una activación softmax para convertir las salidas en probabilidades para cada clase.

```python
layers.Dense(número_clases, activation='softmax')
```

### Ejemplo


- Capas:
```python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Seleccionar una imagen para mostrar (por ejemplo, la primera imagen)
imagen_index = 0
imagen = x_train[imagen_index]

# Mostrar información sobre la imagen
print(f"Dimensiones de la imagen: {imagen.shape}")
print(f"Etiqueta de la imagen: {y_train[imagen_index]}")
print(f"Valor mínimo del píxel antes de la normalización: {imagen.min()}")
print(f"Valor máximo del píxel antes de la normalización: {imagen.max()}")

# Visualizar la imagen con escala de grises y barra de colores
plt.figure(figsize=(6, 6))
plt.imshow(imagen, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title(f"Dígito: {y_train[imagen_index]}")
plt.show()

# Normalizar la imagen dividiendo entre 255
imagen_normalizada = imagen / 255.0

# Mostrar información sobre la imagen normalizada
print(f"Valor mínimo del píxel después de la normalización: {imagen_normalizada.min()}")
print(f"Valor máximo del píxel después de la normalización: {imagen_normalizada.max()}")

# Visualizar la imagen normalizada con escala de grises y barra de colores
plt.figure(figsize=(6, 6))
plt.imshow(imagen_normalizada, cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title(f"Dígito Normalizado: {y_train[imagen_index]}")
plt.show()

```

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Cargar y preprocesar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos y agregar una dimensión para canales
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Convertir las etiquetas a formato categórico
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Construir el modelo
model = models.Sequential()

# Capa Convolucional inicial
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Capa de Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Segunda Capa Convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Segunda Capa de Pooling
model.add(layers.MaxPooling2D((2, 2)))

# Capa de Dropout
model.add(layers.Dropout(0.25))

# Aplanar las salidas
model.add(layers.Flatten())

# Capa Densa
model.add(layers.Dense(128, activation='relu'))

# Otra Capa de Dropout
model.add(layers.Dropout(0.5))

# Capa de Salida
model.add(layers.Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_acc:.4f}')
```


### Carga y Preprocesamiento de Datos:

- Las imágenes se normalizan dividiendo entre 255 para escalar los valores de los píxeles entre 0 y 1.
- Se agrega una dimensión para los canales, ya que las capas convolucionales esperan entradas con forma (altura, anchura, canales).

1. Construcción del Modelo:

- Capas Convolucionales: Utilizamos filtros de tamaños (3, 3) con activación relu.
- Capas de Pooling: Reducen la dimensionalidad espacial a la mitad.
- Capas de Dropout: Aplicamos una tasa de 0.25 y 0.5 para prevenir el sobreajuste.
- Capa de Aplanamiento: Convierte la salida 2D en 1D.
- Capa Densa: 128 neuronas con activación relu.
- Capa de Salida: 10 neuronas (una por clase) con activación softmax.


2. Compilación y Entrenamiento:

- Se utiliza el optimizador adam y la función de pérdida categorical_crossentropy.
- El modelo se entrena durante 10 épocas con un tamaño de lote de 128.
- Se reserva el 10% de los datos para validación durante el entrenamiento.
Evaluación:

