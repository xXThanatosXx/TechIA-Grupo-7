# Clasificación de Imágenes de Ropa con CNN utilizando TensorFlow y Keras
## Descripción
Este proyecto consiste en construir y entrenar una red neuronal convolucional (CNN) para clasificar imágenes de ropa utilizando el conjunto de datos Fashion-MNIST. El objetivo es desarrollar un modelo que pueda identificar correctamente 10 categorías diferentes de prendas de vestir.

El conjunto de datos Fashion-MNIST es una alternativa moderna al clásico MNIST y contiene imágenes en escala de grises de 28x28 píxeles de artículos de moda.

## Objetivos
- Construir y entrenar un modelo CNN para la clasificación de imágenes.
- Evaluar el modelo y visualizar los resultados.
- Experimentar con la arquitectura y los hiperparámetros del modelo.

1. Importación de Librerías
Se importan las librerías necesarias para el proyecto:

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

```
2. Carga y Preparación del Conjunto de Datos
Se carga el conjunto de datos Fashion-MNIST y se normalizan las imágenes

```python
# Cargar el conjunto de datos
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalizar los píxeles a valores entre 0 y 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Cambiar la forma de las imágenes para que tengan una dimensión de canal
# Alto, ancho, Canal 
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

```
3. Visualización de Imágenes
Se visualizan algunas imágenes del conjunto de entrenamiento para entender mejor los datos:
```python
class_names = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

```
4. Construcción del Modelo CNN
Se define la arquitectura del modelo CNN:
```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

5. Compilación y Entrenamiento del Modelo
El modelo se compila y se entrena:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

```

6. Evaluación del Modelo
Se evalúa el rendimiento del modelo en el conjunto de prueba:
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nPrecisión en los datos de prueba:', test_acc)
```

7. Visualización de Resultados
Se grafican las curvas de precisión y pérdida:

```python

# Precisión
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()
```


8. Predicciones y Visualización de Resultados
Se hacen predicciones y se visualizan algunas de ellas:

```python

predictions = model.predict(test_images)
```
```python

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape(28,28), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% (Verdadero: {})".format(class_names[predicted_label],
                                      100*np.max(predictions_array),
                                      class_names[true_label]),
                                      color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

```

```python

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

```


9. Guardar el Modelo (Opcional)
Puedes guardar el modelo entrenado para usarlo posteriormente:
```python
model.save('fashion_mnist_cnn_model.h5')

```

