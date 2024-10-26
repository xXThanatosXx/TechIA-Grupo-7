import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar el conjunto de datos CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar los valores de píxeles a un rango de 0 a 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Configuración del data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,      # Rotar las imágenes en un rango de 0 a 20 grados
    width_shift_range=0.2,  # Desplazamiento horizontal de la imagen
    height_shift_range=0.2, # Desplazamiento vertical de la imagen
    shear_range=0.15,       # Transformación de corte
    zoom_range=0.2,         # Aplicar zoom a la imagen
    horizontal_flip=True,   # Voltear la imagen horizontalmente
    fill_mode="nearest"     # Rellenar los píxeles vacíos tras las transformaciones
)

# Crear un modelo simple de red neuronal convolucional
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo usando el generador de data augmentation
batch_size = 64
steps_per_epoch = x_train.shape[0] // batch_size

model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.2f}')
