# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Wczytanie danych MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizacja danych oraz zmiana kształtu (dodanie kanału)
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test  = x_test.astype('float32') / 255.0
x_test  = np.expand_dims(x_test, axis=-1)

# One-hot encoding etykiet
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Budowa modelu CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu (dla demonstracji 3 epoki – można zwiększyć liczbę epok)
print("Trening modelu...")
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

# Ocena modelu
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Zapis modelu do pliku .h5
model.save("mnist_cnn.h5")
print("Model zapisany jako 'mnist_cnn.h5'")
