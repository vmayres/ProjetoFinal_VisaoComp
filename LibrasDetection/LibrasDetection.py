#Bibliotecas
import os
import cv2
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

# Caminho do diretório onde o dataset foi salvo
saved_dataset_path = r"LibrasDetection\processed_dataset"


# Carrega o dataset
loaded_dataset = tf.data.Dataset.load(saved_dataset_path)

# Itera sobre o dataset e imprime os dados
for data, label in loaded_dataset.take(5):  # Ajuste o número de exemplos a ser exibido
    print("Dados (frames do video):", data.shape)
    print("Label:", label.numpy())

# num_classes = 0

# data = []
# labels = []

# # Dividir os dados em conjuntos de treino e validação
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# # Codificar as labels para one-hot
# X_train_hot = to_categorical(X_train, num_classes=num_classes)
# y_train_hot = to_categorical(y_train, num_classes=num_classes)

# # Construção do modelo Conv3D
# model = Sequential([
#     Input(shape=(num_frames, altura, largura, canais)),
#     Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
#     MaxPooling3D(pool_size=(2, 2, 2)),
#     Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
#     MaxPooling3D(pool_size=(2, 2, 2)),
#     Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
#     MaxPooling3D(pool_size=(2, 2, 2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # Compilação do modelo
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Exibir o resumo do modelo
# model.summary()

# # Treinamento do modelo
# model.fit(X_train_hot, y_train_hot, batch_size=32, epochs=10, validation_split=0.2)

# predictions = model.predict(X_test)
# print(predictions)

# # Avaliação do modelo
# score = model.evaluate(X_test, y_test, batch_size=32)

# print("Test loss:", score[0])

# print("Test accuracy:", score[1])

# predictions = model.predict(X_test)

