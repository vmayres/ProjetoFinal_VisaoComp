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

# Direcionamento do diretorio do Dataset.
dataset_path = r'Dataset'

#Carrega a lista de labels
label_dict = {}
for i, d in enumerate(sorted(os.listdir(dataset_path))):
    label_dict[d] = i

print("Classes das clissificao de LIBRAS", label_dict)
'''
{'acontecer': 0, 'aluno': 1, 'amarelo': 2, 'america': 3,
'aproveitar': 4, 'bala': 5, 'banco': 6, 'banheiro': 7,
'barulho': 8, 'cinco': 9, 'conhcer': 10, 'espelho': 11,
'esquina': 12, 'filho': 13, 'maca': 14, 'medo': 15,
'ruim': 16, 'sapo': 17, 'vacina': 18, 'vontade': 19}
'''

# Parâmetros dos dados
num_frames = 30
altura, largura = 64, 64
canais = 1  # Grayscale
num_classes = len(label_dict)  # Total de classes

#Inicializa as listas de dados e labels
data = []
labels = []

def extract_frames(video_path, num_frames=30):
    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calcular o intervalo entre os frames que queremos pegar
    interval = frame_count // num_frames
    
    frames = []
    for i in range(num_frames):
        # Pular para o frame específico
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            # Redimensiona o frame se necessário (ex., 64x64)
            frame = cv2.resize(frame, (altura, largura))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = img_to_array(frame)
            frames.append(frame)
        else:
            break

    cap.release()
    return frames

print("[INFO] loading video files...")
# Percorre cada pasta de classe
for label_dir in os.listdir(dataset_path):
    # Cria o caminho completo para a pasta da classe
    class_path = os.path.join(dataset_path, label_dir)

    # Verifica se é um diretório (evita problemas com arquivos extras)
    if os.path.isdir(class_path):
        # Carrega cada arquivo de vídeo na pasta da classe
        for video_file in os.listdir(class_path):
            # Caminho completo para o arquivo de vídeo
            video_path = os.path.join(class_path, video_file)

            # Verifica se o arquivo é um vídeo, usando a extensão
            if video_file.endswith(('.mp4', '.avi')):  # ajuste as extensões conforme necessário
                # Extrai os frames do vídeo e adiciona aos dados
                data.append(extract_frames(video_path))

                # Associa a label correta
                label = label_dict[label_dir]
                labels.append(label)

print("[INFO] Processing video frames...")
# Converter listas para arrays numpy e normalizar os dados
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Dividir os dados em conjuntos de treino e validação
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Codificar as labels para one-hot
X_train_hot = to_categorical(X_train, num_classes=num_classes)
y_train_hot = to_categorical(y_train, num_classes=num_classes)

# Construção do modelo Conv3D
model = Sequential([
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(num_frames, altura, largura, canais)),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Exibir o resumo do modelo
model.summary()

# Treinamento do modelo
model.fit(X_train_hot, y_train_hot, batch_size=32, epochs=10, validation_split=0.2)

predictions = model.predict(X_test)
print(predictions)

# Avaliação do modelo
score = model.evaluate(X_test, y_test, batch_size=32)

print("Test loss:", score[0])

print("Test accuracy:", score[1])

predictions = model.predict(X_test)

