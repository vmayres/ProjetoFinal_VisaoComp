import os
import cv2
import numpy as np

import tensorflow as tf

#
dataset_path = r"Dataset"

#
data = []
labels = []

# Dicionário para associar o nome da pasta à classe
label_dict = {label_dir: idx for idx, label_dir in enumerate(os.listdir(dataset_path))}
print(label_dict)


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
            frame = cv2.resize(frame, (64, 64))
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
            if video_file.endswith(('.mp4')):  # ajuste as extensões conforme necessário
                # Extrai os frames do vídeo e adiciona aos dados
                data.append(extract_frames(video_path))

                # Associa a label correta
                label = label_dict[label_dir]
                labels.append(label)

print(len(data))
# Mostrar o shape de data sem mudar o tipo da lista
data_shape = (len(data), len(data[0]), len(data[0][0]), len(data[0][0][0])) if data else (0,)
print("Shape of data:", data_shape)

print("[INFO] Processing video frames...")
# Converte para tensores do TensorFlow
data = tf.ragged.constant(data)  # Use `tf.ragged` para sequências de tamanhos variáveis
labels = tf.constant(labels)

# Cria o dataset com TensorFlow
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.shuffle(buffer_size=tf.size(data).numpy()).batch(32)  # Ajuste o tamanho do batch conforme necessário

# Mostra a estrutura do dataset
for batch_data, batch_labels in dataset.take(1):
    print("Batch de dados:", batch_data.shape)
    print("Batch de labels:", batch_labels)

# Save the dataset
tf.data.Dataset.save(dataset, os.path.join("LibrasDetection", "processed_dataset"))

print("[INFO] Dataset saved successfully.")
