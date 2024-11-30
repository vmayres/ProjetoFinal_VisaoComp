import os
import io
import sys
import numpy as np
import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from collections import deque
import time

# Set default encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Inicializar o MediaPipe Hands e Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carregar o modelo
model_path = r"D:\ProjetoFinal_VisaoComp\model_mask10classes.keras"
model = load_model(model_path)

# Parâmetros
target_size = (64, 64)  # Dimensão dos frames
num_frames = 60  # Número de frames consecutivos
class_labels = ['Acontecer', 'Amarelo', 'America', 'Aproveitar', 'Banheiro', 
                'Barulho', 'Cinco', 'Espelho', 'Esquina', 'Medo']

# Inicializar a fila para armazenar as últimas previsões
recent_predictions = deque(maxlen=10)  # Janela deslizante

# Função para processar os frames e fazer a predição
def process_frames(frames, model):
    # Redimensionar os frames para o tamanho esperado pelo modelo
    resized_frames = [cv.resize(frame, target_size) for frame in frames]
    # Normalizar os valores dos pixels
    input_frames = np.array(resized_frames) / 255.0
    # Ajustar a forma da entrada para (número de frames, altura, largura, canais)
    input_frames = input_frames.reshape((num_frames, *target_size, 3))
    # Expandir a dimensão para corresponder à entrada do modelo
    input_frames = np.expand_dims(input_frames, axis=0)
    
    # Fazer a predição
    prediction = model.predict(input_frames)
    return prediction

def process_realtime_video():
    # Inicializar a captura de vídeo
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1536)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 864)
    
    frames = []
    frame_count = 0
    predicted_class_label = ""
    prediction_percentages = []

    # Taxa de coleta de frames
    frame_interval = 0.2  # Coletar a cada 200ms
    last_frame_time = time.time()

    # Inicializar o MediaPipe Pose e Hands
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Converter a imagem para RGB
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Processar a imagem com o MediaPipe Pose
            pose_results = pose.process(image_rgb)
            # Processar a imagem com o MediaPipe Hands
            hand_results = hands.process(image_rgb)

            # Criar uma imagem preta para a máscara
            mask_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            # Desenhar as marcas do MediaPipe Pose na máscara
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(mask_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Desenhar as marcas do MediaPipe Hands na máscara
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(mask_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coletar os frames em intervalos consistentes
            if time.time() - last_frame_time >= frame_interval:
                resized_mask = cv.resize(mask_frame, target_size) / 255.0  # Redimensionar e normalizar
                frames.append(resized_mask)
                last_frame_time = time.time()
            
            if len(frames) == num_frames:  # Processar quando atingir 60 frames
                prediction = process_frames(frames, model)
                predicted_class = np.argmax(prediction, axis=-1)[0]
                predicted_class_label = class_labels[predicted_class]
                prediction_percentages = prediction[0] * 100  # Obter as porcentagens
                recent_predictions.append(predicted_class)  # Adicionar a classe à fila
                frames = []  # Limpar os frames

            # Calcular a classe mais frequente na janela deslizante
            if recent_predictions:
                final_class = max(set(recent_predictions), key=recent_predictions.count)
                final_class_label = class_labels[final_class]
            else:
                final_class_label = "Calculando..."

            # Mostrar a predição no frame
            cv.putText(frame, f"Predicted: {final_class_label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            for i, (label, percentage) in enumerate(zip(class_labels, prediction_percentages)):
                cv.putText(frame, f"{label}: {percentage:.2f}%", (10, 60 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            # Mostrar o vídeo real e a máscara
            cv.imshow('Real Video with Predictions', frame)
            cv.imshow('MediaPipe Mask', mask_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar a captura de vídeo e fechar as janelas
    cap.release()
    cv.destroyAllWindows()

# Chamar a função para processar o vídeo em tempo real
process_realtime_video()
