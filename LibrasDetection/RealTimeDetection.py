import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import sys
import os
import io

# Configurar o ambiente para UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Inicializar o MediaPipe Hands e Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carregar o modelo treinado
model_path = r"C:\Users\victor\OneDrive\Documentos\GitHub\ProjetoFinal_VisaoComp\model_mask10classes.keras"
model = load_model(model_path)

# Parâmetros
altura, largura, canais = 64, 64, 1  # Dimensões e canais
num_frames = 30  # Número de frames esperados
target_size = (largura, altura)
frame_rate = 30  # FPS da câmera

class_labels = ['Acontecer', 'Amarelo', 'America', 'Aproveitar', 'Banheiro', 
                'Barulho', 'Cinco', 'Espelho', 'Esquina', 'Medo']

# Função para processar frames capturados
def process_video_frames(frames, target_size, num_frames, model):
    resized_frames = [cv.resize(frame, target_size) / 255.0 for frame in frames]

    if len(resized_frames) > num_frames:
        indices = np.linspace(0, len(resized_frames) - 1, num_frames).astype(int)
        resized_frames = [resized_frames[i] for i in indices]
    elif len(resized_frames) < num_frames:
        padding = num_frames - len(resized_frames)
        resized_frames.extend([resized_frames[-1]] * padding)

    input_frames = np.array(resized_frames).reshape((1, num_frames, *target_size, canais))
    prediction = model.predict(input_frames)
    return prediction

# Função principal para capturar e processar vídeo em tempo real
def realtime_prediction():
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640*1.2)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480*1.2)

    frames = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Converter para RGB e aplicar MediaPipe
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            hand_results = hands.process(image_rgb)

            # Criar máscara preta e desenhar landmarks
            mask_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(mask_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(mask_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Converter máscara para escala de cinza e acumular os frames
            gray_mask = cv.cvtColor(mask_frame, cv.COLOR_BGR2GRAY)
            frames.append(gray_mask)

            # Manter apenas os últimos `num_frames` frames
            if len(frames) > num_frames:
                frames.pop(0)

            # Fazer a previsão quando tivermos frames suficientes
            if len(frames) == num_frames:
                prediction = process_video_frames(frames, target_size, num_frames, model)
                predicted_class = np.argmax(prediction)
                prediction_percentages = prediction[0] * 100

                # Adicionar resultados no feed ao vivo
                cv.putText(frame, f"Classe: {class_labels[predicted_class]}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                for i, (label, percentage) in enumerate(zip(class_labels, prediction_percentages)):
                    cv.putText(frame, f"{label}: {percentage:.2f}%", (10, 80 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            # Mostrar o feed de vídeo com as previsões
            cv.imshow("Real-Time Video with Prediction", frame)

            # Tecla 'q' para sair
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

# Executar o programa
realtime_prediction()