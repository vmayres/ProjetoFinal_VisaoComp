import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Configuração do MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carregar o modelo
model_path = r"D:\ProjetoFinal_VisaoComp\model_teste20classes.keras"
model = load_model(model_path)

# Parâmetros
target_size = (64, 64)  # Dimensão dos frames
num_frames = 60  # Número de frames consecutivos
class_labels = ['Classe1', 'Classe2', 'Classe3', '...']  # Substitua pelos nomes das classes

def process_realtime_video():
    cap = cv.VideoCapture(0)  # Inicializa a captura da câmera
    frame_buffer = deque(maxlen=num_frames)  # Buffer para frames

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Converter o frame para RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            # Criar uma máscara preta
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # Desenhar landmarks na máscara
            if result.pose_landmarks:
                color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                mp_drawing.draw_landmarks(
                    color_mask,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),  # Branco para conexões
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)   # Branco para pontos
                )
                mask = cv.cvtColor(color_mask, cv.COLOR_BGR2GRAY)

            # Converter a máscara para 3 canais e redimensionar
            final_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            resized_mask = cv.resize(final_mask, target_size)

            # Normalizar e adicionar ao buffer
            frame_buffer.append(resized_mask / 255.0)  # Normalizado para [0, 1]

            # Fazer predição quando o buffer estiver cheio
            if len(frame_buffer) == num_frames:
                input_data = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
                prediction = model.predict(input_data)
                predicted_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Mostrar previsão no vídeo
                cv.putText(
                    frame,
                    f"{predicted_class} ({confidence:.2f})",
                    (10, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),  # Azul
                    3
                )

            # Exibir vídeo com a previsão
            cv.imshow('Reconhecimento em Tempo Real', frame)

            # Pressione 'q' para sair
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

# Executar o reconhecimento em tempo real
process_realtime_video()
