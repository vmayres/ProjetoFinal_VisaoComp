import os
import cv2 as cv
import mediapipe as mp
import numpy as np

#* Inicializar o MediaPipe Hands e Pose
# Responsável por detectar as mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Responsável por detectar as poses (braços, rosto e tronco)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

#* Diretório do dataset original e do dataset de máscaras
original_dataset_dir = r"D:\libras-dataset" # Dataset original (interpratação de sinais)
mask_dataset_dir = r"D:\libras-mask-pose"   # Dataset criado somente com as máscaras 

#* Criar a pasta raiz do novo dataset, se não existir
if not os.path.exists(mask_dataset_dir):
    os.makedirs(mask_dataset_dir)

#* Funcoes para aplicar data augmentation ( Zoom e Shift ) nas imagens
# Função para aplicar zoom no centro da imagem
def apply_zoom(image, zoom_factor=1.2):
    height, width = image.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    y1, x1 = (height - new_height) // 2, (width - new_width) // 2
    y2, x2 = y1 + new_height, x1 + new_width
    zoomed_image = cv.resize(image[y1:y2, x1:x2], (width, height))
    return zoomed_image

# Função para deslocar a imagem para a esquerda ou direita
def shift_image(image, shift_pixels):
    height, width = image.shape[:2]
    M = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
    shifted_image = cv.warpAffine(image, M, (width, height))
    return shifted_image

#* Função para criar máscaras a partir de vídeos e aplicar data augmentation
def process_videos_in_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Determinar o caminho relativo e criar diretórios correspondentes no novo dataset
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)

        # Processar cada arquivo de vídeo
        for file in files:
            if file.endswith('.mp4'):  # Apenas arquivos de vídeo
                video_path = os.path.join(root, file)
                output_path_original = os.path.join(target_dir, f"mask_{file}")
                output_path_zoomed = os.path.join(target_dir, f"zoom_{file}")
                output_path_shifted_left = os.path.join(target_dir, f"shiftL_{file}")
                output_path_shifted_right = os.path.join(target_dir, f"shiftR_{file}")
                
                # Processar o vídeo
                cap = cv.VideoCapture(video_path)
                
                # Obter propriedades do vídeo
                frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv.CAP_PROP_FPS)
                
                # Criar os objetos VideoWriter para salvar os vídeos de máscara
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out_original = cv.VideoWriter(output_path_original, fourcc, fps, (frame_width, frame_height))
                out_zoomed = cv.VideoWriter(output_path_zoomed, fourcc, fps, (frame_width, frame_height))
                out_shifted_left = cv.VideoWriter(output_path_shifted_left, fourcc, fps, (frame_width, frame_height))
                out_shifted_right = cv.VideoWriter(output_path_shifted_right, fourcc, fps, (frame_width, frame_height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    #* Criar uma imagem preta
                    mask_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                    #* Converter a imagem para RGB
                    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    
                    #* Processar a imagem com o MediaPipe
                    hand_results = hands.process(image_rgb)
                    pose_results = pose.process(image_rgb)

                    #* Desenhar as marcas do MediaPipe
                    # MediaPipe Hands 
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(mask_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # MediaPipe Pose
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(mask_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    #* Aplica data augmentation nas imagens
                    zoomed_frame = apply_zoom(mask_frame, zoom_factor=1.2)                                  # Aplicar zoom   
                    shifted_left_frame = shift_image(zoomed_frame, shift_pixels=-int(frame_width * 0.1))    # Deslocar a imagem para a esquerda
                    shifted_right_frame = shift_image(zoomed_frame, shift_pixels=int(frame_width * 0.1))    # Deslocar a imagem para a direita

                    #* Escrever os frames processados nos vídeos de saída
                    out_original.write(mask_frame)
                    out_zoomed.write(zoomed_frame)
                    out_shifted_left.write(shifted_left_frame)
                    out_shifted_right.write(shifted_right_frame)

                #* Liberar os objetos de captura e escrita de vídeo
                cap.release()
                out_original.release()
                out_zoomed.release()
                out_shifted_left.release()
                out_shifted_right.release()

# Chamar a função para processar os vídeos no diretório
process_videos_in_directory(original_dataset_dir, mask_dataset_dir)

# Liberar o MediaPipe Hands
hands.close()
pose.close()
