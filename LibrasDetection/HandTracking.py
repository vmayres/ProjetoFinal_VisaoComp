import os
import cv2 as cv
import mediapipe as mp

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Diretório do dataset original e do dataset de máscaras
original_dataset_dir = r"D:\libras-dataset"
mask_dataset_dir = r"D:\libras-mask"

# Criar a pasta raiz do novo dataset, se não existir
if not os.path.exists(mask_dataset_dir):
    os.makedirs(mask_dataset_dir)

# Função para criar máscaras a partir de vídeos
def process_videos_in_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Determinar o caminho relativo e criar diretórios correspondentes no novo dataset
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Processar cada arquivo de vídeo
        for file in files:
            if file.endswith('.mp4'):  # Apenas arquivos de vídeo
                video_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, f"mask_{file}")
                
                # Processar o vídeo
                cap = cv.VideoCapture(video_path)
                
                # Obter propriedades do vídeo
                frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv.CAP_PROP_FPS)
                
                # Criar o objeto VideoWriter para salvar o vídeo de máscara
                out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), False)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Criar uma máscara preta com o mesmo tamanho do quadro
                    mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    mask.fill(0)  # Tornar a máscara preta
                    
                    # Converter o quadro BGR para RGB
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    
                    # Processar o quadro e detectar pose
                    result = pose.process(rgb_frame)
                    
                    # Desenhar landmarks da pose na máscara
                    if result.pose_landmarks:
                        # Converter a máscara para uma imagem colorida
                        color_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                        
                        mp_drawing.draw_landmarks(
                            color_mask,
                            result.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),  # Conexões brancas
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)   # Pontos brancos
                        )
                        
                        # Converter a máscara colorida de volta para escala de cinza
                        mask = cv.cvtColor(color_mask, cv.COLOR_BGR2GRAY)
                    
                    # Escrever o quadro de máscara no vídeo de saída
                    out.write(mask)
                
                # Liberar os objetos de captura e escrita
                cap.release()
                out.release()

# Processar todos os vídeos no dataset original
process_videos_in_directory(original_dataset_dir, mask_dataset_dir)

# Liberar o MediaPipe Pose
pose.close()
