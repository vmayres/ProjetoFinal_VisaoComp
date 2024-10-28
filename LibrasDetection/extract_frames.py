import cv2

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

# Uso do código
video_path = r'Dataset\Aluno\02AlunoSinalizador01-2.mp4'
frames = extract_frames(video_path)

# Exemplo de verificação de quantos frames foram extraídos
print(f"Total de frames extraidos: {len(frames)}")
