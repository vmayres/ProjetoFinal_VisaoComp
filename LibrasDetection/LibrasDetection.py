#Bibliotecas
import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Set default encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# Carregar o modelo
model_path = r"D:\ProjetoFinal_VisaoComp\model_teste20classes.keras"
model = load_model(model_path)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Função para processar os frames e fazer a predição
def process_frames(frames, model):
    # Redimensionar os frames para o tamanho esperado pelo modelo
    resized_frames = [cv2.resize(frame, (64, 64)) for frame in frames]
    # Normalizar os valores dos pixels
    input_frames = np.array(resized_frames) / 255.0
    # Ajustar a forma da entrada para (número de frames, 64, 64, 3)
    input_frames = input_frames.reshape((len(frames), 64, 64, 3))
    # Expandir a dimensão para corresponder à entrada do modelo
    input_frames = np.expand_dims(input_frames, axis=0)
    
    # Fazer a predição
    prediction = model.predict(input_frames)
    
    # Obter a classe prevista
    predicted_class = np.argmax(prediction, axis=-1)
    
    # Imprimir a predição no terminal
    print(f"Predicted: {predicted_class[0]}")
    
    return prediction

if not cap.isOpened():
    print("Erro ao abrir a câmera")
frames = []

# Capturar frames da câmera e processar
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    
    if len(frames) == 30:  # Processar 30 frames de cada vez
        prediction = process_frames(frames, model)
        frames = []  # Limpar a lista de frames para capturar os próximos

    # Mostrar o frame capturado
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()