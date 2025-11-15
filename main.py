import pygame
import cv2
import os
from deepface import DeepFace
import numpy as np

# Inicializar Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Face Recognition Game")
clock = pygame.time.Clock()

# Inicializar a webcam
cap = cv2.VideoCapture(0)

# Carregar embeddings de treino
dataset_dir = "dataset"
known_embeddings = []
known_names = []

print("🔄 Gerando embeddings das imagens de treino...")
for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        embedding = DeepFace.represent(img_path, model_name="Facenet")[0]["embedding"]
        known_embeddings.append(embedding)
        known_names.append(person)

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Loop principal do Pygame
running = True
font = pygame.font.SysFont(None, 48)

while running:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Reconhecimento facial
        result = DeepFace.find(frame_rgb, db_path=dataset_dir, enforce_detection=False, model_name="Facenet")
        if len(result) > 0:
            detected_name = result[0]['identity'][0].split(os.sep)[-2]
        else:
            detected_name = "Desconhecido"
    except:
        detected_name = "Desconhecido"

    # Converte para Pygame
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_surface = pygame.surfarray.make_surface(frame_rgb).convert()
    frame_surface = pygame.transform.rotate(frame_surface, -90)
    frame_surface = pygame.transform.flip(frame_surface, True, False)
    
    # Mostrar na tela
    screen.blit(frame_surface, (0,0))
    
    # Mostrar nome da pessoa
    text = font.render(detected_name, True, (255,0,0))
    screen.blit(text, (50,50))
    
    pygame.display.update()
    clock.tick(30)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
