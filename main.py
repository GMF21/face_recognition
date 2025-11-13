import cv2
import pygame
import face_recognition
import pickle
import numpy as np

# Carregar modelo treinado
with open("modelo_faces.pkl", "rb") as f:
    known_faces, known_names = pickle.load(f)

# Inicializa pygame e camera
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Verificação Facial")

camera = cv2.VideoCapture(0)
font = pygame.font.SysFont("Arial", 32)

running = True
status_text = "Pressione ESPAÇO para verificar"

while running:
    ret, frame = camera.read()
    if not ret:
        break

    # Converte imagem para exibir no pygame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))

    # Mostra imagem e texto
    screen.blit(frame_surface, (0, 0))
    text_surface = font.render(status_text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Verificar rosto
                rgb_small = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
                encodings = face_recognition.face_encodings(rgb_small)

                if encodings:
                    match_results = face_recognition.compare_faces(known_faces, encodings[0])
                    name = "DESCONHECIDO"

                    if True in match_results:
                        name = known_names[match_results.index(True)]
                        status_text = f"Acesso PERMITIDO: {name}"
                    else:
                        status_text = "Acesso NEGADO!"
                else:
                    status_text = "Nenhum rosto detetado."

camera.release()
pygame.quit()
