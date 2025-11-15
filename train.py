import cv2
import os

# Pasta para guardar os dados de treino
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

name = input("Qual é o nome da pessoa? ")
person_dir = os.path.join(dataset_dir, name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

cap = cv2.VideoCapture(0)
count = 0
total_photos = 500

print(f"Começando a captura de {total_photos} fotos para {name}...")

while count < total_photos:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Mostra a imagem
    cv2.imshow("Captura de rosto", frame)
    
    # Salva a imagem
    img_path = os.path.join(person_dir, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1
    
    # Sai se pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Captura concluída!")
cap.release()
cv2.destroyAllWindows()
