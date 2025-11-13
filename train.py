import face_recognition
import os
import pickle

# Pasta com imagens conhecidas
path = "imagens"
known_faces = []
known_names = []

for file_name in os.listdir(path):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(path, file_name))
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(file_name)[0])

# Guarda os dados treinados num ficheiro
with open("modelo_faces.pkl", "wb") as f:
    pickle.dump((known_faces, known_names), f)

print("✅ Treino concluído e modelo guardado!")
