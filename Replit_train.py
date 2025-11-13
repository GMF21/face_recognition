import face_recognition
import os
import pickle

# Pasta com imagens conhecidas
path = "imagens"
known_faces = []
known_names = []

if not os.path.exists(path):
    print(f"❌ Erro: pasta '{path}' não encontrada!")
    print(f"Por favor, crie a pasta '{path}' e adicione imagens .jpg ou .png")
    exit(1)

image_files = [f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]

if not image_files:
    print(f"❌ Erro: nenhuma imagem encontrada na pasta '{path}'!")
    print("Por favor, adicione imagens .jpg ou .png com os nomes das pessoas")
    print("Exemplo: joao.jpg, maria.png")
    exit(1)

for file_name in image_files:
    image_path = os.path.join(path, file_name)
    print(f"Processando: {file_name}")
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
        print(f"⚠️  Aviso: nenhum rosto encontrado em {file_name}, pulando...")
        continue
    
    encoding = encodings[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(file_name)[0])

if not known_faces:
    print("❌ Erro: nenhum rosto foi detectado nas imagens!")
    print("Certifique-se de que as imagens contêm rostos claramente visíveis")
    exit(1)

# Guarda os dados treinados num ficheiro
with open("modelo_faces.pkl", "wb") as f:
    pickle.dump((known_faces, known_names), f)

print(f"✅ Treino concluído! {len(known_faces)} rosto(s) reconhecido(s) e modelo guardado!")
