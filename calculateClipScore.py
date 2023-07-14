import os
import json
import torch
import torchvision.transforms as transforms
from torchmetrics.functional.multimodal import clip_score
import clip
from PIL import Image
from tqdm import tqdm

# Cargar el modelo CLIP pre-entrenado

# Directorios de las carpetas
folder_hiper = "hiper"
folder_plugAndPlay = "plugAndPlay"
folder_imagic = "tedbench\imagic"

'''
def get_clip_score(image_path, text):
    global device, model, preprocess

    image = Image.open(image_path)

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score
'''

def calculate_clip_scores(data, folder_name):
    metric = clip_score(model_name_or_path="openai/clip-vit-base-patch16")

    clip_scores = []

    # Ruta del archivo de resultados
    result_file = os.path.join(folder_name, "clip_scores.txt")

    # Calcular el ClipScore para cada imagen en el archivo JSON
    for item in tqdm(data):
        prompt = item["prompt"]
        output_path = os.path.join(folder_name, item["output"])#os.path.join(folder_name, "output", item["output"])

        # Verificar si el archivo de salida existe
        if not os.path.isfile(output_path):
            print("Image not founded")
            print(output_path)
            continue

        # Cargar la imagen y aplicar transformaciones
        image = Image.open(output_path).convert("RGB")
        transform = transforms.ToTensor()
        tensor_image = transform(image).unsqueeze(0)
        #score = get_clip_score(output_path,prompt)
        score = metric(tensor_image, prompt)

        # Agregar el resultado a la lista de ClipScores
        clip_scores.append(score.item())

    # Guardar los resultados en el archivo
    with open(result_file, "w") as file:
        for score in clip_scores:
            file.write(str(score) + "\n")

    # Calcular el promedio de los ClipScores
    average_score = sum(clip_scores) / len(clip_scores)
    print("Promedio de ClipScores en", folder_name, ":", average_score)

# Procesar la carpeta "hiper"
#json_file_hiper = os.path.join(folder_hiper, "info.json")
#with open(json_file_hiper, "r") as file:
#    data_hiper = json.load(file)
#    print("Comienza calculo del score hiper")
#    calculate_clip_scores(data_hiper, folder_hiper)

# Procesar la carpeta "plugAndPlay"
#json_file_plugAndPlay = os.path.join(folder_plugAndPlay, "info.json")
#with open(json_file_plugAndPlay, "r") as file:
#    data_plugAndPlay = json.load(file)
#    print("Comienza calculo del score plug and play")
#    calculate_clip_scores(data_plugAndPlay, folder_plugAndPlay)

# Procesar la carpeta "imagic"
json_file_imagic = os.path.join(folder_imagic, "info.json")
with open(json_file_imagic, "r") as file:
    data_imagic = json.load(file)
    print("Comienza calculo del score Imagic")
    calculate_clip_scores(data_imagic, folder_imagic)