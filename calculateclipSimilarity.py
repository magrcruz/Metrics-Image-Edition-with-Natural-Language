## clip similarity 
import os
import json
import torch
import torchvision.transforms as transforms
from torchmetrics.functional.multimodal import clip_score
import clip
from PIL import Image
from tqdm import tqdm
from clipSimilarity import ClipSimilarity

# Directorios de las carpetas
folder_hiper = "hiper"
folder_plugAndPlay = "plugAndPlay"


def calculate_clip_scores(data, folder_name):
    clip_scores = []
    clip_similarity = ClipSimilarity()

    transform = transforms.ToTensor()

    # Ruta del archivo de resultados
    result_file = os.path.join(folder_name, "clip_similarity.txt")

    sim_0_avg = 0
    sim_1_avg = 0
    sim_direction_avg = 0
    sim_image_avg = 0
    count = 0

    # Calcular el ClipScore para cada imagen en el archivo JSON
    for item in tqdm(data):
        prompt = item["prompt"]
        #output_path = os.path.join(folder_name, "output", item["output"])
        #input_path = os.path.join(folder_name, "input", item["input"])
        output_path = os.path.join(folder_name, item["output"])
        input_path = os.path.join(folder_name,"input", item["input"])
        image_text = item["image_text"]

        # Verificar si el archivo de salida existe
        if not os.path.isfile(output_path):
            continue
        if not os.path.isfile(input_path):
            continue
        print("Paso x aqui")

        # Cargar la imagen y aplicar transformaciones
        image0 = Image.open(input_path).convert("RGB")
        tensor_image0 = transform(image0).unsqueeze(0)
        image1 = Image.open(output_path).convert("RGB")
        tensor_image1 = transform(image1).unsqueeze(0)

        sim_0, sim_1, sim_direction, sim_image = clip_similarity(
            tensor_image0, tensor_image1, image_text, prompt
        )
        sim_0_avg += sim_0.item()
        sim_1_avg += sim_1.item()
        sim_direction_avg += sim_direction.item()
        sim_image_avg += sim_image.item()
        count += 1

        # Agregar el resultado a la lista de ClipScores
        # clip_scores.append(score.item())

    # Guardar los resultados en el archivo
    with open(result_file, "w") as file:
        for score in clip_scores:
            file.write(str(score) + "\n")

    # Calcular el promedio de los ClipScores
    sim_0_avg /= count
    sim_1_avg /= count
    sim_direction_avg /= count
    sim_image_avg /= count

    print(
        f"{json.dumps(dict(sim_0=sim_0_avg, sim_1=sim_1_avg, sim_direction=sim_direction_avg, sim_image=sim_image_avg))}\n"
    )


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
folder_imagic = "tedbench\imagic"

json_file_imagic = os.path.join(folder_imagic, "info.json")
with open(json_file_imagic, "r") as file:
    data_imagic = json.load(file)
    print("Comienza calculo del score plug and play")
    calculate_clip_scores(data_imagic, folder_imagic)