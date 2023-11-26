from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
import os
import numpy as np
import pandas as pd
from tqdm import tqdm 
import yaml
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import json

class MyMetrics(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = Image.open(image)
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)

        cosine_similarity_I = F.cosine_similarity(img_feat_one, text_feat_two)
        cosine_similarity_T = F.cosine_similarity(img_feat_two, text_feat_two)

        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )

        output = {
            'directional_similarity': abs(float(directional_similarity.detach().cpu())),
            'Clip-I': float(cosine_similarity_I.detach().cpu()),
            'Clip-T': float(cosine_similarity_T.detach().cpu())
        }

        del img_feat_one, img_feat_two, text_feat_one, text_feat_two

        return output

device = "cuda"

clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

myMetrics = MyMetrics(tokenizer, text_encoder, image_processor, image_encoder)
scores = []


dataset = []


input_images = []
original_captions = []
modified_captions = []
edited_images = []

root = "C:/Tesis/pnp-diffusers/"
folder = "C:/Tesis/pnp-diffusers/config"

yamls = [f for f in os.listdir(folder) if f.endswith('.yaml')]


for file in yamls:
    with open(folder + "/" + file, 'r') as file:
        config = yaml.safe_load(file)
        original_image_path = config['image_path']
        output_dir = config['output_path']
        prompt = config['prompt']
        edited_image = root + output_dir + "/output-"+prompt+".png"

        input_images.append(original_image_path)
        original_captions.append("")
        edited_images.append(edited_image)
        modified_captions.append(prompt)

sum_directional_similarity = 0.0
sum_cosine_similarity_I = 0.0
sum_cosine_similarity_T = 0.0

for i in tqdm(range(len(input_images))):
    original_image = input_images[i]
    original_caption = original_captions[i]
    edited_image = edited_images[i]
    modified_caption = modified_captions[i]

    similarity_score = myMetrics(original_image, edited_image, original_caption, modified_caption)
    similarity_score["original"] = original_image
    similarity_score["prompt"] = modified_caption
    scores.append(similarity_score)

#Calcular la media
for output_dict in scores:
    sum_directional_similarity += output_dict['directional_similarity']
    sum_cosine_similarity_I += output_dict['Clip-I']
    sum_cosine_similarity_T += output_dict['Clip-T']

num_elements = len(scores)
mean_directional_similarity = sum_directional_similarity / num_elements
mean_cosine_similarity_I = sum_cosine_similarity_I / num_elements
mean_cosine_similarity_T = sum_cosine_similarity_T / num_elements

# Crea un nuevo diccionario con las medias calculadas
mean_output = {
    'Directional_similarity': mean_directional_similarity,
    'Clip-I': mean_cosine_similarity_I,
    'Clip-T': mean_cosine_similarity_T
}

#print(f"CLIP directional similarity: {np.mean(scores)}")
# CLIP directional similarity: 0.0797976553440094

#print(scores)
#with open('datos.json', 'w') as archivo:
#    json.dump(scores, archivo)

with open('results.json', 'w') as archivo:
    json.dump(mean_output, archivo)