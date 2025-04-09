"""
Pre-process CSV files :
Remove special characters (including accents)
Remove emojis
Remove urls
Remove hashtags

"""

import csv 
import glob
import os
import re
import unicodedata

def remove_accents(merged_text):
    return ''.join(x for x in unicodedata.normalize('NFKD', merged_text) if unicodedata.category(x)[0] in ('L', 'Z', 'N')).lower() 
    # enlever les accents des strings avant de clean le text mais en gardant espaces et chiffres


def clean_text(text):
    #text = re.sub(r'\s', ' ', text) # mettre tous les espaces en espaces normaux
    text = re.sub(r'http\S+|www\.\S+', '', text) # supprimer urls
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text) # supprimer unicode emojis
    text = re.sub(r'#\w+', '', text) # supprimer hashtags
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text) # garder uniquement chiffres et lettres
    return text


files_path = './data/*.csv'

resultats = []

for csv_file in glob.glob(files_path):
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for ligne in reader:
            id_val = ligne.get('id')
            titre = ligne.get('title', '')
            description = ligne.get('description', '')
            message = ligne.get('message', '')
            
            # merge les colonnes titre description et message
            merged_text = f"{titre} {description} {message}"

            # enlever accents partout
            unaccented_text = remove_accents(merged_text)
            
            # Nettoyer le texte
            cleaned_text = clean_text(unaccented_text)
            
            # Ajouter au résultat
            resultats.append({'id': id_val, 'texte': cleaned_text})


output_path = './data/resultats/result.csv'

with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
    champs = ['id', 'texte']
    post = csv.DictWriter(f_out, fieldnames=champs)
    post.writeheader()
    for element in resultats:
        post.writerow(element)