"""
Make a mini-RAG on a sample of the corpus (250 lines).
This script was written with the help of ChatGPT.
"""
import csv
from openai import OpenAI
import glob
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import instructor
from pydantic import BaseModel, Field
from typing import List

# --- CONFIG ---

CSV_FOLDER = './data'
PROMPT_FILE = './prompt.txt'
COLUMN_MESSAGE_INDEX = 2  # index de la colonne 'message'
COLUMN_DESCRIPTION_INDEX = 3  # index de la colonne 'description'

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Utilisation d'un modèle rapide

# --- SETUP ---

# Charger le modèle d'embed
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = instructor.from_openai( # initialisation du client
    OpenAI(
        base_url="http://localhost:5005/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

# --- SCHEMAS Pydantic ---

class Section(BaseModel):
    tag: str = Field(description="Activité mentionnée (sports, politique, agriculture, etc.)")
    keyword: str = Field(description="Mot-clé justifiant le tag")
    excerpt: str = Field(description="Extrait de contexte autour du mot-clé")

class MetadataExtraction(BaseModel):
    summary: str = Field(description="Résumé du texte (< 30 mots)")
    location: str = Field(description="Lieu où les activités se passent (si mentionné)")
    sections: List[Section] = Field(description="Liste des sections avec activités")

# Charger le prompt
def load_prompt(path):
    with open(path, "r") as f:
        return f.read()

prompt_template = load_prompt(PROMPT_FILE)

# --- LIRE LES DESCRIPTIONS ET MESSAGES DES CSV ---

def read_descriptions_and_messages(folder):
    combined_texts = []
    files = glob.glob(f"{folder}/*.csv")
    for file in files:
        with open(file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Sauter l'en-tête
            for row in reader:
                if len(row) > max(COLUMN_MESSAGE_INDEX, COLUMN_DESCRIPTION_INDEX):
                    message = row[COLUMN_MESSAGE_INDEX].strip()
                    description = row[COLUMN_DESCRIPTION_INDEX].strip()
                    if message and description:
                        combined_texts.append(f"{message} {description}")
    return combined_texts

combined_texts = read_descriptions_and_messages(CSV_FOLDER)

# --- EMBEDDING & FAISS ---

# Encoder les descriptions et messages
embeddings = embedder.encode(combined_texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')
print(f"Encodage terminé.")

# Créer l'index FAISS (Facebook AI Similarity Search)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("Index FAISS créé.")
# Vérifier que l'index FAISS contient bien des données
if index.ntotal == 0:
    print("L'index FAISS est vide. Aucun embedding n'a été ajouté.")
else:
    print(f"L'index FAISS contient {index.ntotal} éléments.")

# --- RAG POUR CHAQUE DESCRIPTION/MESSAGE COMBINÉS ---

for i, combined_text in enumerate(combined_texts):
    print(f"\n Traitement de la ligne {i+1}")

    # Encoder le texte combiné actuel (message + description)
    query_emb = embedder.encode([combined_text], normalize_embeddings=True).astype('float32')

    # Rechercher les textes similaires dans l'index
    print(f"Recherche dans l'index FAISS pour la ligne {i+1}...")
    D, I = index.search(query_emb, k=5)  # top 5 similaires

    if len(I[0]) == 0:  # Si aucune correspondance n'est trouvée
        print(f"Aucune correspondance trouvée pour la ligne {i + 1}. Passage à la suivante.")
        continue  # Passer à la ligne suivante

    similar_texts = [combined_texts[idx] for idx in I[0]]

    # Construire le contexte pour le prompt
    context = "\n\n".join(similar_texts)

    # Générer le prompt avec le contexte
    real_prompt = prompt_template.format(input_text=context)

    # Appeler le modèle avec `Instructor`
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-r1:70b", 
            messages=[{"role": "system", "content": real_prompt}],
            response_model=MetadataExtraction
        )
        for section in response.sections: # rint en temps réel les tags reconnus pour voir où on en est
            print(f"Tag: {section.tag}, Keyword: {section.keyword}, Excerpt: {section.excerpt}")
        print(response.model_dump_json(indent=2))  # Afficher la réponse en format JSON
    except Exception as e:
        print(f" Erreur sur la ligne {i+1}: {e}")


    end_time = time.time()

    print(f" Temps de traitement: {end_time - start_time:.2f} secondes")
