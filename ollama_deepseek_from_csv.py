from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import glob
import time
import csv
import instructor
from instructor.exceptions import InstructorRetryException

prompt = './shorter_prompt.txt'
folder = './data'

def load_prompt(prompt):
    with open(prompt, 'r') as f:
        return f.read()

def read_files(files):
    files = glob.glob(f"{folder}/*.csv")
    return files 

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:5005/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

class Section(BaseModel):
    tag: str = Field(
        description="Human activity mentioned in this section of the text, \
        such as politics, sports, fishing, agriculture, tourism..."
    )
    keyword: str = Field(
        description="Word from the text justifying the tag. Example: the keyword 'football' justifies the tag 'sports'"
    )
    excerpt: str = Field(
        description="Small excerpt from the text giving more context to the extracted word. Example for the word football: 'la coupe du monde de football'"
    )

class MetadataExtraction(BaseModel):
    """Extracted metadata about an example from the Modal examples repo."""
    summary: str = Field(
        ..., description="A brief summary of the text (less than 30 words)."
    )
    location: str = Field(
        ..., description="The place where human activities take place, if any."
    )
    sections: List[Section] = Field(
        description="A list of small excerpts of the document mentioning human activities."
    )

class ResponseModel(BaseModel):
    analysis: str

prompt_template = load_prompt(prompt)

def process_csv_line(line, prompt_template):

    real_prompt = prompt_template.format(input_text=line)  

    try:
        response = client.chat.completions.create(
            model="deepseek-r1:70b", 
            messages=[
                {"role": "system", "content": real_prompt}
            ],
            response_model=MetadataExtraction,
        )
        return response.model_dump_json(indent=2)  # réponse au format json
    except InstructorRetryException as e:
        print(f"Erreur dans la requête pour la ligne : {line}.")
        return None 

def process_csv_file(file):
    start_time = time.time()

    # ouvrir csv et lire ligne par ligne
    with open(file, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            text_to_process = row[9]
            response_json = process_csv_line(text_to_process, prompt_template)
            
            if response_json:
                print(f"Réponse pour la ligne : {response_json}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTemps total pour le fichier {file} : {elapsed_time:.2f} secondes")

files = read_files(folder)

for file in files:
    process_csv_file(file)

