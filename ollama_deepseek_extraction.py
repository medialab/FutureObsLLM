"""
Pre-process textual data in csv files and extract tags from a text.
Inspired from:
https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-structured/instructor_generate.py
https://github.com/instructor-ai/instructor/blob/main/docs/concepts/retrying.md
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import glob
import re
import time
import csv
import instructor
from instructor.exceptions import InstructorRetryException

# variables

prompt = './shorter_prompt.txt'
folder = './data'

# regex pattern
pattern = re.compile(
    r'http\S+|www\.\S+|'      # urls
    r'#\w+|'                  # hashtags
    r'['
        u"\U0001F600-\U0001F64F"  # emojis
        u"\U0001F300-\U0001F5FF"  # symbols & pictograms
        u"\U0001F680-\U0001F6FF"  # transport & cards
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # diverse symbols
        u"\U000024C2-\U0001F251"  # other characters
        r']+', 
        flags=re.UNICODE
)

# functions

def load_prompt(prompt): # load prompt
    with open(prompt, 'r') as f:
        return f.read()

def read_files(folder): # load csv files
    files = glob.glob(f"{folder}/*.csv")
    return files 

def pre_processing(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            titre = row.get('title', '') or ''
            description = row.get('description', '') or ''
            message = row.get('message', '') or ''
            merged_text = f"{titre} {description} {message}" # merge rows titre, description et message
            cleaned_text = pattern.sub('', merged_text) # appliquer le pattern regex
            print(f"Texte nettoyé pour la ligne : {cleaned_text}") # verifier que ça nettoie le texte de chaque ligne
            yield cleaned_text


def process_csv_line(line, prompt_template): # read csv files row by row

    real_prompt = prompt_template.format(input_text=line) 

    try:
        response = client.chat.completions.create(
            model="deepseek-r1:70b", 
            messages=[
                {"role": "system", 
                "content": "You are a world renowned algorithm capable of extracting accurate tags from text"},
                {"role": "user", "content":real_prompt
                }
            ],
            response_model=MetadataExtraction,
        )
        return response.model_dump_json(indent=2)  # json response
    except InstructorRetryException as e:
        print(f"Erreur dans la requête pour la ligne : {line}.")
        return None 

def write_result(output_file, data):
    print(f"{data}")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(data + '\n')
        

def process_csv_file(file_path, prompt_template):
    start_time = time.time()
    output_file = file_path.replace('.csv', '_results') 

    for text_to_process in pre_processing(file_path):
        response_json = process_csv_line(text_to_process, prompt_template)
        if response_json:
            write_result(output_file, response_json)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTemps total pour le fichier {file_path} : {elapsed_time:.2f} secondes")


# classes

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

# execution 

prompt_template = load_prompt(prompt)

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:5005/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

files = read_files(folder)

for file_path in files:
    process_csv_file(file_path, prompt_template)