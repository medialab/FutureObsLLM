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
import os
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
        u"\U0001F900-\U0001F9FF" # other symbols and pictograms
        u"\U0001F680-\U0001F6FF"  # transport & cards
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # diverse symbols
        u"\U0001FA70-\U0001FAFF" # other symbols
        u"\U00002600-\U000026FF" # miscellaneous symbols
        u"\U000024C2-\U0001F251"  # other characters
        u"\U0001F780-\U0001F7FF" # geometric shapes
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
            title = row.get('title', '')
            id_row = row.get('id', '') # get id from original csv files
            description = row.get('description', '')
            message = row.get('message', '')
            merged_text = f"{title} {description} {message}" # merge rows "titre", "description" and "message"
            cleaned_text = pattern.sub('', merged_text) # apply regex pattern
            print(f"Text cleaned for row : {cleaned_text}") # check that each row has been cleaned
            yield cleaned_text, id_row

def write_result(output_file, data):
    print(f"{data}")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(data + '\n')

def write_csv(output_csv_path, data_rows):  # csv output
    fieldnames = ['id', 'row', 'file', 'summary', 'location', 'category', 'keyword', 'context', 'impact', 'impact_excerpt']
    file_exists = os.path.exists(output_csv_path)

    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in data_rows:
            writer.writerow(row)

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

class Impact(BaseModel):
    impact: str = Field(
        description ="One keyword summarizing the positive or negative impact mentioned in the text."
    )
    excerpt: str = Field(
        description="Small excerpt from the text giving contect to the tag extracted for the impact. Example for the word pollution : 'la plage est polluée'"
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
    impact: List[Impact] = Field(
        description="A list containing a tag and an excerpt from the document mentioning the positive and/or negative impact of human activities"
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


# main loop to analyse each row of the csv file based on the prompt and get a json response
for file_path in files:
    start_time = time.time()
    output_file = file_path.replace('.csv', '_results')
    output_csv_path = output_file + ".csv"

    for idx, row in enumerate(pre_processing(file_path), start=1): # enumerate rows so that the idx can be printed in the exception
        id_row = row[1]
        cleaned_text = row[0]
        csv_rows = [] 
        try:
            response = client.chat.completions.create(
                model="deepseek-r1:70b",
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": row}
                ],
                response_model=MetadataExtraction,
            )
            response_json = response.model_dump_json(indent=2)               
            if response_json:
                write_result(output_file, response_json)
                for section in response.sections:
                    for impact in response.impact:
                        csv_rows.append({ # csv rows
                            'id':id_row,
                            'row':idx,
                            'file':file_path.split("/")[-1],
                            'summary' : response.summary,
                            'location': response.location,
                            'category': section.tag,
                            'keyword': section.keyword,
                            'context': section.excerpt,
                            'impact': impact.impact,
                            'impact_excerpt' : impact.excerpt,
                            })

                write_csv(output_csv_path, csv_rows) # write in the csv only if there was no exception

        except InstructorRetryException as e:
            print(f"Error on row {idx}: {row}") # print index and text of the row
            print(f"Exception message: {e}") # print exception message

            # if extraction exception, keep only id, row and file name
            csv_rows.append({
                'id': id_row,
                'row': idx,
                'file': file_path.split("/")[-1],
                'summary': '',
                'location': '',
                'category': '',
                'keyword': '',
                'context': '',
                'impact': '',
                'impact_excerpt': '',
            })

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal time for file {file_path} : {elapsed_time:.2f} seconds") # total seconds taken to load the response for each file