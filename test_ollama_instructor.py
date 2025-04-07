"""
Adapted from https://python.useinstructor.com/examples/ollama/
Example usage: python test_ollama_instructor.py "Rosa Parks"
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from argparse import ArgumentParser

import instructor

ap = ArgumentParser()
ap.add_argument("name", type=str, help="The full name of a person.")
args = ap.parse_args()
question = f"Who is {args.name}?"
print(question)

class Character(BaseModel):
    name: str
    age: int
    gender: str
    fact: List[str] = Field(..., description="A list of facts about the person")


# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:5005/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

resp = client.chat.completions.create(
    model="deepseek-r1:70b",
    messages=[
        {
            "role": "user",
            f"content": question,
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))