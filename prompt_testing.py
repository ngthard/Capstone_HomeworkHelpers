import fitz  # PyMuPDF
import os
import nltk
from nltk.tokenize import sent_tokenize
import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint


# Set up the LLM

# Enter secrets and necessary keys here: 


# If using an environment variable:
# load_dotenv()
#
# openai_key = os.getenv('OPENAI_API_KEY')
# access_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Model Setup
# model = ChatOpenAI()

# model = HuggingFaceEndpoint(
#     endpoint_url = 'mistralai/Mistral-7B-Instruct-v0.2',
#     max_new_tokens = 512,
#     temperature = 0.2,
#     huggingfacehub_api_token = access_token
# )

model = HuggingFaceEndpoint(
    endpoint_url = 'HuggingFaceTB/SmolLM-360M',
    max_new_tokens = 512,
    temperature = 0.2,
    huggingfacehub_api_token = access_token
)

style = "star wars"

# Prompt 1
template = """
Convert the following phrase(s) from a science textbook into a story about {style}, while perfectly preserving all the educational material the same. It should TEACH the child reading it the same knowledge as the initial phrase. 

Make sure to preserve all of the facts and pieces of knowledge within it. 

phrase: {phrase}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Prompt 2
template_2 = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction: Rewrite the following children’s science text in the style of {style}

science text: {text}
"""

prompt_2 = ChatPromptTemplate.from_template(template_2)

output_parser = StrOutputParser()

chain_2 = prompt_2 | model | output_parser

def call_llm(style, phrase):
    return chain.invoke({"style" : style, "phrase" : phrase})

def call_llm_2(style, text):
    return chain_2.invoke({'style' : style, 'text' : text})


def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Strip newline characters from each line
        lines = [line.strip() for line in lines]
    return lines

# Example usage
raw_list_of_facts = read_file_to_list('scientific_facts_output.txt')

list_of_facts = []
for fact in raw_list_of_facts:
    if fact == '':
        break
    else:
        list_of_facts.append(fact)

print(len(list_of_facts))

# Running model for prompt 1 - Uncomment below to run

# converted_phrases = []
# for fact in list_of_facts:
#     converted_phrases.append(call_llm('star wars', fact))

# Running model for prompt 2 - Uncomment below to run

converted_phrases = []
for fact in list_of_facts:
    converted_phrases.append(call_llm_2('star wars', fact))

# with open("model_output_2.txt", "w") as file:
#     for item in converted_phrases:
#         file.write(f"{item}\n")

with open("model_output_smollm.txt", "w") as file:
    for item in converted_phrases:
        file.write(f"{item}\n")