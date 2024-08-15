# Import required libraries

import fitz  # PyMuPDF
import re

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import human
import os
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableMap
import openai
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
import pickle
from langchain_community.document_loaders import PyPDFLoader


# Mount Google Drive to access Colab storage
# drive.mount('/content/drive')

# Directory containing PDFs

wd = os.getcwd()
pdf_directory = '\Preprocessing\Data\spelling_pb-grade-3.pdf'
absolute_path = wd + pdf_directory

# Extracting the text either directly into a string or testing other ways
# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text_data = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        text_data.append(text)
    full_text = "\n".join(text_data)
    return full_text


# Function to preprocess the extracted text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    chapters = text.split('CHAPTER')
    return chapters


# full_text = extract_text_from_pdf(r'C:\Users\submi\PycharmProjects\Capstone_HomeworkHelpers\Preprocessing\Data\spelling_pb-grade-3.pdf')

# Enter environmental variables and secrets here:


embeddings = OpenAIEmbeddings()
print(type(embeddings))

model = ChatOpenAI()
#
# def create_retriever(text):
#
#     # docs = WebBaseLoader(urls).load()
#     # loader = PyPDFLoader(pdf_path)
#     # pages = loader.load_and_split()
#     # print(type(pages))
#     text_splitter = RecursiveCharacterTextSplitter()
#     # documents = text_splitter.split_documents(docs)
#     embeddings = OpenAIEmbeddings()
#     db = FAISS.from_documents(pages, embeddings)
#     print(type(db))
#     # retriever = db.as_retriever()
#     retriever = "hello"
#     return retriever
#
# def create_retriever(pdf_path):
#
#     # docs = WebBaseLoader(urls).load()
#     loader = PyPDFLoader(pdf_path)
#     pages = loader.load_and_split()
#     print(type(pages))
#     # text_splitter = RecursiveCharacterTextSplitter()
#     # documents = text_splitter.split_documents(docs)
#     embeddings = OpenAIEmbeddings()
#     db = FAISS.from_documents(pages, embeddings)
#     print(type(db))
#     # retriever = db.as_retriever()
#     retriever = "hello"
#     return retriever

# retriever = create_retriever(pdf_directory)

# print(type(retriever))
#
# template = (
#     """
#     Please give me an outline of the children's text book below:
#
#     {textbook}
#     """
# )
#
# prompt = ChatPromptTemplate.from_template((template))
#
# # query_cr_questions = f"What are some job interview questions for a {title} at {company}?"
#
# # chain_cr_questions = RunnableMap({
# #     "context": lambda x: create_search_retriever(query_cr_questions).get_relevant_documents(
# #         f"What are some job interview questions for a {title} at {company}?"),
# #     "company": lambda x: x["company"],
# #     "title": lambda x: x["title"]
# # }) | prompt_cr_questions| model | output_parser
#
output_parser = StrOutputParser()
# chain = prompt | model | output_parser
#
# print(chain.invoke({'textbook': full_text}))

# Testing initial turning into homework problems:

problem_test = """
Words in Sentences

Write a spelling word to complete each sentence.

I had a ___________ of soup for lunch.
We bought a ___________ of bread at the store.
A penguin chick hatches and ___________ up.
Pieces of ice ___________ on top of the water.
Mark had to ___________ his dog for digging up the flowers.
The ground in Antarctica is covered in ___________.
They used to heat houses with ___________.
I brought my cat to school for ___________ and tell.
There were many ___________ necklaces in the window of the store.
The girls ___________ cookies outside the store.
On her birthday, Maggie will ___________ out the candles on her cake.
We had to ___________ the sponges in water.
Opposite

Write the spelling word that is the opposite in meaning to the word below.

fast ___________
hide ___________
sink ___________
praise ___________

Word Bank:
gold
bowl
soak
sold
snow
loaf
roast
coast
scold
coal
slow
float
show
grows
blow
"""

style = "Sharks"

template_2 = """
Transform the set of problems in the children's textbook below into problems wrapped in a story about {style}. Make sure the problems are intertwined IN the narrative.
Please preserve each homework problem as closely as possible, while maintaining the integrity of the story. Please feel free to keep the length of the output as long as necessary.
Do not SOLVE the problems, but just preserve the actual problems for the student to solve. 

{problems}
"""

prompt_2 = ChatPromptTemplate.from_template(template_2)

chain_2 = prompt_2 | model | output_parser

# narrative = chain_2.invoke({'style': style, 'problems': problem_test})
# print(narrative)


challenger_template = """
Does the narrative below have an interesting plot? Please rate the story from 1 - 10, and then give me a paragraph-long feedback/explanation on why you gave it that rating, as well as resulting improvements you can make to make the narrative stronger.
Use this format to do so: 
Story Rating: _____
Story Feedback: _____
Story Improvements: _____

Does the narrative below accurately preserve all the homework problems in the problem set below? Please rate how well it does from a 1 - 10, and then give me a paragraph-long feedback/explanation on why you gave it that rating, as well as resulting improvements you can make to improve problem preservation accuracy.
Use this format to do so: 
Accuracy Rating: _____
Accuracy Feedback: _____
Accuracy Improvements: ____


Narrative: 
{narrative}


Problem Set:
{problem_set}

"""
prompt_challenger = ChatPromptTemplate.from_template(challenger_template)
chain_challenger = prompt_challenger | model | output_parser

# challenger_feedback = chain_challenger.invoke({'narrative':narrative, 'problem_set': problem_test})

# print(chain_challenger.invoke({'narrative':narrative, 'problem_set': problem_test}))


template_feedback_incorporated = """
Please use the "Improvements" about the plot rating and preservation of problems to REWRITE the narrative, making it stronger.

Make sure to preserve all of the homework problems listed in the problem set below. Do NOT solve the problems, but instead just include the actual problems themselves, like a worksheet, so someone else may be able to solve them.

Both the feedback and narrative are included below, as well as the problem set for your reference in terms of what to preserve.

Feedback:
{feedback}

Narrative:
{narrative}

Problem Set - When constructing the narrative, make sure to include the problems outlined here intertwined into the narrative itself and DO NOT SOLVE THEM: 
{problem_set}

Lastly, before giving an answer, make SURE the problems themselves are included so someone can solve them. 

"""
#
prompt_challenger_feedback = ChatPromptTemplate.from_template(template_feedback_incorporated)
chain_challenger_feedback = prompt_challenger_feedback | model | output_parser
# challenger_feedback_incorporated = chain_challenger_feedback.invoke({'feedback' : challenger_feedback, 'narrative': narrative, 'problem_set':problem_test})
#
# print(narrative)
# print(challenger_feedback)
# print(challenger_feedback_incorporated)

#
# narrative = chain_challenger_feedback.invoke({'feedback' : challenger_feedback, 'narrative': narrative, 'problem_set':problem_test})
# challenger_feedback = chain_challenger.invoke({'narrative':narrative, 'problem_set': problem_test})
# narrative = chain_challenger_feedback.invoke({'feedback' : challenger_feedback, 'narrative': narrative, 'problem_set':problem_test})
# challenger_feedback = chain_challenger.invoke({'narrative':narrative, 'problem_set': problem_test})
#
# print(chain_challenger_feedback.invoke({'feedback' : challenger_feedback, 'narrative': narrative, 'problem_set':problem_test}))
# print(chain_challenger.invoke({'narrative':narrative, 'problem_set': problem_test}))
#


def convert_homework(style, problems):

    narrative = chain_2.invoke({'style': style, 'problems': problems})
    # challenger_feedback = chain_challenger.invoke({'narrative':narrative, 'problem_set': problems})
    # challenger_feedback_incorporated = chain_challenger_feedback.invoke({'feedback' : challenger_feedback, 'narrative': narrative, 'problem_set':problems})

    return narrative
