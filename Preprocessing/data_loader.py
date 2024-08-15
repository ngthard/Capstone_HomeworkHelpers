import fitz  # PyMuPDF
import os
import nltk
from nltk.tokenize import sent_tokenize
import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser


nltk.download('punkt')

# Secrets

# Open the PDF file
cwd = os.getcwd()
book_name = 'Freekidsbooks-5thGradeScience.pdf'
specific_folder = '\\Preprocessing\\Data\\'
pdf_document = cwd + specific_folder + book_name
doc = fitz.open(pdf_document)


# Set LLM Model
model = ChatOpenAI()
template = """
Is the following sentence a FACT or piece of knowledge about science? 

Sentence: {sentence}

If so, reply with the one word "Yes"
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

def call_llm(sentence):
    return chain.invoke({"sentence" : sentence})

# Iterate through each page and get all sentences
# all_sentences = []
# for page_number in range(doc.page_count):
#     page = doc.load_page(page_number)
#     text = page.get_text("text")  # Get the text content of the page
#
#     # Converting page text into sentences
#     sentences = sent_tokenize(text)
#     all_sentences.extend(sentences)

    # # Split text into paragraphs
    # paragraphs = text.split('\n\n')  # Assumes paragraphs are separated by double newlines
    #
    # # Iterate through each paragraph
    # for i, paragraph in enumerate(paragraphs):
    #     print(f"Page {page_number + 1}, Paragraph {i + 1}:")
    #     print(paragraph)
    #     print('-' * 80)

# Close the PDF document

# Iterate through each page and get all text
full_text = []
for page_number in range(doc.page_count):
    page = doc.load_page(page_number)
    text = page.get_text("text")  # Get the text content of the page
    full_text.append(text)

doc.close()
# print(full_text)
# Calling the first prompt

# science_facts = []
# for sentence in all_sentences:
#     if call_llm(sentence) == "Yes":
#         science_facts.append(sentence)
#
# for i, sentence in enumerate(science_facts, start = 1):
#     print(f"sentence {i}: {sentence}")


model = ChatOpenAI(model_name = 'gpt-4')
template_2 = """
Instructions: 
Please go through the text below and list the give me back the particular sentence or sentences that contain interesting scientific facts. 
Please return the words VERBATIM.
Please ONLY return the sentences and nothing else (no numbers to denote them etc.).
If you there is no text to analyze, please return "no text to analyze".

TEXT: {text}

"""

prompt_2 = ChatPromptTemplate.from_template(template_2)

output_parser = StrOutputParser()

chain_2 = prompt_2 | model | output_parser

def llm_call_2(text):
    return chain_2.invoke({'text':text})

# print(full_text)
interesting_facts = []
for page in full_text:
    interesting_facts.append(chain_2.invoke({'text':page}))

facts_string = " ".join(interesting_facts)
facts = sent_tokenize(facts_string)

with open("output3.txt", "w") as file:
    for item in facts:
        file.write(f"{item}\n")

