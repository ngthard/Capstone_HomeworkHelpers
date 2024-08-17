# Import required libraries

import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
from langchain_huggingface import HuggingFaceEndpoint

# Not Currently used Libraries but likely to be used in the future:


# CODE BELOW TO IMPORT PAGES FROM PDFs // COMMENT OUT IF NOT USING

# wd = os.getcwd()
# pdf_directory = '\Preprocessing\Data\spelling_pb-grade-3.pdf'
# absolute_path = wd + pdf_directory

# Extracting the text either directly into a string or testing other ways
# Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_path):
#     pdf_document = fitz.open(pdf_path)
#     text_data = []
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         text = page.get_text()
#         text_data.append(text)
#     full_text = "\n".join(text_data)
#     return full_text


# Function to preprocess the extracted text
# def preprocess_text(text):
#     text = re.sub(r'\s+', ' ', text)
#     chapters = text.split('CHAPTER')
#     return chapters






# LOADING NECESSARY ENVIRONMENT VARIABLES
# load_dotenv()

# openai_key = os.getenv('OPENAI_API_KEY')
# access_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')



# CREATING MODELS OBJECTS WE WANT TO UTILIZE // ENTERING THE MODEL IN THE DICTIONARY AT THE BOTTOM OF THIS SECTION
# WILL ENSURE THAT IT IS RUN AND HAS AN OUTPUT

model_gpt = ChatOpenAI()

model_mistral = HuggingFaceEndpoint(
    endpoint_url = 'mistralai/Mistral-7B-Instruct-v0.2',
    max_new_tokens = 512,
    temperature = 0.2,
    huggingfacehub_api_token = access_token
)

model_smollm = HuggingFaceEndpoint(
    endpoint_url = 'HuggingFaceTB/SmolLM-360M',
    max_new_tokens = 512,
    temperature = 0.2,
    huggingfacehub_api_token = access_token
)

model_dic = {
    'model_smollm' : model_smollm,
    'model_mistral' : model_mistral,
    'model_gpt' :model_gpt
}


# NECESSARY FUNCTIONS THAT INVOKE THE PROPER CHAINS. THEY ARE USED IN THE FOR LOOP BELOW AS WELL AS IN THE STREAMLIT APPLICATION

def convert_homework_simple(style, problems):

    narrative = chain_conversion.invoke({'style': style, 'problems': problems})

    return narrative

def convert_homework_revisions(style, problems):

    narrative = chain_conversion.invoke({'style': style, 'problems': problems})
    challenger_feedback = chain_challenger.invoke({'narrative':narrative, 'problem_set': problems})
    challenger_feedback_incorporated = chain_challenger_feedback.invoke({'feedback' : challenger_feedback, 'narrative': narrative, 'problem_set':problems})

    return challenger_feedback_incorporated





# FOR LOOP THAT RUNS THE HOMEWORK CONVERSION THROUGH EVERY MODEL WE'VE SET IN THE DICTIONARY ABOVE
# CHANGING THE TEMPLATES, STYLE, PROMPTS, AS WELL AS THE CONVERSION FUNCTION WILL RESULT IN DIFFERENT OUTPUTS

for repo in model_dic:
    model = model_dic[repo]

    #Setting the output parser

    output_parser = StrOutputParser()


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

    template_conversion = """
    Transform the set of problems in the children's textbook below into problems wrapped in a story about {style}. 
    
    Make sure the problems are intertwined IN the narrative. Present the problems as problems that need to be solved. 
    
    Please preserve each homework problem as closely as possible, while maintaining the integrity of the story. 
    Please free to keep the length of the output as long as necessary.
    Do not SOLVE the problems, but just preserve the actual problems for the student to solve. 
    
    {problems}
    """

    prompt_conversion = ChatPromptTemplate.from_template(template_conversion)

    chain_conversion = prompt_conversion | model | output_parser



    template_challenger = """
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
    prompt_challenger = ChatPromptTemplate.from_template(template_challenger)
    chain_challenger = prompt_challenger | model | output_parser




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

    print(convert_homework_simple(style, problem_test))

