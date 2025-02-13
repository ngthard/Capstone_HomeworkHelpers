# Homework Helpers Capstone Project

## Team Information
- **Team Members:** Nathan Goldhardt, Vinu Lakkur, Ashish Agnihotri
- **U-Mich Emails:** [goldhard@umich.edu](mailto:goldhard@umich.edu), [vlakkur@umich.edu](mailto:vlakkur@umich.edu), [asagniho@umich.edu](mailto:asagniho@umich.edu)
- **Team Number:** 19

## Project Overview
### Project Name: Homework Helpers

### Taglines:
- Helping Kids Learn Since 2024
- Math is Boring, Batman is Not
- Learning Legends
**- The Homework Heroes**
- Fun with Facts
- EduVenturers

### Project Description
The Homework Helpers project leverages AI to enhance educational experiences for children by personalizing learning materials. Our goal is to transform routine homework and reading into engaging, story-driven adventures tailored to each child’s interests. For example, a 3rd grader who loves Batman but dislikes math can learn through a Batman-themed math problem.

Our application utilizes Large Language Models (LLMs) and generative AI to adapt educational content into immersive, themed narratives. This innovative approach supports teachers, educators, and parents, with a particular focus on benefiting children in underserved communities. By aligning educational content with the narratives and characters that resonate with children, we aim to make learning more engaging and effective.

### Research Question
**How can Retrieval-Augmented Generation (RAG) enhance educational materials to improve engagement and learning outcomes for children?**

### Related Work
Our work is informed by and builds upon various studies and technologies in the field, including:
- **SEED-Story: Multimodal Long Story Generation with Large Language Models:** This study, which aims to generate cohesive stories and corresponding images, inspired us, particularly in maintaining the integrity of textbook translation while producing captivating outputs.
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks:** This approach, introduced by Lewis et al. (2020), combines document retrieval with text generation, which is integral to our project's architecture.

## Methodology
### Data Preparation and Collection
We sourced our data from publicly accessible online textbooks, focusing on grades 1-5 for simplicity. Initially, we extracted text using PyMuPDF and stored it in a RAG vectorstore using OpenAI embeddings. We experimented with various approaches, including using a multimodal LLM to handle visual content, but eventually narrowed our focus to text-based problems. Our primary dataset consists of 1,500 pieces of writing extracted from a 5th-grade science textbook, chosen for its simplicity and consistency.

### Exploratory Analysis
Our exploratory analysis involved multiple approaches to prompt engineering and chaining LLM inferences to produce desirable outputs. We encountered challenges such as inconsistency in LLM outputs, complexity in problem types, and the stochastic nature of LLMs. Ultimately, we standardized our focus on preserving the accuracy of content while wrapping it in a narrative style.

### Techniques and Models
We experimented with several models, including GPT-2, GPT-3.5, Mistral, Flan T-5, LLaMA 2, and LLaMA 3.1 8B Instruct. After evaluating them on metrics like cosine similarity, lexical diversity, and sentiment polarity, we selected LLaMA 3.1 8B Instruct for its performance, size, and efficiency. This model, combined with our RAG pipeline, allowed us to generate accurate, themed educational content.

### Guardrails and Ethical Considerations
We implemented multiple guardrails to ensure the content generated is appropriate for children. These include input filtering for profanity and harmful topics, output filtering, sentiment analysis, and the use of OpenAI’s Moderation API. We also incorporated methodologies to prevent bias, including diverse training data and instruction tuning.

## Minimum Viable Product (MVP)
The MVP is a functional web or mobile application that integrates an LLM with supporting models. The application allows users to input educational content and generates personalized, themed outputs aimed at engaging and educating children. The deliverables include the application codebase and a detailed project write-up.

## Evaluation Strategy
Our evaluation strategy is built on the LLM-as-a-Judge mechanism, allowing the model to self-assess the quality of its outputs. We employed a multi-phase evaluation process, incorporating meta-judging and scoring adjustments to ensure the relevance, clarity, and accuracy of the generated content. This approach helped us refine the LLM's ability to produce high-quality educational material.

## Key Findings
The LLaMA model demonstrated consistent performance in generating engaging content, with potential for further refinement through iterative improvements in evaluation and model tuning. The application of meta-rewarding principles was particularly effective in enhancing the model's self-assessment capabilities.

## Datasets
We used the following datasets to develop and refine our models:
- **Loads of Learning:** Free textbook series PDFs for grades 1-8.
- **Freekidsbooks:** Free PDFs of children’s textbooks for grades 1-8.
- **Project Gutenberg:** Thousands of free textbooks.
- **Internet Archive:** A vast collection of older but valuable textbooks.
- **Children’s Book Test:** A raw, tokenized version of many children’s books.

All datasets are publicly accessible and free of usage restrictions. They can be found in the "Data" section under "Preprocessing." 

## Required Resources
- **Cloud Computing:** Resources such as Great Lakes, Google Colab, AWS, or GCP.
- **Advising:** Guidance on Gen AI, LLMs, RAG, Langchain, and model fine-tuning.
- **Budget:** Potential funding for conducting human panel evaluations.

## Using this Github 
Our repository is broken into a few different sections. "Preprocessing" contains scripts to load, ingest, and process the data in a variety of maners. "SupservisedLearning" and "UnsupervisedLearning" both contain some very initial data anlysis to validate and determine the right data to use for our final project. "Exploratory Analysis" contains scripts we used to test out inital models, propts, and techniques on our data to start to determine the right process for our application. "Evaluation" contains to code that we used for our final evaluation technique. "RAG" congains most of the code we used to tune and implement our final model, LLaMA 3.

## Instructions for LLM Finetuning and RAG Creation

### Step 1: Text Extraction
- Use children's academic books as the source to extract text of interest. This serves as the foundation for generating educational prompts and responses.

### Step 2: Prompt Generation
- Run the `scientific_facts_prompts_generation` script. This step generates prompts in the desired format, compatible with various LLMs, tailored to the extracted text.

### Step 3: LLM Response Generation
- Execute the `llm_response_generation` script. This step uses different LLMs to generate responses based on the prompts and stores these responses in respective files.

### Step 4: LLM Finetuning
- Utilize the `llm_finetuning` script to fine-tune the 'LLaMA 3.1 8B Instruct' model. This step adapts the model to better generate the type of content needed for the educational tasks.

### Step 5: RAG Creation
- Run the `llama3_rag` script to create a Retrieval-Augmented Generation (RAG) pipeline using the fine-tuned 'LLaMA 3.1 8B Instruct' model. This allows the model to retrieve relevant content and generate responses based on the context.

### Step 6: Response Comparison
- Finally, use the `llm_response_comparison` script to compare the responses generated by the LLaMA RAG output with those from other models, such as GPT-3.5. This step helps in evaluating the performance and effectiveness of the fine-tuned model.



In order to properly use our scripts, one must install all requirements included in the requirements.txt file. They must also gather their own secrets, keys, and environment variables for LLMs and inference apis as they are personal credentials and can't be shared within the repository.  

## Acknowledgements
We extend our gratitude to Dr. O'Brien and the MADS staff for their invaluable support and guidance throughout this project.
