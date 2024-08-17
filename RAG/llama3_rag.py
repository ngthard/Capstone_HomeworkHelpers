import os
import boto3
import pinecone
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pinecone import Index
from typing import List, Tuple

def load_finetuned_model_from_s3(s3_bucket: str, model_s3_path: str, local_dir: str = './model') -> Tuple:
    """
    Loads the fine-tuned model and tokenizer from S3.

    Args:
    s3_bucket (str): The S3 bucket where the model artifacts are stored.
    model_s3_path (str): The path in S3 to the fine-tuned model.
    local_dir (str): The local directory to store the downloaded model files. Defaults to './model'.

    Returns:
    model, tokenizer: The loaded LLAMA model and tokenizer.
    """
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)

    # Download model files from S3
    for file_name in ["pytorch_model.bin", "config.json", "tokenizer.json"]:
        s3.download_file(s3_bucket, f"{model_s3_path}/{file_name}", os.path.join(local_dir, file_name))
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    return model, tokenizer

def create_embeddings_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Creates a SentenceTransformer model for generating embeddings.

    Args:
    model_name (str): The name of the SentenceTransformer model to use. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
    SentenceTransformer: The loaded embeddings model.
    """
    return SentenceTransformer(model_name)

def initialize_pinecone(index_name: str, dimension: int, api_key: str, environment: str = "us-west1-gcp") -> Index:
    """
    Initializes Pinecone, creates a new index, and returns the index object.

    Args:
    index_name (str): The name of the Pinecone index to create or connect to.
    dimension (int): The dimension of the embeddings.
    api_key (str): Your Pinecone API key.
    environment (str): The environment where Pinecone is hosted. Defaults to "us-west1-gcp".

    Returns:
    Index: The Pinecone index object.
    """
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=dimension)
    
    return pinecone.Index(index_name)

def index_documents(index: Index, embeddings_model: SentenceTransformer, documents: List[str]):
    """
    Indexes documents into the Pinecone vector database.

    Args:
    index (Index): The Pinecone index object.
    embeddings_model (SentenceTransformer): The embeddings model.
    documents (List[str]): A list of documents to be indexed.

    Returns:
    None
    """
    embeddings = embeddings_model.encode(documents, convert_to_tensor=False)
    metadata = [{"document": doc} for doc in documents]
    ids = [str(i) for i in range(len(documents))]
    index.upsert(vectors=zip(ids, embeddings, metadata))

def query_pinecone(index: Index, query: str, embeddings_model: SentenceTransformer, top_k: int = 5) -> List[str]:
    """
    Queries the Pinecone vector database and retrieves the most similar documents.

    Args:
    index (Index): The Pinecone index object.
    query (str): The query string.
    embeddings_model (SentenceTransformer): The embeddings model.
    top_k (int): The number of top similar documents to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of retrieved documents.
    """
    query_embedding = embeddings_model.encode([query])[0]
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return [match["metadata"]["document"] for match in results["matches"]]

def generate_response(model, tokenizer, query: str, context: List[str]) -> str:
    """
    Generates a response using the fine-tuned LLAMA model and context from the Pinecone database.

    Args:
    model: The fine-tuned LLAMA model.
    tokenizer: The tokenizer for the LLAMA model.
    query (str): The user's query.
    context (List[str]): A list of context documents retrieved from Pinecone.

    Returns:
    str: The generated response.
    """
    context_combined = " ".join(context)
    input_text = f"### Input: {context_combined} {query}\n\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """
    Main function to execute the RAG pipeline.

    1. Load the fine-tuned LLAMA model from S3.
    2. Initialize the embeddings model and Pinecone.
    3. Index the RAG documents into Pinecone.
    4. Query the Pinecone database for relevant documents.
    5. Generate a response using the LLAMA model with the retrieved context.
    """
    # Configuration
    s3_bucket = 'your-s3-bucket-name'
    model_s3_path = 'llama-3.1-8b-instruct-finetuned'
    pinecone_api_key = 'your-pinecone-api-key'
    index_name = 'rag-index'
    rag_documents = [
        "Superman can lift heavy objects. This relates to physics concepts of force and gravity.",
        "Spiderman's agility is a great example to explain momentum and acceleration.",
        # Add more documents related to science and math concepts linked to comic characters
    ]
    
    # Load the fine-tuned LLAMA model from S3
    model, tokenizer = load_finetuned_model_from_s3(s3_bucket, model_s3_path)
    
    # Create the embeddings model
    embeddings_model = create_embeddings_model()
    
    # Initialize Pinecone
    index = initialize_pinecone(index_name, dimension=embeddings_model.get_sentence_embedding_dimension(), api_key=pinecone_api_key)
    
    # Index the RAG documents
    index_documents(index, embeddings_model, rag_documents)
    
    # Example Query
    query = "How does Spiderman's agility relate to momentum in physics?"
    
    # Query Pinecone for relevant context
    context = query_pinecone(index, query, embeddings_model)
    
    # Generate response using the LLAMA model with the retrieved context
    response = generate_response(model, tokenizer, query, context)
    
    print("Generated Response:", response)

# Example of how to call the main function
if __name__ == "__main__":
    main()
