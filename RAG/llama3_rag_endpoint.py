import os
import boto3
import json
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import pinecone

# Step 1: Load and Fine-Tune Model on AWS S3

def load_finetuned_model_from_s3(s3_bucket: str, model_s3_path: str, local_dir: str = './model'):
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

def fine_tune_model(model_name, dataset, s3_bucket, output_dir='./finetuned_model'):
    """
    Fine-tunes the model using the given dataset and saves the model back to S3.

    Args:
    model_name (str): The base model name.
    dataset: The dataset for fine-tuning.
    s3_bucket (str): The S3 bucket to store the fine-tuned model.
    output_dir (str): The directory to save the fine-tuned model locally.

    Returns:
    str: The S3 path to the fine-tuned model.
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fine-tuning code would be placed here, omitted for brevity.

    # Save the fine-tuned model locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Upload the fine-tuned model to S3
    s3 = boto3.client('s3')
    for file_name in os.listdir(output_dir):
        s3.upload_file(os.path.join(output_dir, file_name), s3_bucket, f'{output_dir}/{file_name}')
    
    return f"s3://{s3_bucket}/{output_dir}/"

# Step 2: Define Inference Script for SageMaker Endpoint

def create_inference_script():
    """
    Creates the inference script to be used by the SageMaker endpoint.
    """
    inference_code = '''
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import pinecone

class RAGModel:
    def __init__(self):
        # Load fine-tuned LLAMA model
        self.model = AutoModelForCausalLM.from_pretrained('/opt/ml/model')
        self.tokenizer = AutoTokenizer.from_pretrained('/opt/ml/model')
        
        # Load SentenceTransformer embeddings model
        self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Pinecone
        pinecone.init(api_key='your-pinecone-api-key', environment="us-west1-gcp")
        self.index = pinecone.Index('rag-index')

    def query_pinecone(self, query: str, top_k: int = 5):
        query_embedding = self.embeddings_model.encode([query])[0]
        results = self.index.query(query_embedding, top_k=top_k, include_metadata=True)
        return [match["metadata"]["document"] for match in results["matches"]]

    def generate_response(self, query: str, context: list):
        context_combined = " ".join(context)
        input_text = f"### Input: {context_combined} {query}\\n\\n"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict(self, query: str):
        context = self.query_pinecone(query)
        response = self.generate_response(query, context)
        return response

def lambda_handler(event, context):
    rag_model = RAGModel()
    query = event.get('query', '')
    response = rag_model.predict(query)
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response})
    }
'''
    with open('inference.py', 'w') as f:
        f.write(inference_code)

# Step 3: Deploy Model to SageMaker

def deploy_sagemaker_model(model_s3_uri, role, endpoint_name='rag-llama-endpoint'):
    """
    Deploys the fine-tuned model to a SageMaker endpoint.

    Args:
    model_s3_uri (str): The S3 URI of the fine-tuned model.
    role (str): The SageMaker execution role.
    endpoint_name (str): The name of the SageMaker endpoint.

    Returns:
    str: The name of the deployed SageMaker endpoint.
    """
    sagemaker_client = boto3.client('sagemaker')

    model = PyTorchModel(
        model_data=model_s3_uri,
        role=role,
        entry_point='inference.py',
        framework_version='1.8.0',
        py_version='py3',
        sagemaker_session=sagemaker.Session()
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name,
    )
    
    return endpoint_name

# Step 4: Invoke SageMaker Endpoint

def invoke_rag_endpoint(query, endpoint_name):
    """
    Invokes the SageMaker endpoint with a query.

    Args:
    query (str): The user's input query.
    endpoint_name (str): The name of the SageMaker endpoint.

    Returns:
    str: The generated RAG response.
    """
    runtime = boto3.client('sagemaker-runtime')
    
    payload = json.dumps({'query': query})
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    
    result = json.loads(response['Body'].read().decode())
    return result['response']

# Main Execution Function

def main():
    """
    Main function to orchestrate the entire RAG system deployment on SageMaker.

    1. Fine-tune the model using the dataset.
    2. Create the inference script.
    3. Deploy the model to SageMaker as an endpoint.
    4. Invoke the endpoint with a sample query.
    """
    # Configuration
    s3_bucket = 'your-s3-bucket-name'
    model_s3_path = 'llama-3.1-8b-instruct-finetuned'
    role = get_execution_role()
    
    # Load and fine-tune the model
    model, tokenizer = load_finetuned_model_from_s3(s3_bucket, model_s3_path)
    
    # Fine-tune model (skipped here, assuming it's already fine-tuned)
    # fine_tuned_s3_uri = fine_tune_model('llama-3.1-8b', dataset, s3_bucket)
    
    # Create the inference script
    create_inference_script()
    
    # Deploy the model to SageMaker
    endpoint_name = deploy_sagemaker_model(model_s3_uri=f's3://{s3_bucket}/{model_s3_path}', role=role)
    
    # Example Query
    query = "How does Spiderman's agility relate to momentum in physics?"
    
    # Invoke the SageMaker endpoint
    response = invoke_rag_endpoint(query, endpoint_name)
    
    print("RAG Response:", response)

# Example of how to call the main function
if __name__ == "__main__":
    main()