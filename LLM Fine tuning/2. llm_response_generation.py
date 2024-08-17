import boto3
import json
import os
import tarfile
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_jsonl_file(file_path):
    """
    Loads a JSONL file and returns a list of entries.

    Args:
    file_path (str): The path to the JSONL file.

    Returns:
    list: A list of dictionaries, where each dictionary is a JSONL entry.
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def load_model_and_tokenizer(model_name, s3_bucket, s3_key):
    """
    Loads a model and tokenizer from S3.

    Args:
    model_name (str): The name of the model (for local caching and identification).
    s3_bucket (str): The S3 bucket where the model artifacts are stored.
    s3_key (str): The key in S3 where the model artifacts are located.

    Returns:
    model, tokenizer: The loaded model and tokenizer.
    """
    # Download the model artifacts from S3
    s3 = boto3.client('s3')
    local_model_dir = f"./{model_name}"
    
    # Ensure the directory exists
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Download model artifacts
    model_tar = f"{local_model_dir}/model.tar.gz"
    s3.download_file(s3_bucket, s3_key, model_tar)
    
    # Extract the model tarball
    with tarfile.open(model_tar) as tar:
        tar.extractall(path=local_model_dir)
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir)
    
    return model, tokenizer

def generate_responses(model, tokenizer, inputs):
    """
    Generates responses from a model for a list of inputs.

    Args:
    model: The loaded model.
    tokenizer: The loaded tokenizer.
    inputs (list): A list of input prompts.

    Returns:
    list: A list of model responses corresponding to the inputs.
    """
    responses = []
    for input_data in inputs:
        input_text = input_data['prompt']
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response_text)
    return responses

def save_responses(responses, output_file):
    """
    Saves the model responses to a JSONL file.

    Args:
    responses (list): A list of model responses.
    output_file (str): The path to the output JSONL file.

    Returns:
    None
    """
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(json.dumps({"response": response}) + '\n')

def process_responses(input_file_path, output_file_path):
    """
    Processes the response JSONL file to format and extract the required fields.

    Args:
    input_file_path (str): Path to the input JSONL file with raw responses.
    output_file_path (str): Path to the output JSONL file with processed responses.

    Returns:
    None
    """
    with open(output_file_path, 'w') as output_file:
        with open(input_file_path, 'r') as file:
            for line in file:
                try:
                    # Safely parse the line
                    prompt_part = line.split('"prompt":')[1].split('### Input:')[-1].strip()
                    generated_text_part = line.split('"response":')[-1].strip().strip('"')

                    # Clean the extracted parts
                    ground_truth = prompt_part.split('"')[0].strip()
                    response = generated_text_part.replace('\n', ' ').strip()

                    # Create a JSON object for each line
                    json_record = {
                        "ground_truth": ground_truth,
                        "response": response
                    }

                    # Write each JSON object to the output file as a new line
                    output_file.write(json.dumps(json_record) + "\n")
                except Exception as e:
                    continue

def main():
    """
    Main function to execute the script.

    1. Load input prompts from a JSONL file.
    2. Load models from S3.
    3. Generate responses using each model.
    4. Save the responses to separate JSONL files.
    5. Process the response files to format them and extract the required fields.
    """
    input_file = '/path/to/input_prompt_file.jsonl'  # Path to the input JSONL file
    s3_bucket = 'your-s3-bucket-name'  # Your S3 bucket name
    
    # Define models and corresponding S3 keys
    models = {
        "gpt3.5": "path/to/gpt3.5/model.tar.gz",
        "flan-t5": "path/to/flan-t5/model.tar.gz",
        "llama-3.1-8b-instruct": "path/to/llama/model.tar.gz"
    }

    # Load input data
    input_data = load_jsonl_file(input_file)
    
    for model_name, s3_key in models.items():
        # Load model and tokenizer from S3
        model, tokenizer = load_model_and_tokenizer(model_name, s3_bucket, s3_key)
        
        # Generate responses for the inputs
        responses = generate_responses(model, tokenizer, input_data)
        
        # Save responses to a JSONL file
        raw_output_file = f'./{model_name}_responses.jsonl'
        save_responses(responses, raw_output_file)
        
        # Process the response file to extract and format fields
        processed_output_file = f'./{model_name}_responses_processed.jsonl'
        process_responses(raw_output_file, processed_output_file)
        
        print(f"Processed responses saved to {processed_output_file}")

# Example of how to call the main function
if __name__ == "__main__":
    main()
