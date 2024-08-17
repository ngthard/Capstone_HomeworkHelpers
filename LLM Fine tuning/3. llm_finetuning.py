import boto3
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from datasets import load_dataset

def load_jsonl_dataset(file_path):
    """
    Loads a JSONL file into a Hugging Face Dataset format.

    Args:
    file_path (str): The path to the JSONL file.

    Returns:
    dataset: A Hugging Face Dataset object.
    """
    dataset = load_dataset('json', data_files={'train': file_path})
    return dataset['train']

def preprocess_function(examples, tokenizer):
    """
    Tokenizes the input and completion fields.

    Args:
    examples (dict): A dictionary containing the dataset examples.
    tokenizer: The tokenizer used for tokenizing the inputs and outputs.

    Returns:
    dict: Tokenized inputs and outputs.
    """
    inputs = examples['### Input']
    outputs = examples['completion']
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=512).input_ids

    model_inputs['labels'] = labels
    return model_inputs

def fine_tune_model(model_name, s3_bucket, dataset, tokenizer):
    """
    Fine-tunes a model using the provided dataset and tokenizer.

    Args:
    model_name (str): The name of the model.
    s3_bucket (str): The S3 bucket where the fine-tuned model will be stored.
    dataset: The dataset for fine-tuning.
    tokenizer: The tokenizer used for the model.

    Returns:
    str: The path to the fine-tuned model on S3.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./{model_name}-finetuned",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir='./logs',
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Save the fine-tuned model locally
    output_dir = f"./{model_name}-finetuned"
    trainer.save_model(output_dir)

    # Upload the fine-tuned model to S3
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            s3.upload_file(os.path.join(root, file), s3_bucket, f"{model_name}-finetuned/{file}")
    
    # Return the S3 path to the fine-tuned model
    s3_path = f"s3://{s3_bucket}/{model_name}-finetuned/"
    return s3_path

def main():
    """
    Main function to execute the fine-tuning process for each model.
    """
    jsonl_file = '/path/to/your_jsonl_file.jsonl'  # Path to your JSONL file
    s3_bucket = '/path/to/your-s3-bucket-name'  # Your S3 bucket name
    
    # Load the dataset
    dataset = load_jsonl_dataset(jsonl_file)
    
    # Define models
    models = {
    "gpt3.5": "gpt-3.5-turbo",  # OpenAI GPT-3.5 model
    "flan-t5": "google/flan-t5-large",  # Google Flan-T5 large model
    "llama-3.1-8b-instruct": "meta-llama/Llama-3-8b-hf"  # LLaMA 3 8B instruct model from Meta
    }
    
    # Fine-tune each model and save to S3
    fine_tuned_model_paths = {}
    for model_name, base_model_name in models.items():
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        s3_path = fine_tune_model(base_model_name, s3_bucket, dataset, tokenizer)
        fine_tuned_model_paths[model_name] = s3_path
    
    # Print the S3 paths to the fine-tuned models
    for model_name, path in fine_tuned_model_paths.items():
        print(f"Fine-tuned {model_name} model saved at: {path}")

# Example of how to call the main function
if __name__ == "__main__":
    main()