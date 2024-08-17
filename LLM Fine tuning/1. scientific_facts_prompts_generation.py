import json

def extract_texts_from_file(file_path, num_texts=61):
    """
    Extracts the first `num_texts` enclosed in quotes from the file.

    Args:
    file_path (str): The path to the text file.
    num_texts (int): The number of quoted texts to extract. Default is 61.

    Returns:
    list: A list of extracted quoted texts.
    """
    texts = []
    with open(file_path, 'r') as file:
        content = file.read()
        quotes = content.split('"')
        # Extracting every second element from the split list
        for i in range(1, len(quotes), 2):
            if len(texts) < num_texts:
                texts.append(quotes[i])
            else:
                break
    return texts

def create_jsonl_from_texts(texts, output_file):
    """
    Creates a JSONL file with structured data using a given template.

    Args:
    texts (list): A list of texts to be included in the JSONL file.
    output_file (str): The path to the output JSONL file.

    Returns:
    None
    """
    template = {
        "prompt": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{INPUT}\n\n"
        )
    }

    with open(output_file, 'w') as jsonl_file:
        for text in texts:
            entry = {
                "prompt": template["prompt"].replace("{INPUT}", text).replace("{instruction}", "Provide a detailed analysis of the input.")
            }
            jsonl_file.write(json.dumps(entry) + '\n')

def main():
    """
    Main function to execute the script.
    
    Usage:
    1. Extract texts from the provided file.
    2. Create a JSONL file using the extracted texts.
    """
    input_file = '/path/to/scientific_facts_output.txt'  # Path to the input file
    output_file = '/path/to/input_prompt_file.jsonl'  # Path to the output JSONL file
    
    # Step 1: Extract the first 61 quoted texts from the file
    texts = extract_texts_from_file(input_file, num_texts=61)
    
    # Step 2: Create a JSONL file with the extracted texts using the template
    create_jsonl_from_texts(texts, output_file)
    
    print(f"JSONL file created at {output_file}")

# Example of how to call the main function
if __name__ == "__main__":
    main()
