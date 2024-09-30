from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

file_path = "E:\\projects\\customer-service\\tts\\transcriptions.txt"

def brain():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", torch_dtype="auto")

    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as f:
        file_contents = f.read()

    # Tokenize the contents
    inputs = tokenizer(file_contents, return_tensors="pt").to("cpu")

    # Generate tokens
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.75,
        top_p=0.95,
        do_sample=True,
    )

    # Decode and print the output
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))

# Call the function
brain()
