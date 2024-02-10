import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

parser = argparse.ArgumentParser(description='Test your custom model')
parser.add_argument('--prompt', '-p', help="enter a custom prompt", type= str)
args = parser.parse_args()

config_path = "./trained_model/config.json"
config = GPT2Config.from_json_file(config_path)

def load_model(model_directory):
    # Load the configuration from the saved configuration file
    config = GPT2Config.from_pretrained(model_directory)
    
    # Initialize the model with this configuration
    model = GPT2LMHeadModel(config)
    
    # Load the model's state dictionary
    model_state_dict = torch.load('model.pth', map_location="cpu")
    model.load_state_dict(model_state_dict)
    
    # Assuming the tokenizer is also saved in the same directory
    tokenizer = GPT2Tokenizer.from_pretrained('./model_tokenizer/')

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generate text based on the given prompt using the trained model.
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # Path to your trained model
    # model_path = '/app/trained_model'  # Path if running inside Docker
    model_path = 'trained_model'

    # Load the trained model and tokenizer
    model, tokenizer = load_model(model_path)

    # Prompt for text generation
    default_prompt = "What is your prediction for the future?"
    prompt = args.prompt or default_prompt

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt)
    print("Generated Text:\n", generated_text)

if __name__ == "__main__":
    main()
