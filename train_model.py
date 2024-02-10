import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd

output_dir='./trained_model'

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Force PyTorch to use CPU
device = torch.device("cpu")
model.to(device)

# Load your dataset from CSV
csv_path = "dataset.csv"
df = pd.read_csv(csv_path)
texts = df["prompt"].tolist()  # assuming "prompt" is the column containing your text

# Tokenize the texts
tokenized_texts = tokenizer(
  texts,
  return_tensors="pt",
  padding=True,
  truncation=True,
  max_length=128,
)

class CustomDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {
            key: value[idx] for key, value in self.tokenized_texts.items()
        }

# Create CustomDataset
dataset = CustomDataset(tokenized_texts)

# # Create TextDataset
# dataset = TextDataset(
#   tokenized_texts,
#   tokenizer,
#   block_size=128
# )

# Set up data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    # output_dir=
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the trained model
# model.save_pretrained('/app/trained-model')
# tokenizer.save_pretrained('/app/trained-model')

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

torch.save(model.state_dict(), 'model.pth')
tokenizer.save_pretrained('./model_tokenizer/')
