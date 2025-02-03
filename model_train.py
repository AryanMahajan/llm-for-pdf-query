from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments

from dataset_load import load_dataset, load_data_collator

def train(train_file_path,model_name,output_dir,overwrite_output_dir,per_device_batch_size,num_of_epochs):
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path,tokenizer)
    data_collator = load_data_collator(tokenizer)
    
    tokenizer.save_pretrained(output_dir)
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    model.save_pretrained(output_dir)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=num_of_epochs,
        per_device_train_batch_size=per_device_batch_size,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    
train_file_path = "QA_data.txt"
model_name = "gpt2"
output_dir = "output"
overwrite_output_dir = True
per_device_batch_size = 16
num_of_epochs = 50.0


train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_batch_size=per_device_batch_size,
    num_of_epochs=num_of_epochs
)