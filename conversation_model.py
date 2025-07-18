import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import(
    AutoTokenizer, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    AdamW, get_linear_schedule_with_warmup
)

import json
import numpy as np
from tqdm import tqdm
import os
import wandb
from datetime import datetime

class ConversationDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        special_tokens = {
            'pad_token': '<pad>',
            'eos_token': '<eos>',
            'bos_token': '<bos>',
            'sep_token': '<sep>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        print(f"loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        target = item['target']

        input_text = f"{self.tokenizer.bos_token}{context}{self.tokenizer.sep_token}{target}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            input_text,
            truncation = True,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors = 'pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_token_id).nonzeros(as_tuple=True)[0]
        if len(sep_positions)>0:
            sep_pos = sep_positions[0]
            labels[:sep_pos+1] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels':labels,
            'context': context,
            'target': target
        }
class ConversationTrainer:
    def __init__(self, model_name = 'gpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        print(f"loading {model_name} model") 
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.train_dataset=None
        self.val_dataset=None

        print(f"run in {device}")
    def prepare_data(self, train_file, val_file, max_length=512):
        print('prepare')
        self.train_dataset = ConversationDataset(train_file, self.tokenizer, max_length)
        self.val_dataset = ConversationDataset(val_file, self.tokenizer, max_length)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"Training examples: {len(self.train_dataset)}")
        print(f"Validation examples: {len(self.val_dataset)}")
    
    def train(self, output_dir = './conversation_model', epochs=3, batch_size=4, learning_rate=5e-5,
              warmup_steps = 100, save_steps = 500, eval_steps = 500, gradient_accumulation_steps = 1, use_wandb = False):
        os.makedirs(output_dir, exist_ok=True)
        if use_wandb:
            wandb.init(
                project="conversation-model",
                config={
                    "model_name": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_length":self.train_dataset.max_length
                }
            )
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader)* epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print("Start training")
        global_step = 0
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")
            for step, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                total_train_loss += loss.item()
                if (step+1)% gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                train_pbar.set_postfix({
                    'loss': loss.item()* gradient_accumulation_steps,
                    'lr': scheduler.get_last_lr()[0]
                })
                if use_wandb and global_step %10 == 0: 
                    wandb.log({
                        'train_loss': loss.item()*gradient_accumulation_steps,
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch': epoch
                    })
                if global_step % eval_steps ==0:
                    val_loss = self.evaluate(val_loader)
                    print(f"\nValidation loss: {val_loss}")
                    if use_wandb: 
                        wandb.log({'val_loss': val_loss})
                    if val_loss < best_val_loss:
                        best_val_loss=val_loss
                        self.save_model(os.path.join(output_dir, 'best_model'))
                        print('Save best model')
                if global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
                    self.save_model(checkpoint_dir)
                    print(f"save checkpoint: {checkpoint_dir}")
            avg_train_loss  = total_train_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if use_wandb:
                wandb.log({
                    'epoch_train_loss': avg_train_loss,
                    'epoch_val_loss': val_loss,
                    'epoch': epoch
                })
        final_model_dir = os.path.join(output_dir, 'final_model')
        self.save_model(final_model_dir)
        print(f"training complete, model save to {final_model_dir}")
        if use_wandb:
            wandb.finish()
    def evaluate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
        self.model.train()
        return total_val_loss / len(val_loader)
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    def generate_response(self, context, max_length=100, temperature=0.7, top_p=0.9):
        self.model.eval()
        input_text = f"{self.tokenizer.bos_token}{context}{self.tokenizer.sep_token}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)  
                    # Extract response (after separator)
        if self.tokenizer.sep_token in generated_text:
            response = generated_text.split(self.tokenizer.sep_token)[-1].strip()
        else:
            response = generated_text
        
        return response

def main():
    trainer =ConversationTrainer(model_name='gpt2')
    trainer.prepare_data(
        train_file='conversation_data/train_data.json',
        val_file='conversation_data/val_data.json',
        max_length=512
    )
    trainer.train(
        output_dir='./conversation_model',
        epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        warmup_steps=100,
        save_steps=500,
        eval_steps=500,
        use_wandb=False 
    )
if __name__ == "__main__":
    main()