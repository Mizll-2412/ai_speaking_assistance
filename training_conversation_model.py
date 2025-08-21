from preprocess_conversation_data import EnglishConversationDataset, build_conversation_pairs
from transformers import AutoTokenizer
from torch.optim import AdamW 
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from conversation_model import ConversationModel
import torch.nn as nn
import time

import torch



if __name__ == "__main__":
    raw_data = load_dataset("imranali291/english-conversation")['train']
    print('start')
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    epochs = 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = EnglishConversationDataset(data= raw_data, tokenizer= tokenizer, max_length= 128, train= True)
    test_dataset = EnglishConversationDataset(data= raw_data, tokenizer= tokenizer, max_length= 128, train= False)
    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=False,
        shuffle=True
    )
    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=False
    )
    print('data loaded')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConversationModel("t5-small").to(device)
    print('model loaded')
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} bắt đầu...")  # DEBUG
        epoch_start = time.time()  # DEBUG
        
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(trainloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if epoch == 0 and batch_idx == 0:
                print(f"Batch đầu: input shape {input_ids.shape}, device {input_ids.device}")

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or loss.item() > 10:
                print(f"Loss bất thường batch {batch_idx}: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                current_avg = total_loss / (batch_idx + 1)
                print(f"Batch {batch_idx}/{len(trainloader)} - Loss: {loss.item():.4f}, Avg: {current_avg:.4f}")

        avg_loss = total_loss / len(trainloader)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        print(f"Thời gian epoch: {epoch_time:.1f}s")  # DEBUG
        

        model.eval()
        with torch.no_grad():
            test_response = model.generate_response("How are you?")
            print(f"Test response: {test_response}")
        print("-" * 50)  # DEBUG
        
        save_path = "./trained_conversation_model"
        model.save_model(save_path)
        print('save success')