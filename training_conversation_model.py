from preprocess_conversation_data import EnglishConversationDataset, build_conversation_pairs
from transformers import AutoTokenizer
from torch.optim import AdamW 
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from conversation_model import ConversationModel
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    raw_data = load_dataset("imranali291/english-conversation")['train']
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    epochs = 3
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConversationModel("t5-small").to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in trainloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    
    # response = model.generate_response("how are you?")
    # print(response)



                
            
