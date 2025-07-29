from preprocess_conversation_data import EnglishConversationDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from conversation_model import ConversationModel


if __name__ == "__main__":
    raw_data = load_dataset("imranali291/english-conversation")['train']
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainloader = EnglishConversationDataset(raw_data, tokenizer, max_length=128, max_context_turns=4, train=True)
    testloader = EnglishConversationDataset(raw_data, tokenizer, max_length=128, max_context_turns=4, train=False)

    trainloader = DataLoader(
        dataset=trainloader,
        batch_size=16,
        num_workers=4,
        drop_last=False,
        shuffle=True
    )
    testloader = DataLoader(
        dataset=testloader,
        batch_size=16,
        num_workers=4,
        drop_last=False
    )
    model = ConversationModel("t5-small")
    
