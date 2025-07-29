# data_loader.py
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

class ConversationDataLoader:
    def __init__(self, dataset_name="imranali291/english-conversation", test_size=0.2, random_state=42):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        
    def load_and_split_data(self):
        """Load dataset và chia thành train/test"""
        print("Loading dataset...")
        raw_data = load_dataset(self.dataset_name)['train']
        
        # Convert to list for easier handling
        data_list = [item for item in raw_data]
        
        # Split data
        train_data, test_data = train_test_split(
            data_list, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"Total conversations: {len(data_list)}")
        print(f"Train conversations: {len(train_data)}")
        print(f"Test conversations: {len(test_data)}")
        
        return train_data, test_data
    
    def save_data(self, train_data, test_data, train_path="train_data.json", test_path="test_data.json"):
        """Lưu data ra file JSON"""
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"Data saved to {train_path} and {test_path}")

if __name__ == "__main__":
    loader = ConversationDataLoader()
    train_data, test_data = loader.load_and_split_data()
    loader.save_data(train_data, test_data)