import json
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from collections import Counter
from nltk.corpus import stopwords
from datasets import load_dataset

# nltk.download('punkt')
# nltk.download('stopwords')

class PersonaChatPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.contractions = {
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "won't": "will not", "wouldn't": "would not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not",
            "mustn't": "must not", "needn't": "need not"
        }
    
    def load_personachat_data(self, split='train'):
        try:
            dataset = load_dataset("bavard/personachat_truecased", name='full', split=split)
            conversations = []
            
            for i, example in enumerate(dataset):
                conversation = {
                    'conversation_id': i,
                    'personality': example['personality'],
                    'utterances': []
                }                
                history = example['history']                
                for j, utterance in enumerate(history):
                    speaker = 'user' if j % 2 == 0 else 'bot'
                    conversation["utterances"].append({
                        'text': utterance,
                        'speaker': speaker
                    })
                candidates = example['candidates']
                if candidates and len(candidates) > 0:
                    conversation["utterances"].append({
                        'text': candidates[0],
                        'speaker': 'bot' if len(history) % 2 == 0 else 'user'
                    })
                conversations.append(conversation)
            print(f"Successfully loaded {len(conversations)} conversations from bavard/personachat_truecased")
            return conversations
        except Exception as e:
            print(f"Error loading data from Hugging Face: {e}")
            try:
                dataset = load_dataset("convai-challenge/conv_ai_2", split=split)
                conversations = []
                for i, example in enumerate(dataset):
                    conversation = {
                        'conversation_id': i,
                        'personality': example.get('personality', []),
                        'utterances': []
                    }
                    if 'dialog' in example:
                        for j, utterance in enumerate(example['dialog']):
                            conversation['utterances'].append({
                                'text': utterance,
                                'speaker': 'user' if j % 2 == 0 else 'bot'
                            })
                    
                    conversations.append(conversation)
                
                print(f"Successfully loaded {len(conversations)} conversations from ConvAI2")
                return conversations
                
            except Exception as e2:
                print(f"Error loading ConvAI2 data: {e2}")
                return None
    
    def clean_text(self, text):
        text = text.lower()
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def filter_conversations(self, conversations, min_turns=3, max_turns=20):
        filtered = []
        for conv in conversations: 
            if 'utterances' in conv:
                turn_count = len(conv['utterances'])
                if min_turns <= turn_count <= max_turns:
                    filtered.append(conv)
        print(f"Filtered {len(filtered)} conversations from {len(conversations)} total")
        return filtered
    
    def extract_personas_and_dialogues(self, conversations):
        processed_data = []
        for conv in conversations: 
            try:
                persona_info = {
                    'conversation_id': conv['conversation_id'],
                    'persona_1': conv.get('personality', []),
                    'persona_2': conv.get('partner_personality', []),
                    'dialogue': []
                }
                
                if 'utterances' in conv:
                    for j, utterance in enumerate(conv['utterances']):
                        turn_data = {
                            'turn_id': j,
                            'speaker': utterance.get('speaker', 'user' if j % 2 == 0 else 'bot'),
                            'text': self.clean_text(utterance.get('text', '')),
                            'original_text': utterance.get('text', ''),
                            'word_count': len(utterance.get('text', '').split())
                        }
                        persona_info["dialogue"].append(turn_data)
                
                processed_data.append(persona_info)
            except Exception as e:
                print(f"Error processing conversation {conv.get('conversation_id', 'unknown')}: {e}")
                continue
        
        return processed_data
    
    def create_training_pairs(self, processed_data): 
        training_pairs = []
        for conv in processed_data:
            persona_1 = " ".join(conv['persona_1'])
            persona_2 = " ".join(conv['persona_2'])
            dialogue = conv['dialogue']
            for i in range(len(dialogue) - 1):
                context_turns = dialogue[:i + 1]
                context = f"Persona: {persona_1}\n"
                context += "Conversation:\n"
                for turn in context_turns:
                    speaker_label = "You" if turn['speaker'] == 'user' else 'Partner'
                    context += f"{speaker_label}: {turn['text']}\n"
                target = dialogue[i + 1]['text']
                training_pair = {
                    'conversation_id': conv['conversation_id'],
                    'turn_id': i,
                    'context': context.strip(),
                    'target': target,
                    'persona': persona_1,
                    'input_length': len(context.split()),
                    'target_length': len(target.split())
                }
                training_pairs.append(training_pair)
        
        return training_pairs
    
    def analyze_data_statistics(self, training_pairs):
        if not training_pairs:
            print("No training pairs to analyze!")
            return {}
            
        stats = {
            'total_pairs': len(training_pairs),
            'avg_input_length': np.mean([pair['input_length'] for pair in training_pairs]),
            'avg_target_length': np.mean([pair['target_length'] for pair in training_pairs]),
            'max_input_length': max([pair['input_length'] for pair in training_pairs]),
            'max_target_length': max([pair['target_length'] for pair in training_pairs]),
            'unique_conversations': len(set([pair['conversation_id'] for pair in training_pairs]))
        }
        print("Dataset Statistics:")
        print(f"Total training pairs: {stats['total_pairs']}")
        print(f"Average input length: {stats['avg_input_length']:.2f} words")
        print(f"Average target length: {stats['avg_target_length']:.2f} words")
        print(f"Max input length: {stats['max_input_length']} words")
        print(f"Max target length: {stats['max_target_length']} words")
        print(f"Unique conversations: {stats['unique_conversations']}")
        
        return stats
    
    def save_processed_data(self, training_pairs, output_file):
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_pairs, f, indent=2, ensure_ascii=False)
            print(f"Processed data saved to {output_file}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def create_train_val_split(self, training_pairs, val_ratio=0.05, test_ratio=0.1):
        np.random.shuffle(training_pairs)
        total_size = len(training_pairs)
        val_size = int(total_size * val_ratio)
        test_size = int(total_size * test_ratio)
        train_size = total_size - val_size - test_size

        train_data = training_pairs[:train_size]        
        val_data = training_pairs[train_size:train_size + val_size]
        test_data = training_pairs[train_size + val_size:]

        print("Data split:")
        print(f"Train: {len(train_data)}")
        print(f"Val: {len(val_data)}")
        print(f"Test: {len(test_data)}")
        
        return train_data, val_data, test_data

def main():
    preprocessor = PersonaChatPreprocessor()
    print("1. Loading Data...")
    raw_data = preprocessor.load_personachat_data(split='train')
    if raw_data:
        print(f"Loaded {len(raw_data)} conversations")
        print("\n2. Filtering conversations...")
        filtered_conversations = preprocessor.filter_conversations(raw_data)
        print("\n3. Extracting personas and dialogues...")
        processed_data = preprocessor.extract_personas_and_dialogues(filtered_conversations)
        print("\n4. Creating training pairs...")
        training_pairs = preprocessor.create_training_pairs(processed_data)
        print("\n5. Analyzing data statistics...")
        stats = preprocessor.analyze_data_statistics(training_pairs)
        print("\n6. Creating data split...")
        train_data, val_data, test_data = preprocessor.create_train_val_split(training_pairs)
        print("\n7. Saving processed data...")
        preprocessor.save_processed_data(train_data, 'conversation_data/train_data.json')
        preprocessor.save_processed_data(val_data, 'conversation_data/val_data.json')
        preprocessor.save_processed_data(test_data, 'conversation_data/test_data.json')
        
        print("\nPreprocessing completed successfully!")
        
        if train_data:
            print("\nSample training pair:")
            print(f"Context: {train_data[0]['context']}")
            print(f"Target: {train_data[0]['target']}")
    else:
        print("Failed to load data. Please check your internet connection or try local file loading.")

if __name__ == "__main__":
    main()