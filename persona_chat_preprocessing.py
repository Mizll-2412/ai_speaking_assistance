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
            dataset = load_dataset("bavard/personachat_truecased", split)
            conversations = []
            for i, example in enumerate(dataset):
                conversation = {
                    'conversation_id':i,
                    'personality': example['personality'],
                    'utterances': []
                }
                for j, (user_msg, bot_msg) in enumerate(zip(example['history'], example['candidates'])):
                    conversation["utterances"].append({
                        'text':user_msg,
                        'speaker': bot_msg
                    })
                    if isinstance(bot_msg, list) and len(bot_msg)>0:
                        conversation["utterances"].append({
                            'text': bot_msg[0],
                            'speaker':'bot'
                        })
                    elif isinstance(bot_msg, str):
                        conversation["utterances"].append({
                            'text':bot_msg,
                            'speaker': 'bot'
                        })
                conversations.append(conversation)
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
                    
                    # Process dialogue
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
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def filter_conversations(self, conversations, min_turns=3, max_turns=20):
        filtered = []
        for cov  in conversations:
            if 'utterances' in cov:
                turn_count = len(cov['utterances'])
                if min_turns<= turn_count <= max_turns:
                    filtered.append(cov)
        print(f"Filtered {len(filtered)} conversations from {len(conversations)} total")
        return filtered
    def extract_personas_and_dialogues(self, conversations):
        processed_data = []
        for i, conv in conversations:
            try:
                persona_info = {
                    'conversation_id': i,
                    'persona_1': conv.get('personality', []),
                    'persona_2': conv.get('partner_personality', []),
                    'dialogue': []
                }
                if 'utterances' in conv:
                    for j, utterance in enumerate(conv['utterances']):
                        turn_data = {
                            'turn_id': j,
                            'speaker': 'user' if j % 2==0 else 'bot',
                            'text': self.clean_text(utterance.get('text','')),
                            'original_text': utterance.get('text',''),
                            'word_count': len(utterance.get('text', '').split())
                        }
                        persona_info["dialogue"].append(turn_data)
                processed_data.append(persona_info)
            except Exception as e:
                print(f"Error processing conversation {i}: {e}")
                continue
        return processed_data
    def create_training_pair(self, processed_data):
        training_pairs = []
        for conv in processed_data:
            persona_1 = " ".join(conv['persona_1'])
            persona_2 = " ".join(conv['persona_2'])
            dialogue = conv['dialogue']
            for i in range(len(dialogue)-1):
                context_turns = dialogue[:i+1]
                context = f"Persona: {persona_1}\n"
                context+= "Convesation: \n"
                for turn in context_turns:
                    speaker_label = "You" if turn['speaker'] == 'user' else 'Partner'
                    context+=f"{speaker_label}: {turn['text']}\n"
                target = dialogue[i+1]['text']
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
    