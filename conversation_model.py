# model.py
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class ConversationModel(nn.Module):
    def __init__(self, model_name="t5-small"):
        super(ConversationModel, self).__init__()
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate_response(self, input_text, max_length=50, num_beams=4, temperature=0.7):
        self.model.eval()
        formatted_input = f"answer: {input_text}"
        
        inputs = self.tokenizer(
            formatted_input,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"Model loaded from {load_path}")

class ModelConfig:
    def __init__(self):
        self.model_name = "t5-small"
        self.max_length = 128
        self.learning_rate = 5e-5
        self.batch_size = 16
        self.num_epochs = 3
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.save_steps = 1000
        self.eval_steps = 500

if __name__ == "__main__":
    model = ConversationModel()
    response = model.generate_response("How are you?")
    print(f"Response: {response}")