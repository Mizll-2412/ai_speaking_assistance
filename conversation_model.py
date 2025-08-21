# conversation_model.py
import torch
import torch.nn as nn
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

class ConversationModel(nn.Module):
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        super(ConversationModel, self).__init__()
        self.model_name = model_name
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        # Giữ lại để tương thích, nhưng bạn sẽ không dùng cho training
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate_response(self, input_text, max_length=60, num_beams=4, temperature=0.7):
        self.model.eval()
        inputs = self.tokenizer([input_text], return_tensors="pt", truncation=True)

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            reply_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return response

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        self.model = BlenderbotForConditionalGeneration.from_pretrained(load_path)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(load_path)
        print(f"Model loaded from {load_path}")
        return self


if __name__ == "__main__":
    model = ConversationModel()
    user_text = "Hello, can you help me practice English conversation?"
    response = model.generate_response(user_text)
    print(f"User: {user_text}")
    print(f"Bot: {response}")
