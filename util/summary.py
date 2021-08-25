import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

class Summary:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, path="./kobart_summary"):
        self.model = BartForConditionalGeneration.from_pretrained(path)
        self.tokenizer = get_kobart_tokenizer()
        
    def summarize(self, text):
        text = text.replace('\n', '')
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)

        output = self.model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
    
        return output