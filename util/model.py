import torch
from transformers import GPT2Config, GPT2LMHeadModel

def model_loading(model_path,status=False,PU='cpu') : 

    if status == False:
        print("Checkpoint에서 모델을 다운로드하지 않았습니다.")
        print("다운로드 여부 설정값 default는 False")
        return
    
    kogpt2_config = {
            "initializer_range": 0.02,
            "layer_norm_epsilon": 0.000001,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_positions": 1024,
            "vocab_size": 51200,
            "activation_function": "gelu"
    }

    checkpoint = torch.load(model_path, map_location=PU)

    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

    kogpt2model.load_state_dict(checkpoint['model_state_dict'])

    kogpt2model.train()

    kogpt2model.to(torch.device(PU))

    model = kogpt2model
    
    return model,checkpoint


from torch import nn
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        ret = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        pooler = ret[1]
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc