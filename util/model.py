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