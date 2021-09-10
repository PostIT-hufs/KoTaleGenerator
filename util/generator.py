import torch.nn.functional as F
import torch
from tqdm import trange, tqdm
def sample_sequence(vocab, model, length, context, segments_tokens=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    device='cpu', tokenizer=None):
    taleindex=[]
    with open(vocab,'r',encoding='cp949') as file:
        taleindex=file.readlines()
        taleindex = list(map(lambda s: s.strip(), taleindex))
        taleindex = list(map(int, taleindex))
    context = tokenizer.encode(context)
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        with tqdm(total = length) as pbar:
            ind = 0
            cnt=0
            skipped = 0
            while(ind<=length):
                inputs = {'input_ids': generated}
                if segments_tokens != None:
                    inputs['token_type_ids'] = torch.tensor(segments_tokens[:generated.shape[1]]).unsqueeze(0).repeat(num_samples, 1)

                outputs = model(**inputs) # 참고: GPT-2/Transfo-XL/XLNet/CTRL(캐시된 숨겨진 상태)과 함께 '과거'를 사용할 수도 있음
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
                # CTRL의 반복 페널티(https://arxiv.org/abs/1909.05858)
                for i in range(num_samples):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= repetition_penalty

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                if temperature == 0: # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if int(next_token[0][0]) not in taleindex:
                    skipped +=1
                    print("스킵된 token >> " ,tokenizer.decode(next_token[0][0]))

                    if(cnt>3):
                        print("그냥 넘어갑니다")
                        cnt=0
                    else: 
                        cnt=cnt+1
                        continue
                generated = torch.cat((generated, next_token), dim=1)
                pbar.update(1)
                ind=ind+1
                cnt=0
    print(f"Word not in corpus => {skipped} skipped")        
    return generated



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # top-k의 마지막 토큰보다 확률이 낮은 모든 토큰을 제거
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 임계값 이상의 누적 확률을 가진 토큰 제거
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 첫 번째 토큰도 임계값보다 높게 유지하려면 인덱스를 오른쪽으로 이동
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 정렬된 텐서를 원래 인덱싱에 분산
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits