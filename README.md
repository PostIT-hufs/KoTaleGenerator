# KoTaleGenerator
발표자료는 [Uploading kotalegenerator.pdf…](이 곳)에서 확인하실 수 있습니다.


## Download Model
---
```
pip install gdown
python download_model.py
```

## Requirements
```
torch == 1.5.1
transformers == 4.9.2
streamlit == 0.87.0
streamlit_lottie == 0.0.2
```

## Data
---
- 00년도 이후의 신춘문예 동화 부문 수상 작품들을 언론사별로 수집하여 153편
- 웹 사이트에서 크롤링으로 수집한 379편
- 모두 약 5.4MB
- 이후 정제 작업을 통하여 일반적인 동화 문장의 특징을 벗어난 문장들을 제거

## Fine-tuning
---
- batch_size=2, shuffle=True, pin_memory=True 
- 학습률: 0.00005
- 학습 epoch: 약 3,000 번
- Adam 사용
## **사용법**
---
```
pip install -r requirements.txt
streamlit run flow.py
```
