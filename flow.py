import torch
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from util.model import model_loading as load
from PIL import Image
import time
import os

from util.generator import sample_sequence
from util.similarity import similarity

image = Image.open('./data/이미지/palette.png')

size=400
st.image(image,
    width =size)
st.title("Palette makes you a story")
loaded = False

@st.cache(allow_output_mutation = True)
def load_model(checkpointPath):
    return load(checkpointPath, PU = 'cpu', status = True)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    TOKENS_DICT = {
        'bos_token':'<s>',
        'eos_token':'</s>',
        'unk_token':'<unk>',
        'pad_token':'<pad>',
        'mask_token':'<mask>'
    }
    tokenizer.add_special_tokens(TOKENS_DICT)
    return tokenizer

model,_ = load_model('./modelCheckpoint/talestart1890w.tar')
st.success("모델이 다운되었습니다.")
tokenizer = load_tokenizer()
st.success("준비가 끝났습니다.")

st.markdown("## **이야기 준비**")

st.markdown("### **이야기 길이**")
selected_item = st.radio("길이를 어떻게 조정할까요?", ("전체 토큰 개수로", "문단 별 문장 수로"))

generationOption = 0
if selected_item == "전체 토큰 개수로":
    generationOption = 0
    maxsent = int(st.slider('',0.0, 500.0, 100.0))
    st.write("토큰 길이 ", maxsent,"개의 이야기가 생성됩니다.")
else:
    generationOption = 1
    col1, col2, col3 = st.columns(3)
    with col1:
        p1 = st.selectbox('첫번째 문단 문장 수',
                       ('3개', '5개', '7개'))
        p1 = int(p1[:-1])

    with col2:
        p2 = st.selectbox('두번째 문단 문장 수',
                       ('10개', '15개', '20개'))
        p2 = int(p2[:-1])
    with col3:
        p3 = st.selectbox('세번째 문단 문장 수',
                       ('3개', '5개', '7개'))
        p3 = int(p3[:-1])

basesavepath = './data/결과/'
if not os.path.exists(basesavepath):
    os.mkdir("./data/결과/")
st.markdown("### **시작 Context**")
context = st.text_input('아래 텍스트칸에 입력 후 Enter')
lottie_url = "https://assets1.lottiefiles.com/packages/lf20_5puu9o0n.json"
lottie_json = load_lottieurl(lottie_url)
col1, col2,_,_,_,_,_,_= st.columns(8)
with col1:
    bt1 = st.button("START")
with col2:
    bt2 = st.button("SHOW")
    
if bt1 :
    sysout = "이야기를 생성합니다."
    with st.spinner(sysout):
        st_lottie(lottie_json,width =200)
        vocab_path = "./data/정제/index_tale_plus_novel.txt"  
        maxtoken = [maxsent if generationOption==0 else 30*sum([p1,p2,p3])][0]
        p = sample_sequence(
            vocab=vocab_path,
            model=model, 
            length = maxtoken,
            context=context,
            num_samples=1, 
            repetition_penalty=2.0,
            top_p=0.9,
            tokenizer = tokenizer)
        p = tokenizer.decode(p[0,:].tolist())
        if generationOption== 0:
            with open("./data/결과/result001.txt",'w') as file:
                file.write(p)
            st.write("생성이 완료 되었습니다.")
 
            result= [x+'.' for x in p.split('.')]
            term = len(result)//5
            i = 0
            sims = []
            while i<=len(result)-term:
                pre = " ".join(result[i:i+term])
                post = " ".join(result[i+term:i+(term*2)])
                sim = similarity(pre,post)
                sims.append(sim)
                i = i+term
            st.write("이야기의 흐름을 검사했습니다.")
            
        else:
            p= p.split('.')
            first = ". ".join(p[:p1])
            second = ". ".join(p[p1:p1+p2])
            third = ". ".join(p[p1+p2:p1+p2+p3])
            with open("./data/결과/result001.txt",'w') as file:
                file.write(first+'\n')
                file.write(second+'\n')
                file.write(third+'\n')

elif bt2:
    try:
        with open("./data/결과/result001.txt",'r') as file:
            data = file.readlines()
            data = " ".join(data)
            data = data.replace("</s>","")
        st.write(data)
        print(data)
    except:
        st.warning("먼저 이야기를 생성해주세요.")
        

        
