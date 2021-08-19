from torch.utils.data import Dataset
class Data_Set(Dataset):

  def __init__ (self, file_path,tokenizer):
    self.data = []
    self.tokenizer = tokenizer

    f = open(file_path,'r',encoding='cp949')
    file = f.read()
    file = file.split('\n')

    dataset = []
    now = ''

    for i, line in enumerate(file):
      if i % 30 == 0 and i != 0:
        dataset.append(now)
        now = ''

      now = now + '\n' + line

    for line in dataset:
      indexing_word = tokenizer.bos_token+ line + tokenizer.eos_token
      indexing_word = tokenizer.encode(indexing_word)
      self.data.append(indexing_word)

    f.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]

from random import randint
def make_trainset(filepath):
    lines = []
    with open (filepath, 'r', encoding= 'cp949') as txt:
        for line in txt:
            line = line[:-1]
            if '"' not in line:
                for tmp in line.split('.'):
                    if tmp!='' and tmp!=' ':
                        lines.append(tmp+'.')
            else:
                lines.append(line)
    pre = []
    post = []
    label = []
    for i in range(len(lines)-1):
        if lines[i] != 'end of file.':
            pre.append(lines[i])
            post.append(lines[i+1])
            label.append(1)
            while True:
                randomnum = randint(1,len(lines)-1)
                if randomnum not in [i, i+1]:
                    break
            pre.append(lines[i])
            post.append(lines[randomnum])
            label.append(0)
    datalist = []
    for pres, posts, lab in zip(pre,post,label):
        datalist.append([pres,posts, lab])
    return datalist

from tqdm import tqdm, tqdm_notebook, trange
import gluonnlp as nlp
import numpy as np

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx1, sent_idx2, label_idx, bert_tokenizer, max_len,
                 pad, pair, pre_max_len):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=pre_max_len, pad=pad, pair=pair)

        self.sentences = [pad_transform(transform, d[sent_idx1], d[sent_idx2], pre_max_len, max_len) for d in tqdm(dataset)]
#         self.sentences = [transform([d[sent_idx1], d[sent_idx2]]) for d in tqdm(dataset)]
        self.labels = [np.int32(d[label_idx]) for d in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


def pad_transform(transform, sent1, sent2, pre_max_len, max_len):
    indices = np.indices([pre_max_len])[0]
    tokens, valid_len, segs = transform([sent1, sent2])
    if valid_len < max_len:
        return tokens[:max_len], np.array((tokens != 1).sum(), dtype='int32'), segs[:max_len]
    
    idx2 = indices[segs == 1][:-1]
    idx1 = indices[:idx2[0]][1:-1]
    if len(idx1) < len(idx2) and (len(idx1) * 2 + 3) < max_len:
        idx2 = idx2[:(max_len - 3 - len(idx1))]
    elif len(idx2) < len(idx1) and (len(idx2) * 2 + 3) < max_len:
        idx1 = idx1[-(max_len - 3 - len(idx2)):]
    else:
        if len(idx1) < len(idx2):
            idx1, idx2 = idx1[-((max_len - 3) // 2):], idx2[:((max_len - 3) // 2 + 1)]
        else:
            idx1, idx2 = idx1[-((max_len - 3) // 2 + 1):], idx2[:((max_len - 3) // 2)]
    return (np.concatenate([[2], tokens[idx1], [3], tokens[idx2],[3]]), 
             np.array(max_len, dtype='int32'), 
             np.array([0] * (len(idx1) + 2) + [1] * (len(idx2) + 1)))