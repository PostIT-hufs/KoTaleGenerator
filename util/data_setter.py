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