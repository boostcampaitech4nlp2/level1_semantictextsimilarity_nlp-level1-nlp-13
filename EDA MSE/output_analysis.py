from transformers import AutoTokenizer
from pathlib import Path
import pandas as pd
# TODO : Tokenizer를 Dataset에 의존적으로 설정.


class OutputEDA():
    def __init__(self, model_name, file_header):
        self.remove_lastfiles()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.s1 = []
        self.s2 = []
        self.label = []
        self.pred = []
        self.file_header = file_header
    
    def remove_lastfiles(self):
        for f in Path('EDA/output').glob('*.csv'):
            try:
                f.unlink()
            except OSError as e:
                print(f"Error:{ e.strerror}")
    
    def append(self, s1, s2, label, pred): 
        self.s1 += [self.tokenizer.decode(s) for s in s1]
        self.s2 += [self.tokenizer.decode(s) for s in s2]
        self.label += [l.item() for l in label]
        self.pred += [p.item() for p in pred]
        assert len(self.s1) == len(self.s2)
        assert len(self.label) == len(self.s1)
        assert len(self.pred) == len(self.label)     
        
    def getEDA(self, epoch):
        data = {'s1': self.s1,
                's2': self.s2,
                'label': self.label,
                'pred': self.pred}
        output = pd.DataFrame(data)
        output.to_csv(f'EDA/output/{self.file_header}_{epoch}.csv')
        self.s1 = []
        self.s2 = []
        self.label = []
        self.pred = []