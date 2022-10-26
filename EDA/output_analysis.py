from transformers import AutoTokenizer
import pandas as pd
# TODO : Tokenizer를 Dataset에 의존적으로 설정.

class OutputEDA():
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.s1 = []
        self.s2 = []
        self.label = []
        self.pred = []
    
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
        output.to_csv(f'EDA/output/{epoch}.csv')
        self.s1 = []
        self.s2 = []
        self.label = []
        self.pred = []
        