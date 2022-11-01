from transformers import AutoTokenizer
from pathlib import Path
import wandb
import pandas as pd
import os
import datetime


def get_time():
    now = str(datetime.datetime.now())
    date, time = now.split(" ")
    y, m, d = date.split("-")
    time = time.split(".")[0]
    return y[2:]+m+d+"-"+time


class OutputEDA():
    def __init__(self, model_name, file_header):
        #self.remove_lastfiles()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.reset()
        self.file_header = file_header
    
    def remove_lastfiles(self):
        for f in Path('EDA/output/').glob('*.csv'):
            try:
                f.unlink()
            except OSError as e:
                print(f"Error:{ e.strerror}")
    
    def appendf(self, label, pred, aux, s1, s2=None): 
        if s2 is None:
            for s in s1:
                ds = self.tokenizer.decode(s)  
                is1, is2, *_ = ds.split('[SEP]')
                self.s1.append(is1)
                self.s2.append(is2)
        else:
            self.s1 += [self.tokenizer.decode(s) for s in s1]
            self.s2 += [self.tokenizer.decode(s) for s in s2]
        self.label += [l.item() for l in label]
        self.pred += [p.item() for p in pred]
        self.rtt += [r[-1].item() for r in aux]
        self.source += [r[:-1].tolist().index(1) for r in aux]
            
    def reset(self):
        self.s1 = []
        self.s2 = []
        self.label = []
        self.pred = []
        self.rtt = []
        self.source = []

    def save(self, epoch, val_pearson=None):
        data = {'s1': self.s1,
                's2': self.s2,
                'label': self.label,
                'pred': self.pred,
                'rtt': self.rtt,
                'source': self.source}
        output = pd.DataFrame(data)
        if not os.path.exists('EDA/output/'):
            os.makedirs('EDA/output/')
        wandb.log({"table": output})
        time = get_time()
        filename = f'EDA/output/{self.file_header}_e{epoch}_{time}.csv'
        output.to_csv(filename)
        print("EDA saved to ", filename)
