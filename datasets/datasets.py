from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from collections import defaultdict
from typing import List, Tuple
import random


class KorSTSDatasets(Dataset):
    def __init__(self, dir, model_name, stopword=False):
        super(KorSTSDatasets, self).__init__()
        self.tsv = pd.read_csv(dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.s1 = [self.tokenizer.encode(s1) for s1 in self.tsv["sentence_1"]]
        self.s2 = [self.tokenizer.encode(s2) for s2 in self.tsv["sentence_2"]]
        if "label" in self.tsv.keys():
            self.y = self.tsv["label"]
        else:
            self.y = None
        self.rtt, self.source = self.set_eda()

        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.mask_id = self.tokenizer.mask_token_id

        #read stopwords
        if stopword:
            stopwords = []
            f = open('./stopwords_ver2.txt')
            lines = f.readlines()
            for line in lines:
                if '\n' in line:
                    stopwords.append(line[:-1])

            s1s = []
            s2s = []

            for s1 in tsv["sentence_1"]:
                sentence_tokens = []
                for word in tokenizer.tokenize(s1):
                    tmp_word = word
                    if "##" in word:
                        tmp_word = word.replace('##', '')
                    if tmp_word not in stopwords:
                        sentence_tokens.append(word)
                s1s.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence_tokens)))
            for s2 in tsv["sentence_2"]:
                sentence_tokens = []
                for word in tokenizer.tokenize(s2):
                    tmp_word = word
                    if "##" in word:
                        tmp_word = word.replace('##', '')
                    if tmp_word not in stopwords:
                        sentence_tokens.append(word)
                s2s.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence_tokens)))
        #read stopwords end
            self.s1 = [tokenizer.encode(s1) for s1 in s1s]
            self.s2 = [tokenizer.encode(s2) for s2 in s2s]

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        data = torch.IntTensor(self.s1[idx]), torch.IntTensor(self.s2[idx])
        if "label" in self.tsv.keys():
            label = float(self.y[idx])/5
        else:
            label = None
            
        aux = self.get_aux(idx)
        
        return data, label, aux
    
    def set_eda(self):
        rtt_filter = self.tsv['source'].str.contains('rtt') 
        rtt = pd.Series([0] * len(self.tsv))
        rtt[rtt_filter] = 1
        
        nsmc_filter = self.tsv['source'].str.contains('nsmc')    
        petition_filter = self.tsv['source'].str.contains('petition') 
        slack_filter = self.tsv['source'].str.contains('slack') 
        source = pd.Series([0] * len(self.tsv))
        source[nsmc_filter] = 0
        source[petition_filter] = 1
        source[slack_filter] = 2
        
        return rtt, source
    
    def get_aux(self, idx):
        rtt = torch.tensor([self.rtt[idx]], dtype=torch.long)
        source = torch.tensor(self.source[idx], dtype=torch.long)
        source = torch.nn.functional.one_hot(source, 3)
        aux = torch.cat([source, rtt])
        return aux
        

class KorSTSDatasets_for_BERT(KorSTSDatasets):
    def __init__(self, dir, model_name, stopword):
        super(KorSTSDatasets_for_BERT, self).__init__(dir, model_name, stopword)

    def __getitem__(self, idx):
        data = self.s1[idx][:-1] + [self.sep_id] + self.s2[idx][1:]
        data = torch.IntTensor(data)
        if "label" in self.tsv.keys():
            label = float(self.y[idx])/5
        else:
            label = None
            
        aux = self.get_aux(idx)

        return data, label, aux

class KorNLIDatasets(KorSTSDatasets):
    def __init__(self, dir, model_name):
        super(KorNLIDatasets, self).__init__(dir, model_name)

    def __getitem__(self, idx):
        data = self.s1[idx][:-1] + [self.sep_id] + self.s2[idx][1:]
        data = torch.IntTensor(data)
        if "label" in self.tsv.keys():
            label = 1 if self.y[idx] > 0.5 else 0
        else:
            label = None
            
        aux = self.get_aux(idx)

        return data, label, aux

class KorSTSDatasets_for_MLM(KorSTSDatasets):
    def __init__(self, dir, model_name):
        super(KorSTSDatasets_for_MLM, self).__init__(dir, model_name)
        self.sentences = self.s1 + self.s2

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        sentence, label = self.masking(sentence)
        
        return sentence, label
        
    def masking(self, sentence):
        label = torch.zeros(len(sentence)).int()
        for i, s in enumerate(sentence):
            masking = random.random()
            if masking < 0.15:
                label[i] = sentence[i]
                masking /= 0.15
                if masking < 0.8:
                    sentence[i] = self.mask_id
                elif masking < 0.9:
                    sentence[i] = random.randrange(self.tokenizer.vocab_size)
                else:
                    sentence[i] = sentence[i]
        sentence = torch.IntTensor(sentence)
        label = torch.IntTensor(label)

        return sentence, label

class Collate_fn(object):
    def __init__(self, pad_id=0, model_type="SBERT"):
        self.pad_id = pad_id
        self.model_type = model_type
    
    def __call__(self, batch):
        # batch = list([((s1, s2), label), ((s1, s2), label), ...])
        if self.model_type == "SBERT":
            s1_batches = []
            s2_batches = []
            labels = []
            auxes = []
            for b in batch:
                data, label, aux = b
                s1, s2 = data
                s1_batches.append(s1)
                s2_batches.append(s2)
                labels.append(label)
                auxes.append(aux)
                
            s1_batch = pad_sequence(s1_batches, batch_first=True, padding_value=self.pad_id)
            s2_batch = pad_sequence(s2_batches, batch_first=True, padding_value=self.pad_id)
            if label != None:
                return s1_batch.long(), s2_batch.long(), torch.FloatTensor(labels), torch.stack(auxes, 0)
            else:
                return s1_batch.long(), s2_batch.long(), None
        elif self.model_type in ["BERT", "BERT_NLI"]:
            s1 = []
            labels = []
            auxes = []
            for b in batch:
                data, label, aux = b
                s1.append(data)
                labels.append(label)
                auxes.append(aux)
            s1_batch = pad_sequence(s1, batch_first=True, padding_value=self.pad_id)
            if label != None:
                return s1_batch.long(), torch.FloatTensor(labels), torch.stack(auxes, 0)
            else:
                return s1_batch.long(), None
        elif self.model_type == "MLM":
            inputs = []
            labels = []
            for b in batch:
                x, y = b
                inputs.append(x)
                labels.append(y)
            inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
            return inputs.long(), labels.long()

def bucket_pair_indices(
    sentence_length: List[Tuple[int, int]],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:
    batch_indices_list = []
    bucket = defaultdict(list)
    for idx, length in enumerate(sentence_length):
        s1_len, s2_len = length
        x = s1_len//max_pad_len
        y = s2_len//max_pad_len
        bucket[(x, y)].append(idx)
        if len(bucket[(x, y)]) == batch_size:
            batch_indices_list.append(bucket[(x, y)])
            bucket[(x, y)] = []
    for key in bucket.keys():
        batch_indices_list.append(bucket[key])

    random.shuffle(batch_indices_list)

    return batch_indices_list

