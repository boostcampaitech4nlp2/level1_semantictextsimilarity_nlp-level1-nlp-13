from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from collections import defaultdict
from typing import List, Tuple
import random


class KorSTSDatasets(Dataset):
    def __init__(self, dir, model_name):
        super(KorSTSDatasets, self).__init__()
        tsv = pd.read_csv(dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        #read stopwords
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
                
        rtt_filter = tsv['source'].str.contains('rtt') 
        self.rtt = pd.Series([0] * len(tsv))
        self.rtt[rtt_filter] = 1
        
        nsmc_filter = tsv['source'].str.contains('nsmc')    
        petition_filter = tsv['source'].str.contains('petition') 
        slack_filter = tsv['source'].str.contains('slack') 
        self.source = pd.Series([0] * len(tsv))
        self.source[nsmc_filter] = 0
        self.source[petition_filter] = 1
        self.source[slack_filter] = 2
        
        try:
            self.y = tsv["label"]
        except:
            self.y = [0 for _ in range(len(self.s1))]
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = torch.IntTensor(self.s1[idx]), torch.IntTensor(self.s2[idx])
        # cosine similarity의 범위 [-1. ~ 1.] 사이 값으로 정규화 필요.
        # label = float(self.y[idx]) * 0.4 - 1
        label = float(self.y[idx])
        
        # torch.cat으로 aux만들기 위해 1차원 텐서로 변환
        rtt = torch.tensor([self.rtt[idx]], dtype=torch.long)
        source = torch.tensor(self.source[idx], dtype=torch.long)
        source = torch.nn.functional.one_hot(source, 3)
        # source + rtt 
        aux = torch.cat([source, rtt])
        
        return data, label, aux

class KorSTSDatasets_for_BERT(KorSTSDatasets):
    def __init__(self, dir, model_name):
        super(KorSTSDatasets_for_BERT, self).__init__(dir, model_name)

    def __getitem__(self, idx):
        data = self.s1[idx][:-1] + [self.sep_id] + self.s2[idx][1:]
        data = torch.IntTensor(data)
        label = float(self.y[idx])
        
        # torch.cat으로 aux만들기 위해 1차원 텐서로 변환
        rtt = torch.tensor([self.rtt[idx]], dtype=torch.long)
        source = torch.tensor(self.source[idx], dtype=torch.long)
        source = torch.nn.functional.one_hot(source, 3)
        # source + rtt 
        aux = torch.cat([source, rtt])
        
        return data, label, aux

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
            return s1_batch.long(), s2_batch.long(), torch.FloatTensor(labels), auxes
        else:
            s1 = []
            labels = []
            auxes = []
            for b in batch:
                data, label, aux = b
                s1.append(data)
                labels.append(label)
                auxes.append(aux)
                
            s1_batch = pad_sequence(s1, batch_first=True, padding_value=self.pad_id)
            return s1_batch.long(), torch.FloatTensor(labels), auxes


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

