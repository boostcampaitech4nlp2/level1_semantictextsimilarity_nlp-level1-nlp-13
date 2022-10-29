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
        self.tsv = pd.read_csv(dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.s1 = [tokenizer.encode(s1) for s1 in self.tsv["sentence_1"]]
        self.s2 = [tokenizer.encode(s2) for s2 in self.tsv["sentence_2"]]
        if "label" in self.tsv.keys():
            self.y = self.tsv["label"]
            self.b_y = self.tsv["binary-label"]
        else:
            self.y = None

        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        data = torch.IntTensor(self.s1[idx]), torch.IntTensor(self.s2[idx])
        if "label" in self.tsv.keys():
            label = float(self.y[idx])/5
        else:
            label = None
        return data, label

class KorSTSDatasets_for_BERT(KorSTSDatasets):
    def __init__(self, dir, model_name):
        super(KorSTSDatasets_for_BERT, self).__init__(dir, model_name)

    def __getitem__(self, idx):
        data = self.s1[idx][:-1] + [self.sep_id] + self.s2[idx][1:]
        data = torch.IntTensor(data)
        if "label" in self.tsv.keys():
            label = float(self.y[idx])
        else:
            label = None

        return data, label

class KorNLIDatasets(KorSTSDatasets):
    def __init__(self, dir, model_name):
        super(KorNLIDatasets, self).__init__(dir, model_name)

    def __getitem__(self, idx):
        data = self.s1[idx][:-1] + [self.sep_id] + self.s2[idx][1:]
        data = torch.IntTensor(data)
        if "label" in self.tsv.keys():
            label = int(self.b_y[idx])
        else:
            label = None

        return data, label

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
            for b in batch:
                data, label = b
                s1, s2 = data
                s1_batches.append(s1)
                s2_batches.append(s2)
                labels.append(label)
                
            s1_batch = pad_sequence(s1_batches, batch_first=True, padding_value=self.pad_id)
            s2_batch = pad_sequence(s2_batches, batch_first=True, padding_value=self.pad_id)
            if label != None:
                return s1_batch.long(), s2_batch.long(), torch.FloatTensor(labels)
            else:
                return s1_batch.long(), s2_batch.long(), None
        else:
            s1 = []
            labels = []
            for b in batch:
                data, label = b
                s1.append(data)
                labels.append(label)
            s1_batch = pad_sequence(s1, batch_first=True, padding_value=self.pad_id)
            if label != None:
                return s1_batch.long(), torch.FloatTensor(labels)
            else:
                return s1_batch.long(), None

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

