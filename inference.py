from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
import wandb
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import os
from set_seed import set_seed
from transformers import AutoTokenizer
import torchmetrics
import pandas as pd

from models import SBERT_base_Model, BERT_base_Model
from datasets import KorSTSDatasets, Collate_fn, bucket_pair_indices, KorSTSDatasets_for_BERT


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_type', default='BERT', type=str)
    parser.add_argument('--model_path', default='results/klue-roberta-large2.pt', type=str)
    parser.add_argument('--valid_path', default='NLP_dataset/han_processed_dev.csv', type=str)
    parser.add_argument('--test_path', default='NLP_dataset/han_processed_test.csv', type=str)
    parser.add_argument('--stopword',default=False)
    args = parser.parse_args()

    test_datasets = KorSTSDatasets_for_BERT(args.test_path, args.model_name, args.stopword)
    valid_datasets = KorSTSDatasets_for_BERT(args.valid_path, args.model_name, args.stopword)
    collate_fn = Collate_fn(test_datasets.pad_id, args.model_name)

    test_loader = DataLoader(
        test_datasets, 
        collate_fn=collate_fn,
        batch_size=64,
    )
    valid_loader = DataLoader(
        valid_datasets,
        collate_fn=collate_fn,
        batch_size=64,
    )
    model = BERT_base_Model(args.model_name)
    model.load_state_dict(torch.load(args.model_path))
    print("weights loaded from", args.model_path)
    model.to(device)
    val_predictions = []
    val_labels = []
    test_predictions = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(tqdm(valid_loader)):
            if args.model_type == "SBERT":
                s1, s2, label, aux = data
                s1 = s1.to(device)
                s2 = s2.to(device)
                label = label.to(device)
                aux = aux.to(device)
                logits = model(s1, s2, aux)
            else:
                s1, label, aux = data
                s1 = s1.to(device)
                label = label.to(device)
                aux = aux.to(device)
                logits = model(s1, aux)
                logits = logits.squeeze(-1)
            for logit in logits.to(torch.device("cpu")).detach():
                val_predictions.append(logit)
            for lab in label.to(torch.device("cpu")).detach(): 
                val_labels.append(lab)

        for i, data in enumerate(tqdm(test_loader)):
            if args.model_type == "SBERT":
                s1, s2, label, aux = data
                s1 = s1.to(device)
                s2 = s2.to(device)
                aux = aux.to(device)
                logits = model(s1, s2, aux)
            else:
                s1, label, aux = data
                s1 = s1.to(device)
                aux = aux.to(device)
                logits = model(s1, aux)
                if args.model_type == "FBERT": 
                    logits = logits[:, 0]  
                else:
                    logits = logits.squeeze(-1)
            for logit in logits.to(torch.device("cpu")).detach():
                test_predictions.append(logit)

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.

    pearson = torchmetrics.functional.pearson_corrcoef(torch.tensor(val_predictions), torch.tensor(val_labels))
    print("valid pearson = ",pearson)
    test_predictions = list(round(float(i), 1) for i in test_predictions)
    for i in range(len(test_predictions)):
        if test_predictions[i] < 0:
            test_predictions[i] = 0.0
        elif test_predictions[i] > 5:
            test_predictions[i] = 5.0
    output = pd.read_csv('NLP_dataset/sample_submission.csv')
    output['target'] = test_predictions
    output.to_csv('output.csv', index=False)
 