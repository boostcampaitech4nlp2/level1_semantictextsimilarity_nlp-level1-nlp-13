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

from models import SBERT_base_Model, BERT_base_Model, MLM_Model
from datasets import KorSTSDatasets, Collate_fn, bucket_pair_indices, KorSTSDatasets_for_BERT, KorSTSDatasets_for_MLM
from criterion import CrossEntropyLoss


Datasets = {"SBERT": KorSTSDatasets, "BERT": KorSTSDatasets_for_BERT, "MLM": KorSTSDatasets_for_MLM}
Models = {"SBERT": SBERT_base_Model, "BERT": BERT_base_Model, "MLM": MLM_Model}
Criterions = {"SBERT": nn.MSELoss, "BERT": nn.L1Loss, "MLM": nn.NLLLoss}


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on", device)

    if not config["test_mode"]:
        run = wandb.init(project="sentence_bert", entity="nlp-13", config=config, name=config['log_name'], notes=config['notes'])

    train_datasets = Datasets[config["model_type"]](config["train_csv"], config["base_model"])
    valid_datasets = Datasets[config["model_type"]](config["valid_csv"], config["base_model"])

    # get pad_token_id.
    collate_fn = Collate_fn(train_datasets.pad_id, config["model_type"])

    # pair-bucket sampler
    # train_seq_lengths = [(len(s1), len(s2)) for (s1, s2) in train_datasets.x]
    # train_sampler = bucket_pair_indices(train_seq_lengths, batch_size=config['batch_size'], max_pad_len=10)

    train_loader = DataLoader(
        train_datasets, 
        collate_fn=collate_fn, 
        shuffle=True,
        batch_size=config['batch_size'],
        # batch_sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_datasets,
        collate_fn=collate_fn,
        batch_size=config['batch_size']
    )
    
    model = Models[config["model_type"]](config["base_model"])
        
    print("Base model is", config['base_model'])
    if os.path.exists(config["model_load_path"]):
        model.load_state_dict(torch.load(config["model_load_path"]))
        print("weights loaded from", config["model_load_path"])
    else:
        print("no pretrained weights provided.")
    model.to(device)

    epochs = config['epochs']
    criterion = Criterions[config["model_type"]](ignore_index=0)
    
    optimizer = Adam(params=model.parameters(), lr=config['lr'])

    pbar = tqdm(range(epochs))

    best_val_loss = 1000
    best_pearson = 0

    # training code.
    for epoch in pbar:
        for iter, data in enumerate(tqdm(train_loader)):
            if config["model_type"] == "SBERT":
                s1, s2, label = data
                s1 = s1.to(device)
                s2 = s2.to(device)
                label = label.to(device)
                logits = model(s1, s2)
                loss = criterion(logits.squeeze(-1), label)
                pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())
            elif config["model_type"] == "BERT":
                s1, label = data
                s1 = s1.to(device)
                label = label.to(device)
                logits = model(s1)
                loss = criterion(logits.squeeze(-1), label)
                pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())
            else:
                s1, label = data
                s1 = s1.to(device)
                label = label.to(device)
                logits = model(s1)
                loss = criterion(logits.transpose(1, 2), label)
                ppl = torch.exp(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            if not config["test_mode"]:
                if config["model_type"] != "MLM":
                    wandb.log({"train_loss": loss, "train_pearson": pearson})
                else:
                    wandb.log({"train_loss": loss, "train_PPL": ppl})
            pbar.set_postfix({"train_loss": loss})
        val_loss = 0
        val_pearson = 0
        val_ppl = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader)):
                if config["model_type"] == "SBERT":
                    s1, s2, label = data
                    s1 = s1.to(device)
                    s2 = s2.to(device)
                    label = label.to(device)
                    logits = model(s1, s2)
                    loss = criterion(logits.squeeze(-1), label)
                    pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())
                    val_pearson += pearson.to(torch.device("cpu")).detach().item()
                elif config["model_type"] == "BERT":
                    s1, label = data
                    s1 = s1.to(device)
                    label = label.to(device)
                    logits = model(s1)
                    loss = criterion(logits.squeeze(-1), label)
                    pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())
                    val_pearson += pearson.to(torch.device("cpu")).detach().item()
                else:
                    s1, label = data
                    s1 = s1.to(device)
                    label = label.to(device)
                    logits = model(s1)
                    loss = criterion(logits.transpose(1, 2), label)
                    ppl = torch.exp(loss)
                    val_ppl += ppl.to(torch.device("cpu")),detach().item()
                val_loss += loss.to(torch.device("cpu")).detach().item()
            
            val_loss /= i+1
            if not config["test_mode"]:
                if config["model_type"] != "MLM":
                    val_pearson /= i+1
                    wandb.log({"valid loss": val_loss, "valid_pearson": val_pearson})
                else:
                    val_ppl /= i+1
                    wandb.log({"valid loss": val_loss, "valid_PPL": val_ppl})

            if config["model_type"] == "MLM":
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), config["model_save_path"])
            else:
                if val_pearson > best_pearson:
                    torch.save(model.state_dict(), config["model_save_path"])

    torch.save(model.state_dict(), config["model_save_path"])
    

if __name__ == "__main__":
    # 실행 위치 고정
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 결과 재현성을 위한 랜덤 시드 고정.
    set_seed(13)

    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
    
print('w')