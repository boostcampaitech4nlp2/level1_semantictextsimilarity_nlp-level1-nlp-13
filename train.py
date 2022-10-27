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

from models import SBERT_base_Model, BERT_base_Model
from datasets import KorSTSDatasets, Collate_fn, bucket_pair_indices, KorSTSDatasets_for_BERT


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on", device)

    if not config["test_mode"]:
        run = wandb.init(project="sentence_bert", entity="nlp-13", config=config, name=config['log_name'], notes=config['notes'])

    if config["model_type"] == "SBERT":
        train_datasets = KorSTSDatasets(config['train_csv'], config['base_model'])
        valid_datasets = KorSTSDatasets(config['valid_csv'], config['base_model'])
    elif config["model_type"] == "BERT":
        train_datasets = KorSTSDatasets_for_BERT(config['train_csv'], config['base_model'])
        valid_datasets = KorSTSDatasets_for_BERT(config['valid_csv'], config['base_model'])
    else:
        print("Model type should be 'BERT' or 'SBERT'!")
        return

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

    if config["model_type"] == "SBERT":
        model = SBERT_base_Model(config["base_model"])
    else:
        model = BERT_base_Model(config["base_model"])
        
    print("Base model is", config['base_model'])
    if os.path.exists(config["model_load_path"]):
        model.load_state_dict(torch.load(config["model_load_path"]))
        print("weights loaded from", config["model_load_path"])
    else:
        print("no pretrained weights provided.")
    model.to(device)

    epochs = config['epochs']
    criterion = nn.L1Loss()
    optimizer = Adam(params=model.parameters(), lr=config['lr'])

    pbar = tqdm(range(epochs))

    best_val_loss = 1000
    best_pearson = 0

    for epoch in pbar:
        for iter, data in enumerate(tqdm(train_loader)):
            if config["model_type"] == "SBERT":
                s1, s2, label = data
                s1 = s1.to(device)
                s2 = s2.to(device)
                label = label.to(device)
                logits = model(s1, s2)
            else:
                s1, label = data
                s1 = s1.to(device)
                label = label.to(device)
                logits = model(s1)
            loss = criterion(logits.squeeze(-1), label)
            pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            if not config["test_mode"]:
                wandb.log({"train_loss": loss, "train_pearson": pearson})
            pbar.set_postfix({"train_loss": loss})

        val_loss = 0
        val_pearson = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader)):
                if config["model_type"] == "SBERT":
                    s1, s2, label = data
                    s1 = s1.to(device)
                    s2 = s2.to(device)
                    label = label.to(device)
                    logits = model(s1, s2)
                else:
                    s1, label = data
                    s1 = s1.to(device)
                    label = label.to(device)
                    logits = model(s1)
                loss = criterion(logits.squeeze(-1), label)
                pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())
                val_loss += loss.to(torch.device("cpu")).detach().item()
                val_pearson += pearson.to(torch.device("cpu")).detach().item()
                if not config["test_mode"]:
                    wandb.log({"valid loss": loss, "valid_pearson": pearson})
            val_loss /= i
            val_pearson /= i
            if val_pearson > best_pearson:
                torch.save(model.state_dict(), config["model_save_path"])


if __name__ == "__main__":
    # 결과 재현성을 위한 랜덤 시드 고정.
    set_seed(13)

    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
    