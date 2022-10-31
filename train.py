import math
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from EDA import OutputEDA


class EarlyStopping:
    def __init__(self, path, patience=5, verbose=False, mode="min"):
        self.path = path
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.patience_cnt = 0
        self.earlystop = False
        self.best_epoch = False
        
        if self.mode == "min":
            self.ref = math.inf
        elif self.mode == "max":
            self.ref = -math.inf          
        else:
            raise ValueError("mode can be 'min' or 'max' only.")

    def __call__(self, cur_ref, model):
        if (self.mode == "max" and cur_ref > self.ref) \
            or (self.mode == "min" and cur_ref < self.ref):      
                if self.verbose:
                    print(f'Earlystop: the best target value is changed. [{self.ref:.4f} > {cur_ref:.4f}]')
                torch.save(model.state_dict(), self.path)
                self.patience_cnt = 0
                self.ref = cur_ref
                self.best_epoch = True
        else:
            self.patience_cnt += 1
            self.best_epoch = False
            if self.patience_cnt >= self.patience:
                if self.verbose:
                    print('earlystopping')   
                self.earlystop = True
                
def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on", device)

    if not config["test_mode"]:
        run = wandb.init(project="sentence_bert", entity="nlp-13", config=config, name=config['log_name'], notes=config['notes'])
    
    
    if config["model_type"] == "SBERT":
        train_datasets = KorSTSDatasets(config['train_csv'], config['base_model'], config['stopword'])
        valid_datasets = KorSTSDatasets(config['valid_csv'], config['base_model'], config['stopword'])
    elif config["model_type"] == "BERT":
        train_datasets = KorSTSDatasets_for_BERT(config['train_csv'], config['base_model'], config['stopword'])
        valid_datasets = KorSTSDatasets_for_BERT(config['valid_csv'], config['base_model'], config['stopword'])
    else:
        print("Model type should be 'BERT' or 'SBERT'!")
        return
    # EDA
    outputEDA = OutputEDA(config['base_model'], config['log_name'])
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
    
    if not config["test_mode"]:
        wandb.watch(model, log="all")
        
    print("Base model is", config['base_model'])
    if os.path.exists(config["model_load_path"]):
        model.load_state_dict(torch.load(config["model_load_path"]))
        print("weights loaded from", config["model_load_path"])
    else:
        print("no pretrained weights provided.")
    model.to(device)
    

    epochs = config['epochs']
    criterion = nn.MSELoss()
    
    optimizer = Adam(params=model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.35, patience=4, verbose=True)
    
    earlystopping = EarlyStopping(config['model_save_path'], patience=10, verbose=True, mode="max")
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        model.train()
        for iter, data in enumerate(tqdm(train_loader)):
             #TODO : USE aux data [(one hot )]
            if config["model_type"] == "SBERT":
                s1, s2, label, aux = data
                s1 = s1.to(device)
                s2 = s2.to(device)
                label = label.to(device)
                logits = model(s1, s2)

            else:
                s1, label, aux = data
                s1 = s1.to(device)
                label = label.to(device)
                logits = model(s1)
                s2 = None

            pred = logits.squeeze(-1)
            loss = criterion(pred, label)
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
            model.eval()
            for i, data in enumerate(tqdm(valid_loader)):
                if config["model_type"] == "SBERT":
                    s1, s2, label, aux = data
                    s1 = s1.to(device)
                    s2 = s2.to(device)
                    label = label.to(device)
                    logits = model(s1, s2)
                else:
                    s1, label, aux = data
                    s1 = s1.to(device)
                    label = label.to(device)
                    logits = model(s1)
                    s2 = None 
                pred = logits.squeeze(-1)
                outputEDA.appendf(label, pred, aux, s1, s2)
                loss = criterion(pred, label)
                pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), label.squeeze())
                val_loss += loss.to(torch.device("cpu")).detach().item()
                val_pearson += pearson.to(torch.device("cpu")).detach().item()
            val_loss /= i + 1
            val_pearson /= i + 1
            if not config["test_mode"]:
                    wandb.log({"valid loss": val_loss, "valid_pearson": val_pearson})
            # early stopping
            earlystopping(val_pearson, model)
            scheduler.step(val_pearson)
            if earlystopping.best_epoch:
                outputEDA.save(epoch, val_pearson)
            outputEDA.reset()
        if earlystopping.earlystop:
            break

    

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