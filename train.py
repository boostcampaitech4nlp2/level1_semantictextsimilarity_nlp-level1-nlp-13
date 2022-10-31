from email.policy import strict
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.optim import Adam
import wandb
import yaml
import argparse
from tqdm import tqdm
import os
from set_seed import set_seed
import torchmetrics

from models import SBERT_base_Model, BERT_base_Model, BERT_base_NLI_Model, MLM_Model
from datasets import KorSTSDatasets, Collate_fn, bucket_pair_indices, KorSTSDatasets_for_BERT, KorNLIDatasets, KorSTSDatasets_for_MLM
from utils import train_step, valid_step
from EDA import OutputEDA


Models = {"BERT": BERT_base_Model, "SBERT": SBERT_base_Model, "BERT_NLI": BERT_base_NLI_Model, "MLM": MLM_Model}
Datasets = {"BERT": KorSTSDatasets_for_BERT, "SBERT": KorSTSDatasets, "BERT_NLI": KorNLIDatasets, "MLM": KorSTSDatasets_for_MLM}
Criterions = {"MAE": nn.L1Loss, "MSE": nn.MSELoss, "BCE": nn.BCELoss, "CE": nn.NLLLoss}

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on", device)

    if not config["test_mode"]:
        run = wandb.init(project="sentence_bert", entity="nlp-13", config=config, name=config['log_name'], notes=config['notes'])

    train_datasets = Datasets[config["model_type"]](config['train_csv'], config['base_model'])
    valid_datasets = Datasets[config["model_type"]](config['valid_csv'], config['base_model'])

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
        try:
            model.load_state_dict(torch.load(config["model_load_path"]))
        except:
            print("Weights dosen't match exactly with keys. So weights will loaded not strictly.")
            model.load_state_dict(torch.load(config["model_load_path"]), strict=False)
        print("weights loaded from", config["model_load_path"])
    else:
        print("no pretrained weights provided.")
    model.to(device)

    epochs = config['epochs']
    if config["model_type"] == "MLM":
        criterion = Criterions[config["loss"]](ignore_index=0)
    else:
        criterion = Criterions[config["loss"]]()
    
    optimizer = Adam(params=model.parameters(), lr=config['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    pbar = tqdm(range(epochs))

    best_val_loss = 1000
    best_pearson = 0

    # training code.
    for epoch in pbar:
        for iter, data in enumerate(tqdm(train_loader)):
            loss, score = train_step(data, config["model_type"], device, model, criterion, optimizer)
            
            if not config["test_mode"]:
                if config["model_type"] != "MLM":
                    wandb.log({"train_loss": loss, "train_pearson": score})
                else:
                    wandb.log({"train_loss": loss, "train_PPL": score})
            pbar.set_postfix({"train_loss": loss})

        val_loss = 0
        val_score = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader)):
                logits, loss, score = valid_step(data, config["model_type"], device, model, criterion)
                val_loss += loss
                val_score += score

            val_loss /= (i+1)
            val_score /= (i+1)
            if not config["test_mode"]:
                if config["model_type"] != "MLM": wandb.log({"valid loss": val_loss, "valid_pearson": val_score})
                else: wandb.log({"valid loss": val_loss, "valid_PPL": val_score})

            if config["model_type"] == "MLM":
                if val_loss < best_val_loss: torch.save(model.state_dict(), config["model_save_path"])
            else:
                if val_score > best_pearson: torch.save(model.state_dict(), config["model_save_path"])
    

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
    