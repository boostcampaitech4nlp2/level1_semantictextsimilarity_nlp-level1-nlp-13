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
import os
from set_seed import set_seed
import torchmetrics

from models import SBERT_base_Model, BERT_base_Model, BERT_base_NLI_Model, MLM_Model
from datasets import KorSTSDatasets, Collate_fn, bucket_pair_indices, KorSTSDatasets_for_BERT, KorNLIDatasets, KorSTSDatasets_for_MLM
from utils import train_step, valid_step, EarlyStopping
from EDA import OutputEDA


Models = {"BERT": BERT_base_Model, "SBERT": SBERT_base_Model, "BERT_NLI": BERT_base_NLI_Model, "MLM": MLM_Model}
Datasets = {"BERT": KorSTSDatasets_for_BERT, "SBERT": KorSTSDatasets, "BERT_NLI": KorNLIDatasets, "MLM": KorSTSDatasets_for_MLM}
Criterions = {"MAE": nn.L1Loss, "MSE": nn.MSELoss, "BCE": nn.BCELoss, "CE": nn.NLLLoss}

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print("training on", device)

    train_datasets = Datasets[config["model_type"]](config['train_csv'], config['base_model'], config["stopword"])
    valid_datasets = Datasets[config["model_type"]](config['valid_csv'], config['base_model'], config["stopword"])

    # EDA
    outputEDA = OutputEDA(config["base_model"], config["log_name"])
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
    
    if not config["test_mode"]:
        pj = "bert-mlm" if config["model_type"] == "MLM" else "sentence_bert"
        run = wandb.init(project=pj, entity="nlp-13", config=config, 
                         name=config['log_name'], notes=config['notes'])
        wandb.watch(model, log="all")

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

    if config["model_type"] == "MLM":
        earlystopping = EarlyStopping(patience=config["early_stopping_patience"], verbose=True, mode="min")
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config["lr_scheduler_factor"], 
                                      patience=config["lr_scheduler_patience"], verbose=True)
    else:
        earlystopping = EarlyStopping(patience=config["early_stopping_patience"], verbose=True, mode="max")
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=config["lr_scheduler_factor"], 
                                      patience=config["lr_scheduler_patience"], verbose=True)

    pbar = tqdm(range(epochs))

    # training code.
    for epoch in pbar:
        model.train()
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
        model.eval()
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(tqdm(valid_loader)):
                logits, loss, score = valid_step(data, config["model_type"], device, model, criterion, outputEDA)
                val_loss += loss
                val_score += score

            val_loss /= (i+1)
            val_score /= (i+1)
            if not config["test_mode"]:
                if config["model_type"] != "MLM": 
                    wandb.log({"valid loss": val_loss, "valid_pearson": val_score})
                else: 
                    wandb.log({"valid loss": val_loss, "valid_PPL": val_score})

            earlystopping(val_score)
            scheduler.step(val_score)
            if earlystopping.best_epoch:
                torch.save(model.state_dict(), config["model_save_path"])
                print("model saved to ", config["model_save_path"])
                if not config["test_mode"]:
                    outputEDA.save(epoch, val_score)
            outputEDA.reset()

        if earlystopping.earlystop:
            break
    

if __name__ == "__main__":
    # 실행 위치 고정
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    # 결과 재현성을 위한 랜덤 시드 고정.
    set_seed(13)

    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
    