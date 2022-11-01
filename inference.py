from models import SBERT_base_Model, BERT_base_Model, BERT_base_NLI_Model
from datasets import KorSTSDatasets, Collate_fn, KorSTSDatasets_for_BERT, KorNLIDatasets
from utils import test_step

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd


Datasets = {"SBERT": KorSTSDatasets, "BERT": KorSTSDatasets_for_BERT, "BERT_NLI": KorNLIDatasets}
Models = {"SBERT": SBERT_base_Model, "BERT": BERT_base_Model, "BERT_NLI": BERT_base_NLI_Model}

def main(config):
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    print("prepare datasets")
    datasets = Datasets[config["model_type"]](config["test_csv"], config["base_model"])

    collate_fn = Collate_fn(datasets.pad_id, config["model_type"])

    data_loader = DataLoader(
        datasets,
        collate_fn=collate_fn,
        batch_size=config["batch_size"]
    )

    print("load model...")
    model = Models[config["model_type"]](config["base_model"])

    model.load_state_dict(torch.load(config["model_load_path"]))
    print("model loaded from", config["model_load_path"])
    
    model.to(device)

    model.eval()

    preds = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            pred = test_step(data, config["model_type"], device, model)
            pred = pred.to(torch.device("cpu")).detach().numpy().flatten()
            preds += list(pred)

    output = pd.read_csv("NLP_dataset/sample_submission.csv")
    preds = [round(np.clip(p, 0, 5), 1) for p in preds]
    output['target'] = preds
    output.to_csv("output.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
