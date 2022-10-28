from models import SBERT_base_Model, BERT_base_Model
from datasets import KorSTSDatasets, Collate_fn, KorSTSDatasets_for_BERT
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd


Datasets = {"SBERT": KorSTSDatasets, "BERT": KorSTSDatasets_for_BERT}
Models = {"SBERT": SBERT_base_Model, "BERT": BERT_base_Model}


def main(config):
    device = torch.device("cuda") if torch.cuda.is_available else toch.device("cpu")

    datasets = Datasets[config["model_type"]](config["test_csv"], config["base_model"])

    collate_fn = Collate_fn(datasets.pad_id, config["model_type"])

    data_loader = DataLoader(
        datasets,
        collate_fn=collate_fn,
        batch_size=config["batch_size"]
    )

    model = Models[config["model_type"]](config["base_model"])

    model.load_state_dict(torch.load(config["model_load_path"]))

    model.to(device)

    model.eval()

    preds = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            if config["model_type"] == "SBERT":
                s1, s2, label = data
                s1 = s1.to(device)
                s2 = s2.to(device)
                pred = model(s1, s2).to(torch.device("cpu")).detach().numpy().flatten()
                preds += list(pred)
            elif config["model_type"] == "BERT":
                s1, label = data
                s1 = s1.to(device)
                pred = model(s1).to(torch.device("cpu")).detach().numpy().flatten()
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
