from transformers import AutoModel, BertForSequenceClassification, RobertaForSequenceClassification
import torch.nn as nn
import torch


class SBERT_base_Model(nn.Module):
    def __init__(self, model_name):
        super(SBERT_base_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size*2, 1)
        self.similarity = nn.CosineSimilarity()

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids).pooler_output
        v = self.bert(tgt_ids).pooler_output

        outputs = self.similarity(u, v)
        
        return outputs


class BERT_base_Model(nn.Module):
    def __init__(self, model_name):
        super(BERT_base_Model, self).__init__()
        if "roberta" in model_name:
            self.bert = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
        else:
            self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids):
        outputs = self.bert(src_ids)

        return outputs.logits
