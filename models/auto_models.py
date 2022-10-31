from transformers import AutoModel, BertForSequenceClassification, AutoModelForSequenceClassification
import torch.nn as nn
import torch

class SBERT_base_Model(nn.Module):
    def __init__(self, model_name):
        super(SBERT_base_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size*2, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids, tgt_ids, aux):
        u = self.bert(src_ids).last_hidden_state
        v = self.bert(tgt_ids).last_hidden_state
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)
        attn_outputs = torch.cat((u, v), dim=-1)
        outputs = self.linear(attn_outputs)

        return outputs


class BERT_base_Model(nn.Module):
    def __init__(self, model_name):
        super(BERT_base_Model, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids, aux):
        # attn_outputs = self.bert(src_ids).last_hidden_state
        # pooler_outputs = torch.mean(attn_outputs, dim=1)
        # outputs = self.linear(pooler_outputs)
        outputs = self.bert(src_ids)

        return outputs.logits
