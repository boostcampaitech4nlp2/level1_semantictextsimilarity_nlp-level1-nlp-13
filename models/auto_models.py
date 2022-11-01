from transformers import AutoModel, BertForSequenceClassification, AutoModelForSequenceClassification, RobertaForMaskedLM
import torch.nn as nn
import torch

class SBERT_base_Model(nn.Module):
    def __init__(self, model_name, dropout_prob):
        super(SBERT_base_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, hidden_dropout_prob=dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size*2, 1)
        self.similarity = nn.CosineSimilarity()

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids).pooler_output
        v = self.bert(tgt_ids).pooler_output

        outputs = self.similarity(u, v)
        
        return outputs


class BERT_base_Model(nn.Module):
    def __init__(self, model_name, dropout_prob):
        super(BERT_base_Model, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, hidden_dropout_prob=dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids):
        outputs = self.bert(src_ids)

        return outputs.logits
    

class BERT_base_NLI_Model(nn.Module):
    def __init__(self, model_name, dropout_prob):
        super(BERT_base_NLI_Model, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, hidden_dropout_prob=dropout_prob)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, src_ids):
        outputs = self.bert(src_ids).logits
        outputs = self.sigmoid(outputs)
        
        return outputs
    

class MLM_Model(nn.Module):
    def __init__(self, model_name, dropout_prob):
        super(MLM_Model, self).__init__()
        self.bert = RobertaForMaskedLM.from_pretrained(model_name, hidden_dropout_prob=dropout_prob)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_ids):
        outputs = self.bert(src_ids).logits
        outputs = self.softmax(outputs)
        
        return outputs


class SimCSE(nn.Module):
    def __init__(self, model_name, dropout_prob):
        super(SimCSE, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, hidden_dropout_prob=dropout_prob)
        self.cosine_similarity = nn.CosineSimilarity()
        
    def forward(self, src_ids):
        outputs1 = self.bert(src_ids).pooler_output
        outputs2 = self.bert(src_ids).pooler_output
        score = self.cosine_similarity(outputs1.unsqueeze(1), outputs2.unsqueeze(0))
        
        return outputs1, score
        
    