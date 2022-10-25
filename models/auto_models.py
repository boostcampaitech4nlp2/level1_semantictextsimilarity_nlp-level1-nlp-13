from transformers import AutoModel
import torch.nn as nn
import torch


class SBERT_base_Model(nn.Module):
    def __init__(self, model_name):
        super(SBERT_base_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size*2, 1)
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids).last_hidden_state
        v = self.bert(tgt_ids).last_hidden_state
        u = torch.mean(u, dim=1)
        v = torch.mean(v, dim=1)
        attn_outputs = torch.cat((u, v), dim=-1)
        outputs = self.linear(attn_outputs)

        return outputs
