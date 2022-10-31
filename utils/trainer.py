import torchmetrics
import torch


def step(data, model_type, device, model, criterion, outputEDA=None):
    if model_type == "SBERT":
        s1, s2, label, aux = data
        s1 = s1.to(device)
        s2 = s2.to(device)
        label = label.to(device)
        logits = model(s1, s2).squeeze()
        loss = criterion(logits, label)
        score = torchmetrics.functional.pearson_corrcoef(logits, label.squeeze())
        
        if outputEDA != None:
            outputEDA.appendf(label, logits, aux, s1, s2)
    else:
        if model_type == "MLM":
            s1, label = data
        else:
            s1, label, aux = data
        s1 = s1.to(device)
        label = label.to(device)
        logits = model(s1).squeeze()
        if model_type in ["BERT", "BERT_NLI"]:
            loss = criterion(logits.squeeze(-1), label)
            score = torchmetrics.functional.pearson_corrcoef(logits, label.squeeze())
        elif model_type == "MLM":
            loss = criterion(logits.transpose(1, 2), label)
            score = torch.exp(loss)
            
        if model_type != "MLM" and outputEDA != None:
            outputEDA.appendf(label, logits, aux, s1, None)
    
    return logits, loss, score


def train_step(data, model_type, device, model, criterion, optimizer):
    logits, loss, score = step(data, model_type, device, model, criterion)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss= loss.detach().item()
    score = score.detach().item()
    
    return loss, score

def valid_step(data, model_type, device, model, criterion, outputEDA):
    logits, loss, score = step(data, model_type, device, model, criterion, outputEDA)
    loss= loss.detach().item()
    score = score.detach().item()
    
    return logits, loss, score
