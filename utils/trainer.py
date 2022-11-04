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
            
    elif model_type == "MLM":
        s1, label = data
        s1 = s1.to(device)
        label = label.to(device)
        logits = model(s1).squeeze()
        loss = criterion(logits.transpose(1, 2), label)
        score = torch.exp(loss)
        
    elif model_type == "BERT":
        s1, label, aux = data
        s1 = s1.to(device)
        label = label.to(device)
        logits = model(s1).squeeze()
        loss = criterion(logits, label)
        score = torchmetrics.functional.pearson_corrcoef(logits, label.squeeze())
    
        if outputEDA != None:
            outputEDA.appendf(label, logits, aux, s1, None)
            
    elif model_type == "BERT_NLI":
        s1, label, aux = data
        s1 = s1.to(device)
        label = label.to(device)
        logits = model(s1).squeeze()
        loss = criterion(logits, label)
        score = torch.sum(torch.max(logits, dim=1).indices == label) / s1.shape[0]
            
    elif model_type == "SimCSE":
        s1, label = data
        s1 = s1.to(device)
        label = label.to(device)
        logits, cos_sim = model(s1)
        loss = criterion(cos_sim, label)
        score = torch.IntTensor([0])
    
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

def test_step(data, model_type, device, model):
    if model_type == "SBERT":
        s1, s2, label = data
        s1 = s1.to(device)
        s2 = s2.to(device)
        logits = model(s1, s2).squeeze()
    else:
        s1, label = data
        s1 = s1.to(device)
        logits = model(s1).squeeze()
    
    return logits
