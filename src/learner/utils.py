import torch
import numpy as np
from tqdm.auto import tqdm


def train(model, train_dl, opt, loss_func, device='cpu'):
    train_loss = 0
    train_counter = 0
    pb = tqdm(train_dl)
    for batch in pb:
        inputs = batch['image']
        inputs = inputs.to(device)
        preds = model(inputs)
        loss = loss_func(preds, inputs)
        loss.backward()
        opt.step()
        opt.zero_grad()
        _batch_size = np.product(inputs.shape)
        train_loss += loss.item()*_batch_size
        train_counter += _batch_size
        pb.set_description(f"Batch Loss:{loss.item():.3f}")
    train_loss = train_loss/train_counter
    return train_loss


def eval(model, valid_dl, loss_func, device = 'cpu'):
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        valid_counter = 0
        pb = tqdm(valid_dl)
        for batch in pb:
            inputs = batch['image']
            inputs = inputs.to(device)
            preds = model(inputs)
            loss = loss_func(preds, inputs)
            _batch_size = np.product(inputs.shape)
            valid_loss += loss.item()*_batch_size
            valid_counter += _batch_size
            pb.set_description(f"Batch Loss:{loss.item():.3f}")
    valid_loss = valid_loss/valid_counter
    return valid_loss


def fit(CFG, model, opt, train_dl, valid_dl, loss_func):
    
    for epoch in tqdm(range(CFG.EPOCHS)):
        model.train()
        train_loss = train(model, train_dl, opt, loss_func, device=CFG.device)
        valid_loss = eval(model, valid_dl, loss_func, device=CFG.device)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")