from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

def evaluate_ner(model, dataloader, pad_tag_id, criterion, device):
    model.to(device)
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['lengths']
            labels = batch['labels'].to(device)

            outputs = model(input_ids, lengths)
            preds = outputs.argmax(dim=-1)

            mask = labels != pad_tag_id
            all_labels.extend(labels[mask].tolist())
            all_preds.extend(preds[mask].tolist())

            _, _, C = outputs.shape
            loss = criterion(outputs.view(-1, C), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1

def evaluate_cls(model, dataloader, criterion, device):
    model.to(device)
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['lengths']
            labels = batch['labels'].to(device)

            outputs = model(input_ids, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, zero_division=0)
    return avg_loss, f1, report

def train_epoch(model, dataloader, task_type, optimizer, criterion, 
                num_epoch, val_dataloader, save_path, device, pad_tag_id=-100):
    best_score = 0.0
    model.to(device)

    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        train_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epoch} [Training]", leave=False)
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['lengths']
            labels = batch['labels'].to(device)

            outputs = model(input_ids, lengths)
            if task_type == 'NER':
                B, L, C = outputs.shape
                outputs = outputs.view(B * L, C)
                labels = labels.view(B * L)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        train_loss = np.mean(train_losses)

        if task_type == 'NER':
            val_loss, val_f1 = evaluate_ner(model, val_dataloader, pad_tag_id, criterion, device)
        else:
            val_loss, val_f1, _ = evaluate_cls(model, val_dataloader, criterion, device)

        print (f"Epoch {epoch+1}/{num_epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_score:
            best_score = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1}")