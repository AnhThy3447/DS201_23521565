import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
                            recall_score, confusion_matrix

def train(model, model_name, dataloader, num_epochs, optimizer, loss_fn, device):
    model.train()

    print("----- Training -----")
    for epoch in range(num_epochs):
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            batch_data = batch["image"].float().to(device)
            batch_labels = batch["label"].long().to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}")

def compute_score(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def evaluate(model, model_name, test_dataloader, loss_fn, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
            batch_data = batch["image"].float().to(device)
            batch_labels = batch["label"].long().to(device)

            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.tolist())


    avg_loss = total_loss / len(test_dataloader)

    print("\n----- Evaluation Results -----")
    print(f"Test Loss: {avg_loss:.4f}")
    compute_score(all_preds, all_labels)