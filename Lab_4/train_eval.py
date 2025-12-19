from tqdm import tqdm
import numpy as np
import torch
from rouge_score import rouge_scorer

def evaluate(model, dataloader, criterion, vocab, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    total_loss = 0
    total_rouge_l = 0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['src_language'].to(device)
            y = batch['tgt_language'].to(device)
            
            outputs = model(x, y[:, :-1]) 
            target = y[:, 1:]
            target = target[:, :outputs.size(1)]
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
            total_loss += loss.item()
            
            # ROUGE-L
            preds = model.predict(x, max_len=y.size(1) + 10)
            preds = preds.cpu()
            y_cpu = y.cpu()
            
            for i in range(x.size(0)):
                str_pred = vocab.decode_sentence(preds[i], "vietnamese")
                str_ref = vocab.decode_sentence(y_cpu[i], "vietnamese")
                
                score = scorer.score(str_ref, str_pred)
                total_rouge_l += score['rougeL'].fmeasure
                sample_count += 1
                
    avg_loss = total_loss / len(dataloader)
    avg_rouge = total_rouge_l / sample_count
    
    return avg_loss, avg_rouge

def train_epoch(model, dataloader, optimizer, criterion, num_epoch,
                val_dataloader, vocab, save_path, device):
    best_score = 0.0
    model.to(device)

    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        train_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epoch} [Training]", leave=False)
        for batch in train_bar:
            x = batch['src_language'].to(device)
            y = batch['tgt_language'].to(device)

            outputs = model(x, y)
            target = y[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        train_loss = np.mean(train_losses)
        val_loss, val_rouge = evaluate(model, val_dataloader, criterion, vocab, device)

        print (f"Epoch {epoch+1}/{num_epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Rouge_L: {val_rouge:.4f}")

        if val_rouge > best_score:
            best_score = val_rouge
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1}")
