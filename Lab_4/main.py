import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_utils.dataset import PhoMTDataset, collate_fn
from data_utils.vocab import Vocab
from train_eval import train_epoch, evaluate
from model.seq2seq import Seq2seq

# ------ Define parameters ------
src_language = "english"
tgt_language = "vietnamese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------ Parse arguments ------
parser = argparse.ArgumentParser()
parser.add_argument("--attention", type=str, default="None")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
attention_name = args.attention

# ------ Load Dataset ------
vocab = Vocab(r'/workspaces/DS210/Lab_4/data', src_language, tgt_language)
train = PhoMTDataset(r'/workspaces/DS210/Lab_4/data/small-train.json', 
                     vocab, src_language, tgt_language)
val = PhoMTDataset(r'/workspaces/DS210/Lab_4/data/small-dev.json', 
                     vocab, src_language, tgt_language)
test = PhoMTDataset(r'/workspaces/DS210/Lab_4/data/small-test.json', 
                     vocab, src_language, tgt_language)

train_dataloader = DataLoader(
    dataset=train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, vocab, src_language, tgt_language)
)

val_dataloader = DataLoader(
    dataset=val,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, vocab, src_language, tgt_language)
)

test_dataloader = DataLoader(
    dataset=test,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, vocab, src_language, tgt_language)
)

# ------ Modeling ------
model = Seq2seq(vocab=vocab, attention_type=attention_name)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------ Training ------
save_path = f"model/BestModel/best_model_{attention_name}.pt"
train_epoch(model, train_dataloader, optimizer, criterion, num_epochs,
            val_dataloader, vocab, save_path, device)
model.load_state_dict(torch.load(save_path))

# ------ Evaluate ------
print("----- Test Results ----_")
_, test_rouge = evaluate(model, test_dataloader, criterion, vocab, device)
print(f"Test Rouge_L Score: {test_rouge:.4f}")
