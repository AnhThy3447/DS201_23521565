import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_utils.phonert import phonert, collate_phonert
from data_utils.uit_viocd import uit_viocd, collate_viocd
from data_utils.utils import build_vocab_phonert, build_vocab_viocd
from model.transformer_model import Transformer_CLS, Transformer_NER
from train_eval import train_epoch, evaluate_cls, evaluate_ner

# ----- Define parameters -----
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Parse arguments -----
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="uit_viocd",
                    choices=["uit_viocd", "phonert"])
parser.add_argument("--model", type=str, default="CLS",
                    choices=["CLS", "NER"])
parser.add_argument("--lr", type=float, default=0.001,
                    help="Choose learning rate")
parser.add_argument("--num_epochs", type=int, default=10,
                    help="Choose num epochs")
args = parser.parse_args()

lr = args.lr
data_name = args.data
model_name = args.model
num_epochs = args.num_epochs

# ----- Load dataset -----
if data_name == 'uit_viocd':
    num_classes = 4
    task_type = 'CLS'
    vocab = build_vocab_viocd('/workspaces/DS210/Lab_5/data/uit_viocd/train.json')
    train_dataset = uit_viocd('/workspaces/DS210/Lab_5/data/uit_viocd/train.json', vocab)
    val_dataset = uit_viocd('/workspaces/DS210/Lab_5/data/uit_viocd/dev.json', vocab)
    test_dataset = uit_viocd('/workspaces/DS210/Lab_5/data/uit_viocd/test.json', vocab)
    collate_fn = lambda batch: collate_viocd(batch, pad_id=vocab["<PAD>"])
elif data_name == 'phonert':
    task_type = 'NER'
    vocab, tag_vocab = build_vocab_phonert('/workspaces/DS210/Lab_5/data/PhoNERT/train_word.json')
    train_dataset = phonert('/workspaces/DS210/Lab_5/data/PhoNERT/train_word.json', vocab, tag_vocab)
    val_dataset = phonert('/workspaces/DS210/Lab_5/data/PhoNERT/dev_word.json', vocab, tag_vocab)
    test_dataset = phonert('/workspaces/DS210/Lab_5/data/PhoNERT/test_word.json', vocab, tag_vocab)
    collate_fn = lambda batch: collate_phonert(batch, pad_id=vocab["<PAD>"], pad_id_tag=tag_vocab["<PAD>"])
    tag_size = len(tag_vocab)
else:
    print ("Wrong data_name")
    exit()

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# ----- Modeling -----
vocab_size = len(vocab)
if model_name == "CLS":
    model = Transformer_CLS(vocab_size=vocab_size, pad_idx=vocab["<PAD>"], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
elif model_name == "NER":
    model = Transformer_NER(vocab_size=vocab_size, pad_idx=vocab["<PAD>"], num_tags=tag_size)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab["<PAD>"])
else: 
    print ("Wrong model_name")
    exit()

optimizer = optim.Adam(model.parameters(), lr=lr)

# ----- Training -----
save_path = f"model/BestModel/best_model_{model_name}.pt"
train_epoch(model, train_dataloader, task_type, optimizer, criterion,
            num_epochs, val_dataloader, save_path, device)
model.load_state_dict(torch.load(save_path))

# ----- Evaluation -----
print("----- Test Results ----")
if task_type == 'CLS':
   _, test_f1, test_report = evaluate_cls(model, test_dataloader, criterion, device)
   print(f"Test F1 Score: {test_f1:.4f}")
   print("Test Classification Report:")
   print(test_report)
elif task_type == 'NER':
   _, test_f1 = evaluate_ner(model, test_dataloader, tag_vocab['<PAD>'], criterion, device)
   print(f"Test F1 Score: {test_f1:.4f}")

