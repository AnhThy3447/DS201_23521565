import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_utils.uit_vsfc import uit_vsfc, collate_vsfc
from data_utils.phonert import phonert, collate_phonert
from data_utils.utils import build_vocab_vsfc, build_vocab_phonert
from model.GRU import GRU
from model.LSTM import LSTM
from model.BiLSTM import BiLSTM_NER
from train_eval import train_epoch, evaluate_ner, evaluate_cls

# ----- Define parameters -----
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Parse arguments -----
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="uit_vsfc",
                    choices=["uit_vsfc", "phonert"])
parser.add_argument("--model", type=str, default="LSTM",
                    choices=["LSTM", "GRU", "BiLSTM"])
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
if data_name == 'uit_vsfc':
   num_classes = 4
   task_type = 'CLS'
   vocab = build_vocab_vsfc('data/UIT-VSFC/UIT-VSFC-train.json')
   train_dataset = uit_vsfc('data/UIT-VSFC/UIT-VSFC-train.json', vocab)
   val_dataset = uit_vsfc('data/UIT-VSFC/UIT-VSFC-dev.json', vocab)
   test_dataset = uit_vsfc('data/UIT-VSFC/UIT-VSFC-test.json', vocab)
   collate_fn=lambda batch: collate_vsfc(batch, pad_id=vocab["<PAD>"])
elif data_name == 'phonert':
   task_type = 'NER'
   vocab, tag_vocab = build_vocab_phonert('data/PhoNERT/train_word.json')
   train_dataset = phonert('data/PhoNERT/train_word.json', vocab, tag_vocab)
   val_dataset = phonert('data/PhoNERT/dev_word.json', vocab, tag_vocab)
   test_dataset = phonert('data/PhoNERT/test_word.json', vocab, tag_vocab)
   collate_fn=lambda batch: collate_phonert(batch, pad_id=vocab["<PAD>"], pad_id_tag=tag_vocab["<PAD>"])
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
   shuffle=False,
   collate_fn=collate_fn
)

test_dataloader = DataLoader(
   dataset=test_dataset,
   batch_size=batch_size,
   shuffle=False,
   collate_fn=collate_fn
)

# ----- Modeling -----
vocab_size = len(vocab)
if model_name == "LSTM":
   model = LSTM(vocab_size=vocab_size, output_dim=num_classes)
elif model_name == "GRU":
   model = GRU(vocab_size=vocab_size, output_dim=num_classes)
elif model_name == "BiLSTM":
   model = BiLSTM_NER(vocab_size=vocab_size, tag_size=tag_size)
else:
   print ("Wrong model_name")
   exit()

if task_type == 'CLS':
   criterion = nn.CrossEntropyLoss()
elif task_type == 'NER':
   criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab['<PAD>'])
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