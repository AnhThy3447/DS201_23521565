import json
import torch
from .utils import encode_sentence
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_vsfc(batch: list[dict], pad_id) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)

    lengths = torch.tensor([item["lengths"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": padded_input_ids,
        "lengths": lengths,
        "labels": labels
    }

class uit_vsfc(Dataset):
    def __init__(self, file_path, vocab):
        self.vocab = vocab

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.texts = [item["sentence"] for item in self.data]
        self.labels = [item["topic"] for item in self.data]

        # Label mapping
        self.label_map = {"lecturer": 0, "training_program": 1, "facility": 2, "others": 3}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Batch encoding
        input_ids = encode_sentence(self.texts[index], self.vocab)
        
        # Lengths
        lengths = input_ids.size(0)

        # Labels
        labels = self.label_map[self.labels[index]]
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": labels
        }
