import json
import torch
from .utils import encode_sentence
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_viocd(batch: list[dict], pad_id) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)

    attention_masks = (padded_input_ids != pad_id).long()
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": padded_input_ids,
        "masks": attention_masks,
        "labels": labels
    }

class uit_viocd(Dataset):
    def __init__(self, file_path, vocab):
        self.vocab = vocab

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.texts = [item["review"] for item in self.data.values()]
        self.domains = [item["domain"] for item in self.data.values()]

        self.label_map = {"app": 0, "fashion": 1, "cosmetic": 2, "mobile": 3}
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_ids = encode_sentence(self.texts[index], self.vocab)
        
        lengths = input_ids.size(0)

        labels = self.label_map[self.domains[index]]
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": labels
        }