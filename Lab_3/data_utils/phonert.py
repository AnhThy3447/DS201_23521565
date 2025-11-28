import json
import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_phonert(batch: list[dict], pad_id, pad_id_tag) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    tag_ids = [item["tag_ids"] for item in batch]
    lengths = torch.tensor([item["lengths"] for item in batch])

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    padded_tag_ids = pad_sequence(tag_ids, batch_first=True, padding_value=pad_id_tag)

    return {
        "input_ids": padded_input_ids,
        "lengths": lengths,
        "labels": padded_tag_ids
    }

class phonert(Dataset):
    def __init__(self, file_path, vocab, tag_vocab):
        self.data = []
        self.vocab = vocab
        self.tag_vocab = tag_vocab

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(item)
        self.texts = [item["words"] for item in self.data]
        self.tags = [item["tags"] for item in self.data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Batch encoding
        input_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in self.texts[index]]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Lengths
        lengths = input_ids.size(0)

        # Tags
        tag_ids = [self.tag_vocab[tag] for tag in self.tags[index]]
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "tag_ids": tag_ids
        }