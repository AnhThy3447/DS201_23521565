import json
import torch
from .utils import encode_sentence
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: list[dict], pad_id) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)

    lengths = torch.tensor([item["lengths"] for item in batch])
    sentiments = torch.stack([item["sentiment"] for item in batch])
    topics = torch.stack([item["topic"] for item in batch])

    return {
        "input_ids": padded_input_ids,
        "lengths": lengths,
        "sentiment": sentiments,
        "topic": topics
    }

class uit_vsfc(Dataset):
    def __init__(self, file_path, vocab):
        self.vocab = vocab

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.texts = [item["sentence"] for item in self.data]
        self.sentiments = [item["sentiment"] for item in self.data]
        self.topics = [item["topic"] for item in self.data]

        # Label mapping
        self.sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.topic_map = {"lecturer": 0, "training_program": 1, "facility": 2, "others": 3}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Batch encoding
        input_ids = encode_sentence(self.texts[index], self.vocab)
        
        # Lengths
        lengths = input_ids.size(0)

        # Labels
        sentiment = self.sentiment_map[self.sentiments[index]]
        sentiment = torch.tensor(sentiment, dtype=torch.long)
        topic = self.topic_map[self.topics[index]]
        topic = torch.tensor(topic, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "sentiment": sentiment,
            "topic": topic
        }
