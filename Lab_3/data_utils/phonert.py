import torch 
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
