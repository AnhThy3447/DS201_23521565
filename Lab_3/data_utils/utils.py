import json
import torch
import string

def preprocess(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(translator)
    return text

def build_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = [item["sentence"] for item in data]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sentence in sentences:
        sentence = preprocess(sentence)
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode_sentence(sentence, vocab):
    sentence = preprocess(sentence)
    words = sentence.split()
    word_id = [vocab.get(word, vocab["<UNK>"]) for word in words]
    return torch.tensor(word_id, dtype=torch.long)