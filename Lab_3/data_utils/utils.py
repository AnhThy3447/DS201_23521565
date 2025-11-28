import json
import torch
import string

def preprocess(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(translator)
    return text

def build_vocab_vsfc(file_path):
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

def build_vocab_phonert(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(item)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    tag_vocab = {"<PAD>": -100}
    for item in data:
        for word in item['words']:
            if word not in vocab:
                vocab[word] = len(vocab)
        for tag in item['tags']:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
    return vocab, tag_vocab