import torch
import json
import torch.nn as nn
import json
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import re

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv2d(1, 100, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, embed_dim))

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3 * 100, num_classes)

    def forward(self, x):
        x = self.embedding(x)      # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1)         # (batch_size, 1, seq_len, embed_dim)

        x1 = functional.relu(self.conv1(x)).squeeze(3)
        x2 = functional.relu(self.conv2(x)).squeeze(3)
        x3 = functional.relu(self.conv3(x)).squeeze(3)

        x1 = functional.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = functional.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = functional.max_pool1d(x3, x3.size(2)).squeeze(2)

        out = torch.cat((x1, x2, x3), dim=1)
        out = self.dropout(out)
        return self.fc(out)

class TextDataset(Dataset):
    def __init__(self, json_path, vocab=None, max_len=200):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.texts = [item["text"] for item in data]
        self.labels = [item["label"] for item in data]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = torch.tensor(self.label_encoder.fit_transform(self.labels))

        self.vocab = vocab or self.build_vocab(self.texts)
        self.max_len = max_len
        self.encoded_texts = [self.encode_text(text) for text in self.texts]
    @staticmethod
    def split_dataset(dataset, test_size=0.2):
        train_indices, val_indices = train_test_split(
            list(range(len(dataset))), test_size=test_size, stratify=dataset.encoded_labels
        )
        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)
        return train_set, val_set
    @staticmethod
    def build_vocab(texts):
        counter = Counter(word for text in texts for word in text.split())
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word in counter:
            vocab[word] = len(vocab)
        return vocab

    def encode_text(self, text):
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in text.split()]
        token_ids = token_ids[:self.max_len]
        token_ids += [self.vocab["<PAD>"]] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids)

    @staticmethod
    def evaluate(model, val_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.encoded_labels[idx]
