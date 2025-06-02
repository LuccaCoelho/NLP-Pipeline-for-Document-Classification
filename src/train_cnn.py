import os
import torch
import torch.nn as nn
import torch.optim as optim
from spacy.cli import evaluate
from torch.utils.data import DataLoader
from text_data_set import TextDataset, TextCNN


def train_with_validation(json_path, model_dir="models", num_epochs=10, batch_size=32):
    # Load data
    full_dataset = TextDataset(json_path)
    train_set, val_set = TextDataset.split_dataset(dataset=full_dataset, test_size=0.2)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model, optimizer, loss
    model = TextCNN(
        vocab_size=len(full_dataset.vocab),
        embed_dim=100,
        num_classes=len(set(full_dataset.labels))
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join("/home/lucca-coelho/nlp-pipeline/models", "best_model.pt")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = TextDataset.evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"📦 Best model saved at epoch {epoch+1} → {best_model_path}")

    return best_model_path

if __name__ == "__main__":
    json_path = "/home/lucca-coelho/nlp-pipeline/data/processed_data.json"
    train_with_validation(json_path)
