import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from model4 import InterviewTransformer

# === Load Dataset ===
dataset = []
with open("interview_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text = f"Q: {data['question']} A: {data['answer']}"
        dataset.append(text)

# === Tokenisasi Manual ===
char_vocab = sorted(set("".join(dataset)))  # Pastikan urutan tetap konsisten
char_vocab.insert(0, "<PAD>")  # Tambahkan token padding di indeks 0
char_to_idx = {char: idx for idx, char in enumerate(char_vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(char_vocab)

# Simpan vocab agar konsisten dengan inference
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f)

# Tokenisasi dataset ke indeks
max_length = 128
# Gunakan karakter padding unik, misalnya " " (spasi)
PAD_CHAR = " "
char_vocab.insert(0, PAD_CHAR)  # Tambahkan karakter padding di indeks 0
char_to_idx = {char: idx for idx, char in enumerate(char_vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Tokenisasi dataset dengan padding karakter " "
tokenized_data = [
    [char_to_idx[char] for char in text[:max_length].ljust(max_length, PAD_CHAR)]
    for text in dataset
]
tokenized_data = torch.tensor(tokenized_data)

# === Model ===
model = InterviewTransformer(vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#loss_fn = nn.CrossEntropyLoss(ignore_index=char_to_idx["<PAD>"])  # Abaikan padding
loss_fn = nn.CrossEntropyLoss(ignore_index=char_to_idx[" "])  # Abaikan padding saat training


# === DataLoader ===
batch_size = 8
dataset = TensorDataset(tokenized_data[:-1], tokenized_data[1:])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Training ===
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# === Simpan Model ===
torch.save(model.state_dict(), "interviewer_transformer.pth")
print("Model telah disimpan!")
