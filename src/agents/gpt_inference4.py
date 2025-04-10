import torch
import json
from model4 import InterviewTransformer

# === Load Vocab ===
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}  # Pastikan kunci integer

vocab_size = len(char_to_idx)

# === Load Model ===
model = InterviewTransformer(vocab_size)
model.load_state_dict(torch.load("interviewer_transformer.pth"))
model.eval()

# === Fungsi Inferensi ===
def generate_text(prompt, max_length=50):
    input_ids = [char_to_idx.get(char, char_to_idx["<PAD>"]) for char in prompt]
    input_tensor = torch.tensor([input_ids])

    generated_text = prompt
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor)
        
        predicted_id = torch.argmax(output[:, -1, :], dim=-1).item()
        generated_char = idx_to_char.get(predicted_id, "?")  # Default ke "?" jika tidak ditemukan

        generated_text += generated_char
        input_ids.append(predicted_id)
        input_tensor = torch.tensor([input_ids[-max_length:]])

    return generated_text

# === Contoh Inferensi ===
print(generate_text("Q: Bisa ceritakan tentang proyek terbesar Anda? A:"))
