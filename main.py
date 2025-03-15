import importlib
import matplotlib.pyplot as plt
import tiktoken
import T1000
importlib.reload(T1000)  # Force reload of T1000
from T1000 import *
from utility import get_gutenberg_book, get_many_books, create_tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, sequences: list[list[int]], max_length: int = None):
        self.data = []
        self.targets = []
        for sequence in sequences:
            if len(sequence) <= 1:
                continue
            
            if max_length is not None and max_length > 0:
                if len(sequence) > max_length:
                    for sub_sequence in [sequence[i:i+max_length] for i in range(0, len(sequence), max_length)]:
                        self.data.append(sub_sequence[:-1])
                        self.targets.append(sub_sequence[1:])
                    continue

            self.data.append(sequence[:-1])
            self.targets.append(sequence[1:])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.long),
                torch.tensor(self.targets[idx], dtype=torch.long))
    
    

def train_transformer(
    model: Transformer,
    #int_sequences: list[list[int]],
    tokenizer: tiktoken.Encoding,
    max_sequence_length: int = 1000,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = model.to(device)
    dataset = TextDataset(int_sequences, max_length=max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_loss_values = []
    
    print(f"Training on {len(dataloader)} batches per epoch")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # inputs: [32, 10], targets: [32, 10]
            optimizer.zero_grad()  # Reset gradients for the batch
            
            # Process each sequence in the batch individually
            batch_loss = 0
            for i in range(inputs.shape[0]):  # Loop over batch_size (32)
                single_input = inputs[i].to(device)   # [10]
                single_target = targets[i].to(device) # [10]
                single_output = model(single_input)   # [10, 10000]
                loss = criterion(single_output, single_target)
                loss.backward()  # Accumulate gradients
                batch_loss += loss.item()
            batch_loss_values.append(batch_loss)

            # Clip gradients and update weights once per batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += batch_loss / batch_size  # Average loss over batch
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Avg Loss: {batch_loss / batch_size:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
    
    return batch_loss_values


def create_word_to_int_mapping(texts: list[str], max_vocab_size: int = 10000) -> tuple[dict[str, int], list[list[int]]]:
    word_counts = {}
    for text in texts:
        words = text.split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size - 1]
    vocab_words = [word for word, _ in vocab]
    
    word_to_int = {"<UNK>": 0}
    for i, word in enumerate(vocab_words, 1):
        word_to_int[word] = i
    
    int_sequences = []
    for text in texts:
        words = text.split()
        int_sequence = [word_to_int.get(word, 0) for word in words]
        int_sequences.append(int_sequence)
    
    return word_to_int, int_sequences


DATA_RAW: list[str] = get_many_books([84, 15, 18, 82, 996, 2600])
print(f"{sum(len(x) for x in DATA_RAW) = }")

'''
word_to_int, int_sequences = create_word_to_int_mapping(DATA_RAW, max_vocab_size=10000)
max_token = max(max(seq) for seq in int_sequences if seq)
print(f"Vocabulary size: {len(word_to_int)}")
print(f"Max token value in int_sequences: {max_token}")
if max_token >= 10000:
    raise ValueError(f"Max token {max_token} exceeds expected vocab size 10000")
word_to_int["the"], int_sequences[0][:10]'
'''
tokenizer, d_vocab = create_tokenizer()
int_sequences = [tokenizer.encode(text) for text in DATA_RAW]

# Initialize and train
config = GPTConfig(d_vocab=d_vocab)
model = Transformer(config)
print(f"Model vocab size: {model.embedding.num_embeddings}")
batch_losses = train_transformer(model, int_sequences, num_epochs=1, batch_size=1)
plt.plot(batch_losses)
plt.title("Batch Loss Values")
plt.xlabel("Batch")
plt.ylabel("Cross Entropy Loss")
plt.savefig("batch_loss_values.png")
torch.save(model, "model.pt")
