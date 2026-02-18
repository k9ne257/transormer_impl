import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 40 # how many independent sequences will we process in parallel?
block_size = 10 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

input = "../data/anything_shakespeare.txt"
with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

# grabbbing unique characters that occur in input data
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: string to int
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: int to string

print("encode:", encode)
print("decode:", decode)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # here we do a 90/10 split
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    bigram_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = bigram_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    bigram_model.train()
    return out

# bigram model (for language modeling validation)
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

bigram_model = BigramLanguageModel(vocab_size)
m = bigram_model.to(device)

# create AdamW optimizer using pytorch
adamWopti = torch.optim.AdamW(bigram_model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # evaluate the loss on train and val sets but not all the time
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    trainingDataBatch, trainingTargetBatch = get_batch('train')

    # evaluation of loss
    logits, loss = bigram_model(trainingDataBatch, trainingTargetBatch)
    adamWopti.zero_grad(set_to_none=True)
    loss.backward()
    adamWopti.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))