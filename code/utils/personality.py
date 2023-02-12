import joblib
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
text = joblib.load('./models/personality_newline_corpus.joblib')
personality = joblib.load('./models/personality_newline.joblib')

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(personality.generate(context, max_new_tokens=2000)[0].tolist()))