# Imports
import requests
import pickle
import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import personality

from tqdm import tqdm
import joblib
from utils import twitter,preprocess

# Methods
# Classify tweets based on category (love, etc)
def identify_classes(tweets):
    classes = model_sgd.predict(tweets)
    print(classes)
    return classes

# Identify the mode of the class that is liked by the user
def identify_mode_class(classes):
    mode_class = pd.DataFrame(classes).value_counts().head(1)
    mode_class_name_index = mode_class.index.get_level_values(0)
    mode_class_name = mode_class_name_index[0]
    return mode_class_name

# Load models
def load_models():
    # Load the vectoriser.
    file = open('./models/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./models/Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    model_sgd = joblib.load('./models/model_sgd.joblib')
    return vectoriser, LRmodel, model_sgd

# pass into a cosine similarity matrix based on the mode class of the user's tweet
def recommendQuotedResponse(quotesMasterDB, depressedTweet):
    quotesTEMP = quotesMasterDB.copy(deep=True)
    quotesTEMP.loc[-1] = depressedTweet
    quotesTEMP.index += 1
    quotesTEMP = quotesTEMP.sort_index()
    # Create the TfidfVectorizer
    tfidf = TfidfVectorizer(tokenizer = preprocess.tokenize)
    quotes_tfidf = tfidf.fit_transform(quotesTEMP.values).toarray()
    similar_quote = cosine_similarity(quotes_tfidf, quotes_tfidf)
    idx = 0
    quote_series = pd.Series(similar_quote[idx]).sort_values(ascending = False)
    top_10_indexes = list(quote_series.iloc[1 : 11].index)
    return quotesTEMP.iloc[top_10_indexes[0]]


# Predict the Sentiment of a tweet (Positive/Negative)
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess.preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

# Get favorite quotes
# Used to feed into tweet class dataframe 
def getFavoriteQuotes():
    quotes_url = "https://efwoods.github.io/EvanWoodsFavoriteQuotes/quotesTwitterDB.json"
    quotesDB = requests.request("GET", quotes_url).json()
    quotesMasterDB = pd.Series(quotesDB["quotes"])
    return quotesMasterDB

def tweet_class_df(db_Series_of_tweets_or_strings):
    classes = identify_classes(db_Series_of_tweets_or_strings)
    tweet_class_df = pd.DataFrame({'quote': db_Series_of_tweets_or_strings, 'category': classes})
    return tweet_class_df

def create_subset(series_of_classes_and_text_either_favorite_or_personality, mode_class):
    return series_of_classes_and_text_either_favorite_or_personality[series_of_classes_and_text_either_favorite_or_personality['category']==mode_class]
    

### create_model
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import twitter

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('evanwoods.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

api, config = twitter.config()
text = twitter.get_my_tweets(api)
temp = ""
for t in text:
    temp = temp + "\n" + t
text = temp

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_personality = decode(m.generate(context, max_new_tokens=2000)[0].tolist())


# Get tweets
api, client = twitter.config()
vectoriser, LRmodel, model_sgd = load_models()

# Loading the models & getting quote list.
# Creating a dataframe of my favorite quotes to suggest based on class

favorite_quotes_classes = tweet_class_df(db_Series_of_tweets_or_strings = getFavoriteQuotes())
personality_classes = tweet_class_df(db_Series_of_tweets_or_strings=personality.generate_personality_list(text,generated_personality))


df, text_l = twitter.get_tweets_by_hashtag(client)

# make a prediction of the positive and negative sentiment of the depressed tweet
preprocessed_df = predict(vectoriser, LRmodel, preprocess.preprocess(text_l))

# identify response & respond
pbar = tqdm(total=preprocessed_df["text"].size)
for text in range(0,preprocessed_df["text"].size):
    tweetid = str(df["id"][text])
    if preprocessed_df["sentiment"][text] == "Negative":
        f = open('hugs_given.txt',"r")
        file_contents = f.read()
        file_contents = file_contents.splitlines()
        f.close()
        if not tweetid in file_contents:
            print("*give hug* to:{}".format(df["id"][text]))
            f = open('hugs_given.txt','a')
            f.write(tweetid + "\n")
            f.close()
            try:
                tweetid = str(df["id"][text])
                print(tweetid)
                print(preprocessed_df["sentiment"][text])
                print(preprocessed_df["text"][text])
                print(df["text"][text])
                print('\n')
                username = twitter.get_twitter_username_from_tweetID(api,df["id"][text])
                
                # Identify which category is most liked by the user
                tweets = twitter.get_users_favorite_tweets(api, username)

                # Make a prediction
                user_classes = identify_classes(tweets)
                mode_class = identify_mode_class(user_classes)

                # Generate a quote based off of the class the user would most like. 

                # Create a subset of my favorite quotes based off of the category that is most liked by the user
                
                quotes_subset = create_subset(personality_classes, mode_class)
                responseTweet = recommendQuotedResponse(quotes_subset['quote'], preprocessed_df["text"][text])
                medicine = '*gives hug* ' + responseTweet
                print(medicine)
                api.update_status(status = medicine, in_reply_to_status_id = tweetid , auto_populate_reply_metadata=True)
            except Exception:
                pass
        else:
            print("tweetId in file contents")
    else:
        print("positive sentiment")
    pbar.update(1)
pbar.close()
