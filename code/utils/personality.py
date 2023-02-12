import joblib
import torch
import pickle
from utils import postprocess
import pandas as pd

def generate_personality_list():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = joblib.load('./models/personality_newlines_corpus.joblib')

    # Load the model from a binary file
    with open("./models/personality_newlines_model.pkl", "rb") as f:
        personality = pickle.load(f)

    chars = sorted(list(set(text)))
    # create a mapping from characters to integers
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_personality = decode(personality.generate(context, max_new_tokens=2000)[0].tolist())
        
    generated_personality_list = postprocess.correct_spelling(generated_personality,'newlines')

    generated_personality_list = [x.strip() for x in generated_personality_list]
    generated_personality_list = list(set(generated_personality_list))
    for i in range(0, len(generated_personality_list)-1):
        if generated_personality_list[i] == '':
            del generated_personality_list[i]
    generated_personality_list_Series = pd.Series(generated_personality_list)
    return generated_personality_list_Series
