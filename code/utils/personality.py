import joblib
import torch
import pickle
import pandas as pd

from autocorrect import Speller
spell = Speller(lang='en')

def correct_spelling(generated_personality_list, delimiter):
    spellchecked_list = []
    if delimiter == 'spaces':
        sentences = generated_personality_list.split('.')
    elif delimiter == 'newlines':
        sentences = generated_personality_list.split('\n')
    else:
        print('Error: define delimiter = spaces | newlines')
        exit()
        
    for sentence in sentences:
        spellchecked_list.append(spell(sentence))
    stripped_spaces_unique_spellchecked_list = list(set(spellchecked_list))
    # stripped_spaces_unique_spellchecked_list = [x.strip() for x in generated_personality_list]
    return stripped_spaces_unique_spellchecked_list

def generate_personality_list(text, generated_personality):
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = joblib.load('./models/personality_newlines_corpus.joblib')

    model = joblib.load('./models/personality_newlines_model.joblib')
    # # Load the model from a binary file
    # with open("./models/personality_newlines_model.joblib", "rb") as f:
    #     model = pickle.load(f)

    chars = sorted(list(set(text)))
    # create a mapping from characters to integers
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_personality = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
        '''
    generated_personality_list_correct_spelling = correct_spelling(generated_personality,'newlines')

    generated_personality_list_correct_spelling_stripped_spaces = [x.strip() for x in generated_personality_list_correct_spelling]
    unique_generated_personality_list = list(set(generated_personality_list_correct_spelling_stripped_spaces))
    
    for i in range(0, len(unique_generated_personality_list)-1):
        if unique_generated_personality_list[i] == '':
            del unique_generated_personality_list[i]
    
    generated_personality_list_Series = pd.Series(unique_generated_personality_list)
    return generated_personality_list_Series
