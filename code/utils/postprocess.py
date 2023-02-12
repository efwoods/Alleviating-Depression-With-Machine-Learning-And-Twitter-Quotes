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
    stripped_spaces_unique_spellchecked_list = [x.strip() for x in generated_personality_list]
    return stripped_spaces_unique_spellchecked_list
        

        
        
    
