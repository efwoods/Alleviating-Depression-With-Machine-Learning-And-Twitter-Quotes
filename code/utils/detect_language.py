from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import gradio as gr

model_path = "jb2k/bert-base-multilingual-cased-language-detection"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

language_dict = {0: 'Arabic',
 1: 'Basque',
 2: 'Breton',
 3: 'Catalan',
 4: 'Chinese_China',
 5: 'Chinese_Hongkong',
 6: 'Chinese_Taiwan',
 7: 'Chuvash',
 8: 'Czech',
 9: 'Dhivehi',
 10: 'Dutch',
 11: 'English',
 12: 'Esperanto',
 13: 'Estonian',
 14: 'French',
 15: 'Frisian',
 16: 'Georgian',
 17: 'German',
 18: 'Greek',
 19: 'Hakha_Chin',
 20: 'Indonesian',
 21: 'Interlingua',
 22: 'Italian',
 23: 'Japanese',
 24: 'Kabyle',
 25: 'Kinyarwanda',
 26: 'Kyrgyz',
 27: 'Latvian',
 28: 'Maltese',
 29: 'Mongolian',
 30: 'Persian',
 31: 'Polish',
 32: 'Portuguese',
 33: 'Romanian',
 34: 'Romansh_Sursilvan',
 35: 'Russian',
 36: 'Sakha',
 37: 'Slovenian',
 38: 'Spanish',
 39: 'Swedish',
 40: 'Tamil',
 41: 'Tatar',
 42: 'Turkish',
 43: 'Ukranian',
 44: 'Welsh'}

examples = ['Transformers are really cool!', 'Трансформеры действительно классные!', '¡Los transformadores son realmente geniales!']

def inference(sentence):
  tokenized_sentence = tokenizer(sentence, return_tensors='pt')
  output = model(**tokenized_sentence)
  predictions = torch.nn.functional.softmax(output.logits, dim=-1)
  certainy, highest_value = torch.max(predictions, dim=-1, keepdim=False,  out=None)
  highest_value_int = highest_value.item()
  language = language_dict[highest_value_int]
  return language

if __name__ == '__main__':
  interFace = gr.Interface(fn=inference,
                              inputs=gr.inputs.Textbox(placeholder="Enter text here", label="Text content", lines=5),
                              outputs=gr.outputs.Label(num_top_classes=6, label="Language of this text is "),
                              verbose=True,
                              examples = examples,
                              title="Language Detector",
                              description="Language detector with support for 45 languages. Created as part of the huggingface course community event.",
                              theme="grass")
  interFace.launch()