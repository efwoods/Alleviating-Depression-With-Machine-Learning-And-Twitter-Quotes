# Configuration

import base64
import hashlib
import os
import re
import json
import requests
import redis
from requests.auth import AuthBase, HTTPBasicAuth
from requests_oauthlib import OAuth2Session, TokenUpdated
from flask import Flask, request, redirect, session, url_for, render_template
from dotenv import dotenv_values
import random
import configparser
import pickle
import pandas as pd
import sklearn
import tweepy
from langdetect import detect


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import nltk
# Importing word tokenize
from nltk import word_tokenize
from nltk.corpus import wordnet
# Importing word stopwords

nltk.download('wordnet')
nltk.download('omw-1.4')
# Downlaod the punkt for punctuation
nltk.download("punkt")
# Downlaod the stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
  
from nltk.stem import WordNetLemmatizer

# Importing word TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# Configuration
config = dotenv_values('./config/.env')
auth = tweepy.OAuthHandler(config["API_KEY"], config["API_KEY_SECRET"])
auth.set_access_token(config["ACCESS_TOKEN"], config["ACCESS_TOKEN_SECRET"])
api = tweepy.API(auth)
bearer_token = config["BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)

# Now we can set the permissions you need for your bot by defining scopes. You can use the authentication mapping guide to determine what scopes you need based on your endpoints. 
scopes = ["tweet.read", "users.read", "tweet.write", "offline.access"]

# Since Twitter’s implementation of OAuth 2.0 is PKCE-compliant, you will need to set a code verifier. This is a secure random string. This code verifier is also used to create the code challenge.
code_verifier = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8")
code_verifier = re.sub("[^a-zA-Z0-9]+", "", code_verifier)

# In addition to a code verifier, you will also need to pass a code challenge. The code challenge is a base64 encoded string of the SHA256 hash of the code verifier.
code_challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
code_challenge = base64.urlsafe_b64encode(code_challenge).decode("utf-8")
code_challenge = code_challenge.replace("=", "")

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
'''
def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText
'''
## Get quotes
df = pd.read_csv('./input/quotes-from-goodread/all_quotes.csv')
df['Quote'] = df['Quote'].apply(lambda x: re.sub("[\“\”]", "", x))
df['Other Tags'] = df['Other Tags'].apply(lambda x: re.sub("[\'\[\]]", "", x))

# Detect Text Language
langs = []
for text in df['Quote']:
    try:
        lang = detect(text).language.code
        langs.append(lang)
    except:
        lang = 'NaN'
        langs.append(lang)
df['lang'] = langs


def tokenize(text):
    
    # Making each letter as lowercase and removing non-alphabetical text
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    
    # Extracting each word in the text
    tokens = word_tokenize(text)
    
    # Removing stopwords
    words = [word for word in tokens if word not in stopwords.words("english")]
    
    # Lemmatize the words
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]
    
    tweetwords = ''
    for word in text_lems:
        # concatonate into a string
        if len(word)>1:
            tweetwords += (word+' ')
    return tweetwords



# Detect Text Language
langs = []
for text in df['Quote']:
    try:
        lang = detect(text)
        langs.append(lang)
    except:
        lang = 'NaN'
        langs.append(lang)
df['lang'] = langs

df['lang'].value_counts().head(10)

# I will only use Eng Quote
df_eng = df[df['lang']=='en']
print(df_eng.shape)
df_eng['Main Tag'].value_counts()

# Preprocessing
# lower case
df_eng['CleanQuote'] = df_eng['Quote'].apply(lambda x: x.lower())
df_eng['CleanQuote'].sample(2)

# remove stopwords and punctuation
stop_nltk = stopwords.words("english")

def drop_stop(input_tokens):
    rempunc = re.sub(r'[^\w\s]','',input_tokens)
    remstopword = " ".join([word for word in str(rempunc).split() if word not in stop_nltk])
    return remstopword

df_eng['CleanQuote'] = df_eng['CleanQuote'].apply(lambda x: drop_stop(x))
df_eng['CleanQuote'].sample(2)

# Tokenize
df_eng['CleanQuote'] = df_eng['CleanQuote'].apply(lambda x: tokenize(x))

# drop empty rows
df_eng.drop(df_eng[df_eng['CleanQuote']==""].index, inplace=True)

###
# specify feature and target
X = df_eng['CleanQuote']
Y = df_eng['Main Tag']

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=48, stratify=Y)
print(f"X_train : {X_train.shape}\nX_test : {X_test.shape}")

# Extract features
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Create a model (SGDClassifier)
pipeline_sgd = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42)),
                        ])

model_sgd = pipeline_sgd.fit(X_train, Y_train)

predict_sgd = model_sgd.predict(X_test)

print(classification_report(predict_sgd, Y_test))