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

import nltk
# Importing word tokenize
from nltk import word_tokenize

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
from sklearn.metrics.pairwise import cosine_similarity




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


def tokenize(text):
    
    # Making each letter as lowercase and removing non-alphabetical text
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    
    # Extracting each word in the text
    tokens = word_tokenize(text)
    
    # Removing stopwords
    words = [word for word in tokens if word not in stopwords.words("english")]
    
    # Lemmatize the words
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]

    return text_lems

# Methods
def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

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

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

# To connect to manage Tweets endpoint, you’ll need an access token. To create this access token, you can create a function called make_token which will pass in the needed parameters and return a token.
def make_token():
    return OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scopes)

# Since your bot will Tweet random facts about cats, you will need to get these from somewhere. There is a cat fact API that you can call to get facts to Tweet. The function parse_fav_quote allows you to make a GET request to the cat fact endpoint and format the JSON response to get a fact you can later Tweet.
def parse_fav_quote():
    url = "https://efwoods.github.io/EvanWoodsFavoriteQuotes/quotesTwitterDB.json"
    fav_quote = requests.request("GET", url).json()
    quote = random.randint(0, len(fav_quote["quotes"]))
    return fav_quote["quotes"][quote]

# To Tweet the cat fact, you can make a function that will indicate it is Tweeting which helps debug and makes a POST request to the Manage Tweets endpoint.
def post_tweet(payload, token):
    print("Tweeting!")
    return requests.request(
        "POST",
        "https://api.twitter.com/2/tweets",
        json=payload,
        headers={
            "Authorization": "Bearer {}".format(token["access_token"]),
            "Content-Type": "application/json",
        },
    )

    
def get_prior_tweets():
    url = "https://api.twitter.com/2/users/1537504318496047106/tweets?max_results=100"
    prev_quotes = requests.request("GET", url).json()
    return prev_quotes
   # search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
#query_params = {'query': '#depressed','tweet.fields': 'author_id'}

def get_depressed_tweets():
    url = "https://api.twitter.com/2/tweets/search/recent?query=%23depressed"
    depressed_tweets = requests.request("GET", url).json()
    return depressed_tweets

def recommendQuotedResponse(quotesMasterDB, depressedTweet):
    quotesTEMP = quotesMasterDB.copy(deep=True)
    quotesTEMP.loc[-1] = depressedTweet
    quotesTEMP.index += 1
    quotesTEMP = quotesTEMP.sort_index()
    # Create the TfidfVectorizer
    tfidf = TfidfVectorizer(tokenizer = tokenize)
    quotes_tfidf = tfidf.fit_transform(quotesTEMP.values).toarray()
    similar_quote = cosine_similarity(quotes_tfidf, quotes_tfidf)
    idx = 0
    quote_series = pd.Series(similar_quote[idx]).sort_values(ascending = False)
    top_10_indexes = list(quote_series.iloc[1 : 11].index)
    return quotesTEMP.loc[top_10_indexes[0]]
    
# Configuration
config = dotenv_values('./config/.env')
auth = tweepy.OAuthHandler(config["API_KEY"], config["API_KEY_SECRET"])
auth.set_access_token(config["ACCESS_TOKEN"], config["ACCESS_TOKEN_SECRET"])
api = tweepy.API(auth)
bearer_token = config["BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)

# Loading the models.
vectoriser, LRmodel = load_models()

quotes_url = "https://efwoods.github.io/EvanWoodsFavoriteQuotes/quotesTwitterDB.json"
quotesDB = requests.request("GET", quotes_url).json()
quotesMasterDB = pd.Series(quotesDB["quotes"])

# Query

num_of_people_to_hug = 100
query = '#depressed'
tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=num_of_people_to_hug)
df = pd.DataFrame(tweets.data, columns=["id","text"])
text_l = []
for text in df["text"]:
    text_l.append(text)
    
# Predict and respond
testdf = predict(vectoriser, LRmodel, text_l)
for text in range(0,testdf["text"].size):
    tweetid = str(df["id"][text])
    if testdf["sentiment"][text] == "Negative":
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
                recommendation = recommendQuotedResponse(quotesMasterDB, df["text"][text])
                medicine = '*gives hug* ' + recommendation
                api.update_status(status = medicine, in_reply_to_status_id = tweetid , auto_populate_reply_metadata=True)
            except Exception:
                pass