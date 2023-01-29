
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

# nltk
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
  
from nltk.stem import WordNetLemmatizer

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