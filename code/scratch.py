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
nltk.download('wordnet')
nltk.download('omw-1.4')
  
from nltk.stem import WordNetLemmatizer

config = dotenv_values('./config/.env')
bearer_token = config["BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)
query = '#depressed'
tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=10)

df = pd.DataFrame(tweets.data, columns=["id","text"])


auth = tweepy.OAuthHandler(config["API_KEY"], config["API_KEY_SECRET"])
auth.set_access_token(config["ACCESS_TOKEN"], config["ACCESS_TOKEN_SECRET"])
api = tweepy.API(auth)

text_l = []
for text in df["text"]:
    text_l.append(text)


###
query = '#depressed'
tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=100)

df = pd.DataFrame(tweets.data, columns=["id","text"])

text_l = []
for text in df["text"]:
    text_l.append(text)
    
# Loading the models.
vectoriser, LRmodel = load_models()
    

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
                api.update_status(status = '*gives hug*', in_reply_to_status_id = tweetid , auto_populate_reply_metadata=True)
            except Exception:
                pass