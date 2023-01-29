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

def preprocess_tweet(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for unformatted_tweet in textdata:
        tweet = unformatted_tweet.text
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

## Get tweets
import tweepy
from dotenv import dotenv_values

config = dotenv_values('./config/.env')
# Authenticate with Twitter API
auth = tweepy.OAuthHandler(config['API_KEY'], config['API_KEY_SECRET'])
auth.set_access_token(config['ACCESS_TOKEN'], config['ACCESS_TOKEN_SECRET'])
api = tweepy.API(auth)

# for getting depressed tweets
bearer_token = config["BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)


def get_users_favorite_tweets(username="EvanWoods"):
    # Get list of the authenticated user's favorite tweets
    favorites = api.get_favorites(screen_name=username,count=20)
    tweets = preprocess_tweet(favorites)
    # Print the text of each tweet
    for tweet in tweets:
        print(tweet+'\n')
    return tweets

## make a prediction
def identify_classes(tweets):
    classes = model_sgd.predict(tweets)
    print(classes)
    return classes

## identify the mode class that is liked by the user
def identify_mode_class(classes):
    mode_class = pd.DataFrame(classes).value_counts().head(1)
    mode_class_name_index = mode_class.index.get_level_values(0)
    mode_class_name = mode_class_name_index[0]
    return mode_class_name

#create a category for my favorite quotes
## load models
def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('./models/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./models/Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

# Loading the models & getting quote list.
# Creating a dataframe of my favorite quotes to suggest based on class
vectoriser, LRmodel = load_models()

quotes_url = "https://efwoods.github.io/EvanWoodsFavoriteQuotes/quotesTwitterDB.json"
quotesDB = requests.request("GET", quotes_url).json()
quotesMasterDB = pd.Series(quotesDB["quotes"])
classes = identify_classes(quotesMasterDB)
favorite_quotes_classes = pd.DataFrame({'quote': quotesMasterDB,'category': classes})

##


# identify which category is most liked by the user
tweets = get_users_favorite_tweets('lexfridman')

## make a prediction
user_classes = identify_classes(tweets)

mode_class = identify_mode_class(user_classes)

## create a subset of my favorite quotes based off of the category that is most liked by the user
my_favorite_quotes_subset = favorite_quotes_classes[favorite_quotes_classes['category']==mode_class]

# pass into a cosine similarity matrix based on the mode class of the user's tweet
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
    return quotesTEMP.iloc[top_10_indexes[0]]

# get depressed tweets
num_of_people_to_hug = 100
query = '#depressed'
tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=num_of_people_to_hug)
df = pd.DataFrame(tweets.data, columns=["id","text"])
text_l = []
for text in df["text"]:
    text_l.append(text)


# run a prediction
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

# make a prediction of the positive and negative sentiment of the depressed tweet
preprocessed_df = predict(vectoriser, LRmodel, preprocess(text_l))
# get the username from the tweetID
def get_twitter_username_from_tweetID(tweetID):
    twitter_data = api.get_status(tweetID)
    username = twitter_data.user.screen_name
    return username
    
# identify response & respond
from tqdm import tqdm
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
                username = get_twitter_username_from_tweetID(df["id"][text])
                # identify which category is most liked by the user
                tweets = get_users_favorite_tweets(username)

                ## make a prediction
                user_classes = identify_classes(tweets)

                mode_class = identify_mode_class(user_classes)

                ## create a subset of my favorite quotes based off of the category that is most liked by the user
                my_favorite_quotes_subset = favorite_quotes_classes[favorite_quotes_classes['category']==mode_class]
                responseTweet = recommendQuotedResponse(my_favorite_quotes_subset['quote'], preprocessed_df["text"][text])
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
