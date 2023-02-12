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
    

# Get tweets
api, client = twitter.config()
vectoriser, LRmodel, model_sgd = load_models()

# Loading the models & getting quote list.
# Creating a dataframe of my favorite quotes to suggest based on class

favorite_quotes_classes = tweet_class_df(db_Series_of_tweets_or_strings = getFavoriteQuotes())
personality_classes = tweet_class_df(db_Series_of_tweets_or_strings=personality.generate_personality_list())


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
