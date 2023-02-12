# Libraries
import tweepy
import pandas as pd
from dotenv import dotenv_values
from utils import preprocess

# Configuration
def config():
    config = dotenv_values('./config/.env')
    # Authenticate with Twitter API
    auth = tweepy.OAuthHandler(config['API_KEY'], config['API_KEY_SECRET'])
    auth.set_access_token(config['ACCESS_TOKEN'], config['ACCESS_TOKEN_SECRET'])
    api = tweepy.API(auth)

    # Authentication used for getting depressed tweets
    bearer_token = config["BEARER_TOKEN"]
    client = tweepy.Client(bearer_token=bearer_token)
    return api, client

# Methods
# get the username from the tweetID
def get_twitter_username_from_tweetID(api, tweetID):
    twitter_data = api.get_status(tweetID)
    username = twitter_data.user.screen_name
    return username

def get_users_favorite_tweets(api, username="EvanWoods"):
    # Get list of the authenticated user's favorite tweets
    favorites = api.get_favorites(screen_name=username,count=20)
    tweets = preprocess.preprocess_tweet(favorites)
    # Print the text of each tweet
    for tweet in tweets:
        print(tweet+'\n')
    return tweets

def get_tweets_by_hashtag(client, hashtag='#depressed',number_of_tweets_to_get=10,):
    # get tweets by hashtag: defaults to "depressed"
    query = hashtag
    tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=number_of_tweets_to_get)
    df = pd.DataFrame(tweets.data, columns=["id","text"])
    text_l = []
    for text in df["text"]:
        text_l.append(text)
    return df,text_l

def get_my_tweets(api):
    timeline = api.user_timeline(count = 1000, max_id = None, tweet_mode = 'extended')
    tweets = preprocess.tweet_quote(timeline)
    return tweets
