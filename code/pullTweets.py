import csv
import tweepy
from dotenv import dotenv_values

# Configuration
config = dotenv_values('./config/.env')



"""bearer_token = config["BEARER_TOKEN"]
client = tweepy.Client(bearer_token=bearer_token)"""

# Set up authentication and API connection
auth = tweepy.OAuthHandler(config["API_KEY"], config["API_KEY_SECRET"])
auth.set_access_token(config["ACCESS_TOKEN"], config["ACCESS_TOKEN_SECRET"])
api = tweepy.API(auth)

# Open a CSV file for writing
with open('tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['created_at', 'text', 'favorite_count'])
    # Get the user's tweets and likes
    for status in tweepy.Cursor(api.favorites, screen_name='EvanWoods').items():
        writer.writerow([status.created_at, status.text, status.favorite_count])
    for status in tweepy.Cursor(api.user_timeline, screen_name='EvanWoods').items():
        writer.writerow([status.created_at, status.text, status.favorite_count])

'''
# Get the current list of followers
original_followers = set(friend.screen_name for friend in tweepy.Cursor(api.friends, screen_name='username').items())

# Wait for some time 
import time
time.sleep(3600)

# Get the list of new followers after some time 
new_followers = set(friend.screen_name for friend in tweepy.Cursor(api.friends, screen_name='username').items())

# Get the new followers who started following the user
new_followers_history = new_followers - original_followers

'''


# Open a CSV file for writing
with open('replies_follows.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['screen_name', 'text', 'created_at'])
    # Get the user's replies
    for status in tweepy.Cursor(api.search_tweets, q='to:EvanWoods').items():
        writer.writerow([status.user.screen_name, status.text, status.created_at])
    # Get the user's follows
    for friend in tweepy.Cursor(api.get_friends, screen_name='EvanWoods').items():
        writer.writerow([friend.screen_name, '', ''])

'''
library(caret)
library(graphics)

confusion_matrix <- confusionMatrix(data = predicted_values, reference = actual_values)
plot(confusion_matrix)
'''
