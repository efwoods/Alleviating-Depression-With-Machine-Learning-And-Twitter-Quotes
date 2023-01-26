import tweepy
from dotenv import dotenv_values

config = dotenv_values('./config/.env')
# Authenticate with Twitter API
auth = tweepy.OAuthHandler(config['API_KEY'], config['API_KEY_SECRET'])
auth.set_access_token(config['ACCESS_TOKEN'], config['ACCESS_TOKEN_SECRET'])
api = tweepy.API(auth)

# Get list of the authenticated user's favorite tweets
favorites = api.favorites()

# Print the text of each tweet
for favorite in favorites:
    print(favorite.text)
