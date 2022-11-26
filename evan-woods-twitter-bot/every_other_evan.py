import main
import redis
import json
import os
from dotenv import dotenv_values

config = dotenv_values(".env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}
# You can now set a variable called twitter, which calls the make_token function to create a new access token. You will also need to obtain the environment variables for client ID and client secret. 
twitter = main.make_token()
client_id = config["CLIENT_ID"]
client_secret = config["CLIENT_SECRET"]
token_url = "https://api.twitter.com/2/oauth2/token"

# Now, you can obtain the access token from Redis, which is saved corresponding with the value of token. You will also need to decode the token and replace the quotes. You can save it into a JSON object and work with it later.
t = main.r.get("token")
bb_t = t.decode("utf8").replace("'", '"')
data = json.loads(bb_t)

# Since access tokens in OAuth 2.0 only stay valid for two hours, you will need to refresh your token. Refresh tokens typically stay valid for about six months.
refreshed_token = twitter.refresh_token(
    client_id=client_id,
    client_secret=client_secret,
    token_url=token_url,
    refresh_token=data["refresh_token"],
)

# To save the token, you will need to ensure it has the proper quotations around it and load into a JSON object before you can save it back into Redis with the value of token.

st_refreshed_token = '"{}"'.format(refreshed_token)
j_refreshed_token = json.loads(st_refreshed_token)
main.r.set("token", j_refreshed_token)

# After saving the newly refreshed token back into Redis, now you can obtain a new cat fact from the cat fact API, pass that into a JSON payload, and Tweet.

catty_fact = main.parse_cat_fact()
payload = {"text": "{}".format(catty_fact)}
main.post_tweet(payload, refreshed_token)